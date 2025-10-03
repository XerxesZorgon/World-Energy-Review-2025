#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
impute_coal.py
--------------
Impute missing Opening Year (opening_year_final) and initial reserves (reserves_initial_EJ)
in Coal_open_mines using a tiered strategy:

  1) Preserve/clean existing 'Opening Year' where present
  2) Heuristics: Closing Year – Life of mine; Production start year
  3) Lightweight model (IterativeImputer) on available numeric features
  4) Group medians by Country and Coal type; then global median
  5) Reserves_initial_EJ:
     a) Direct conversion from reserves mass columns (Mt) via convert_coal_to_EJ
     b) Rate × Life using Capacity/Production and life estimated from peer rows
     c) Group medians; then global median

Adds flags and provenance tags for each imputation.

Usage:
  python impute_coal.py               # expects Energy.db at ./data/Energy.db
  python impute_coal.py --db path/to/Energy.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# sklearn — keep it light (BayesianRidge via IterativeImputer)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

LOG = logging.getLogger("impute_coal")

# Optional unit conversion helper (safe fallbacks if missing)
try:
    from convert_coal_to_EJ import convert_coal_to_ej as coal_to_ej
except Exception as e:
    LOG.warning("convert_coal_to_EJ not available; falling back to default energy density. %s", e)

    def coal_to_ej(qty_mt: float, coal_type: str | None = None) -> float | None:  # type: ignore
        # Fallback default: 22.125 GJ/t ~ middle of common ranks
        try:
            return float(qty_mt) * 22.125 * 1e-3
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        LOG.info("[SCHEMA] Adding %s.%s %s", table, column, sql_type)
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        conn.commit()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def coerce_year(series: pd.Series) -> pd.Series:
    """Extract first 4-digit year from free text like '2024-2025' or 'Oct 2024 - Mar 2025'; clip to plausible range."""
    s = series.astype(str).str.extract(r"(\d{4})", expand=False)
    out = pd.to_numeric(s, errors="coerce").clip(lower=1850, upper=2100)
    return out


def first_existing(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_coal_type(val) -> str | None:
    if pd.isna(val):
        return None
    v = str(val).strip().lower()
    mapping = {
        "anthracite": "Anthracite",
        "bituminous": "Bituminous",
        "subbituminous": "Subbituminous",
        "sub-bituminous": "Subbituminous",
        "lignite": "Lignite",
        "bituminous and subbituminous": "Bituminous and Subbituminous",
        "anthracite & bituminous": "Anthracite & Bituminous",
        "subbituminous / lignite": "Subbituminous / Lignite",
    }
    return mapping.get(v, val if isinstance(val, str) else None)


# ---------------------------------------------------------------------------
# Feature engineering (Phase 2)
# ---------------------------------------------------------------------------

def _freq_encode(series: pd.Series) -> pd.Series:
    """Simple frequency encoding for categorical variables (robust, low-leakage)."""
    vc = series.fillna("<NA>").astype(str).value_counts(dropna=False)
    return series.fillna("<NA>").astype(str).map(vc).astype(float)


def make_year_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a compact feature matrix for opening year modeling.

    Uses available numeric columns and frequency encodes a few categoricals to avoid
    high-dimensional one-hot encoding.
    """
    out = pd.DataFrame(index=df.index)
    # Numeric signals
    for col in [
        "Closing Year", "Year of Production", "Capacity (Mt/y)", "Production (Mt/y)",
        "Latitude", "Longitude"
    ]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    # Derived life from heuristics if present
    if "Closing Year" in df.columns and "Year of Production" in df.columns:
        cy = pd.to_numeric(df["Closing Year"], errors="coerce")
        yp = pd.to_numeric(df["Year of Production"], errors="coerce")
        out["life_hint"] = (cy - yp).where((cy.notna()) & (yp.notna()))
    # Categorical frequency encodings
    if "Country / Area" in df.columns:
        out["freq_country"] = _freq_encode(df["Country / Area"])
    if "Coal type" in df.columns:
        out["freq_type"] = _freq_encode(df["Coal type"].map(normalize_coal_type))
    if "Mine Type" in df.columns:
        out["freq_mine"] = _freq_encode(df["Mine Type"])  # e.g., Underground/Surface
    return out


# ---------------------------------------------------------------------------
# Opening year imputation
# ---------------------------------------------------------------------------

def impute_opening_year(df: pd.DataFrame) -> pd.DataFrame:
    # Start from cleaned 'Opening Year'
    if "Opening Year" in df.columns:
        df["opening_year_final"] = coerce_year(df["Opening Year"]).astype("Int64")
    else:
        df["opening_year_final"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df["_imputed_by_year"] = np.where(df["opening_year_final"].notna(), "original", "")
    df["_imputed_flag_year"] = df["opening_year_final"].isna().astype(int)

    # Heuristic 1: Closing Year – Life of mine
    life_col = first_existing(df, "Life of mine (years)", "Life of mine", "Life (years)")
    close_col = first_existing(df, "Closing Year", "Closure Year")
    if life_col and close_col:
        mask = df["opening_year_final"].isna()
        oy = coerce_year(df[close_col]) - pd.to_numeric(df[life_col], errors="coerce")
        upd = mask & oy.notna()
        df.loc[upd, "opening_year_final"] = oy[upd].round().astype("Int64")
        df.loc[upd, "_imputed_by_year"] = "closing_minus_life"

    # Heuristic 2: Production start year
    prod_start_col = first_existing(df, "Production start year", "First production year")
    if prod_start_col:
        mask = df["opening_year_final"].isna()
        oy = coerce_year(df[prod_start_col])
        upd = mask & oy.notna()
        df.loc[upd, "opening_year_final"] = oy[upd].astype("Int64")
        df.loc[upd, "_imputed_by_year"] = "production_start"

    # Lightweight model on numeric features (only if needed)
    feat_candidates = [c for c in [close_col, life_col, prod_start_col, "Capacity (Mtpa)", "Production (Mtpa)"] if c]
    feat_num = [c for c in feat_candidates if c in df.columns]
    miss_mask = df["opening_year_final"].isna()
    country_col = first_existing(df, "Country / Area", "Country/Area", "Country")
    if miss_mask.any() and feat_num:
        try:
            sub = df.loc[:, feat_num + ([country_col] if country_col else [])].copy()
            # target encode country via current opening_year_final medians
            if country_col:
                enc = df.groupby(country_col, dropna=False)["opening_year_final"].median()
                sub["__enc_country"] = sub[country_col].map(enc)
                sub.drop(columns=[country_col], inplace=True)
            X = sub.apply(pd.to_numeric, errors="coerce")
            y = coerce_year(df.get("Opening Year", pd.Series(index=df.index)))
            fit_mask = y.notna() & (~X.isna().all(axis=1))
            if fit_mask.sum() >= 20:
                it = IterativeImputer(random_state=0)
                it.fit(X.loc[fit_mask], y.loc[fit_mask])
                pred = pd.Series(it.transform(X)[:, 0], index=df.index)
                pred = pred.round().clip(1850, 2100).astype("Int64")
                upd = miss_mask & pred.notna()
                df.loc[upd, "opening_year_final"] = pred[upd].astype("Int64")
                df.loc[upd, "_imputed_by_year"] = "iterative_numeric_model"
                df.loc[upd, "_confidence_year"] = df.loc[upd, "_imputed_by_year"].map(conf_year)

        except Exception as e:
            LOG.warning("Year model skipped: %s", e)

    # Phase 2: GradientBoostingRegressor per global (compact features) to catch remaining gaps
    # Train only on rows that have an ORIGINAL opening year (not imputed earlier)
    orig_col = first_existing(df, "Opening Year")
    if orig_col is not None:
        orig_year = coerce_year(df[orig_col])
        # Training set: rows where original is present and opening_year_final equals original (i.e., not changed)
        train_mask = orig_year.notna()
        X = make_year_features(df.loc[train_mask])
        y = orig_year.loc[train_mask]
        # Some guards
        if len(X) >= 100 and y.notna().sum() >= 100 and X.select_dtypes(include=[float, int]).shape[1] >= 3:
            X = X.fillna(X.median(numeric_only=True))
            y = y.astype(float)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(Xtr, ytr)
            # Residual std for uncertainty bands
            resid = yte - pd.Series(gbr.predict(Xte), index=yte.index)
            resid_std = float(np.nanstd(resid)) if len(resid) else 5.0

            # Predict for remaining NA
            rem_mask = df["opening_year_final"].isna()
            if rem_mask.any():
                Xp = make_year_features(df.loc[rem_mask]).fillna(X.median(numeric_only=True))
                yp = gbr.predict(Xp)
                # Clip to plausible range
                yp = np.clip(yp, 1850, 2100)
                df.loc[rem_mask, "opening_year_final"] = np.round(yp).astype("Int64")
                df.loc[rem_mask, "_imputed_by_year"] = "gbr_year_model"
                df.loc[rem_mask, "_confidence_year"] = df.loc[rem_mask, "_imputed_by_year"].map(conf_year)
                # Uncertainty bands (P10/P50/P90 assuming approx normal residuals)
                p50 = yp
                p10 = yp - 1.2816 * resid_std
                p90 = yp + 1.2816 * resid_std
                df.loc[rem_mask, "opening_year_P50"] = np.clip(p50, 1850, 2100)
                df.loc[rem_mask, "opening_year_P10"] = np.clip(p10, 1850, 2100)
                df.loc[rem_mask, "opening_year_P90"] = np.clip(p90, 1850, 2100)

    # Group medians by (Country), (Coal type), (Country×Coal type)
    coal_type_col = next((c for c in df.columns if "coal" in c.lower() and "type" in c.lower()), None)
    for grp_cols in ([country_col], [coal_type_col], [country_col, coal_type_col]):
        cols = [c for c in grp_cols if c]
        if not cols:
            continue
        med = df.groupby(cols, dropna=False)["opening_year_final"].median()
        mask = df["opening_year_final"].isna()
        if med.notna().any() and mask.any():
            key_index = df.set_index(cols).index
            fill = key_index.map(med)
            upd = mask & pd.notna(fill)
            df.loc[upd, "opening_year_final"] = pd.Series(fill, index=df.index)[upd].astype("Int64")
            df.loc[upd, "_imputed_by_year"] = "group_median:" + "×".join(cols)

    # Global median fallback
    if df["opening_year_final"].isna().any():
        glob = pd.to_numeric(df.get("Opening Year"), errors="coerce").median()
        if pd.notna(glob):
            upd = df["opening_year_final"].isna()
            df.loc[upd, "opening_year_final"] = int(round(glob))
            df.loc[upd, "_imputed_by_year"] = np.where(
                df.loc[upd, "_imputed_by_year"] == "", "global_median", df.loc[upd, "_imputed_by_year"]
            )

    # Final flags (changed or originally missing)
    orig_year = df.get("Opening Year", pd.Series([pd.NA]*len(df), dtype="object"))
    df["_imputed_flag_year"] = (
        orig_year.isna()
        | (coerce_year(orig_year).astype("float") != df["opening_year_final"].astype("float"))
    ).astype(int)

    # Optional confidence tag (coarse)
    def conf_year(tag: str) -> str:
        if tag == "original":
            return "high"
        if tag in ("closing_minus_life", "production_start", "iterative_model"):
            return "medium"
        return "low"

    df["_confidence_year"] = df["_imputed_by_year"].fillna("").map(conf_year)

    return df


# ---------------------------------------------------------------------------
# Reserves (EJ) imputation
# ---------------------------------------------------------------------------

def convert_mt_to_ej(mt: float | None, coal_type_val) -> float | None:
    if mt is None or pd.isna(mt):
        return None
    try:
        key = normalize_coal_type(coal_type_val)
        # Handle case where key is None (use fallback conversion)
        if key is None:
            # Use default energy density: 22.125 GJ/t
            return float(mt) * 22.125 * 1e-3
        val = coal_to_ej(float(mt), key)  # may return None
        return float(val) if val is not None and not pd.isna(val) else None
    except Exception:
        return None


def impute_reserves(df: pd.DataFrame) -> pd.DataFrame:
    # Identify a reserves mass column (Mt)
    reserves_cols = [c for c in df.columns if "reserve" in c.lower() and "mt" in c.lower()]
    res_col = reserves_cols[0] if reserves_cols else None

    country_col = first_existing(df, "Country / Area", "Country/Area", "Country")
    coal_type_col = next((c for c in df.columns if "coal" in c.lower() and "type" in c.lower()), None)

    # A) Direct conversion where reserves mass exists
    qty_ej = pd.Series(np.nan, index=df.index, dtype=float)
    if res_col:
        qty_ej = df.apply(
            lambda r: convert_mt_to_ej(
                pd.to_numeric(r.get(res_col), errors="coerce"),
                r.get(coal_type_col) if coal_type_col else None,
            ),
            axis=1,
        ).astype(float)

    # B) Rate × Life using Capacity/Production and life estimated from peers
    cap_col = first_existing(df, "Capacity (Mtpa)", "Installed capacity (Mtpa)")
    prod_col = first_existing(df, "Production (Mtpa)", "Annual production (Mtpa)")

    life_from_cap = None
    life_from_prod = None
    if res_col and cap_col and cap_col in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            life_from_cap = (
                pd.to_numeric(df[res_col], errors="coerce")
                / pd.to_numeric(df[cap_col], errors="coerce")
            ).replace([np.inf, -np.inf], np.nan)

    if res_col and prod_col and prod_col in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            life_from_prod = (
                pd.to_numeric(df[res_col], errors="coerce")
                / pd.to_numeric(df[prod_col], errors="coerce")
            ).replace([np.inf, -np.inf], np.nan)

    life_est = pd.Series(np.nan, index=df.index, dtype=float)
    life_src_cols = []
    if life_from_cap is not None:
        df["__life_cap"] = life_from_cap
        life_src_cols.append("__life_cap")
    if life_from_prod is not None:
        df["__life_prod"] = life_from_prod
        life_src_cols.append("__life_prod")

    if life_src_cols:
        life_base = df[life_src_cols].median(axis=1, skipna=True)
        life_est = life_base.fillna(life_base.median())

    def estimate_from_rate(rate_mtpa, ctype_val, life_years):
        if pd.isna(rate_mtpa) or pd.isna(life_years):
            return np.nan
        mt_total = float(rate_mtpa) * float(life_years)
        return convert_mt_to_ej(mt_total, ctype_val)

    if qty_ej.isna().any():
        if cap_col and cap_col in df.columns:
            est_cap = df.apply(
                lambda r: estimate_from_rate(
                    pd.to_numeric(r.get(cap_col), errors="coerce"),
                    r.get(coal_type_col) if coal_type_col else None,
                    life_est.loc[r.name] if life_est is not None else np.nan,
                ),
                axis=1,
            )
            qty_ej = qty_ej.fillna(est_cap)
        if prod_col and prod_col in df.columns:
            est_prod = df.apply(
                lambda r: estimate_from_rate(
                    pd.to_numeric(r.get(prod_col), errors="coerce"),
                    r.get(coal_type_col) if coal_type_col else None,
                    life_est.loc[r.name] if life_est is not None else np.nan,
                ),
                axis=1,
            )
            qty_ej = qty_ej.fillna(est_prod)

    # C) Group medians by (Country), (Coal type), (Country×Coal type)
    for grp_cols in ([country_col], [coal_type_col], [country_col, coal_type_col]):
        cols = [c for c in grp_cols if c]
        if not cols:
            continue
        med = df.assign(__ej=qty_ej).groupby(cols, dropna=False)["__ej"].median()
        idx = qty_ej.isna()
        if med.notna().any() and idx.any():
            key_index = df.set_index(cols).index
            fill = key_index.map(med)
            qty_ej.loc[idx & pd.notna(fill)] = pd.Series(fill, index=df.index)[idx & pd.notna(fill)]

    # D) Global median fallback
    if qty_ej.isna().any():
        qty_ej = qty_ej.fillna(np.nanmedian(qty_ej.values))

    # Assign and flags
    prev_ej = df.get("reserves_initial_EJ")
    df["reserves_initial_EJ"] = qty_ej.astype(float)

    # Method tagging: track during assignment rather than post-hoc
    df["_imputed_by_qty"] = ""
    
    # Tag direct reserves conversions
    if res_col:
        direct_mask = pd.to_numeric(df[res_col], errors="coerce").notna()
        df.loc[direct_mask, "_imputed_by_qty"] = np.where(
            df.loc[direct_mask, coal_type_col].notna() if coal_type_col else False,
            "direct_reserves_convert",
            "reserves_no_type_default_density",
        )
    
    # Tag rate×life estimates (check if we used capacity or production)
    rate_mask = df["_imputed_by_qty"] == ""
    if cap_col and cap_col in df.columns:
        used_cap = rate_mask & pd.to_numeric(df[cap_col], errors="coerce").notna() & life_est.notna()
        df.loc[used_cap, "_imputed_by_qty"] = "rate_x_life_capacity"
        rate_mask = rate_mask & ~used_cap
    if prod_col and prod_col in df.columns:
        used_prod = rate_mask & pd.to_numeric(df[prod_col], errors="coerce").notna() & life_est.notna()
        df.loc[used_prod, "_imputed_by_qty"] = "rate_x_life_production"
        rate_mask = rate_mask & ~used_prod
    
    # Remaining are group/global medians
    df.loc[rate_mask & df["reserves_initial_EJ"].notna(), "_imputed_by_qty"] = "group_or_global_median"

    df["_imputed_flag_qty"] = (
        (prev_ej.isna() if prev_ej is not None else True)
        | (prev_ej.ne(df["reserves_initial_EJ"]) if prev_ej is not None else True)
    ).astype(int)

    # Optional confidence tag (coarse)
    def conf_qty(tag: str) -> str:
        if tag in ("direct_reserves_convert",):
            return "high"
        if tag in ("reserves_no_type_default_density", "rate_x_life_capacity", "rate_x_life_production"):
            return "medium"
        return "low"

    df["_confidence_qty"] = df["_imputed_by_qty"].fillna("").map(conf_qty)

    # Cleanup temp cols
    for t in ["__life_cap", "__life_prod"]:
        if t in df.columns:
            del df[t]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ensure_output_columns(conn: sqlite3.Connection) -> None:
    tbl = "Coal_open_mines"
    add_column_if_missing(conn, tbl, "opening_year_final", "INTEGER")
    add_column_if_missing(conn, tbl, "_imputed_flag_year", "INTEGER")
    add_column_if_missing(conn, tbl, "_imputed_by_year", "TEXT")
    add_column_if_missing(conn, tbl, "_confidence_year", "TEXT")
    add_column_if_missing(conn, tbl, "opening_year_P10", "REAL")
    add_column_if_missing(conn, tbl, "opening_year_P50", "REAL")
    add_column_if_missing(conn, tbl, "opening_year_P90", "REAL")
    add_column_if_missing(conn, tbl, "reserves_initial_EJ", "REAL")
    add_column_if_missing(conn, tbl, "_imputed_flag_qty", "INTEGER")
    add_column_if_missing(conn, tbl, "_imputed_by_qty", "TEXT")
    add_column_if_missing(conn, tbl, "_confidence_qty", "TEXT")


def validate_results(df: pd.DataFrame) -> None:
    """Basic validation checks for imputed data."""
    # Check for impossible years
    if "opening_year_final" in df.columns:
        bad_years = ~df["opening_year_final"].between(1850, 2100, inclusive="both")
        bad_years = bad_years & df["opening_year_final"].notna()  # Only check non-null values
        if bad_years.any():
            LOG.warning("Found %d rows with implausible years", bad_years.sum())
    
    # Check for negative reserves
    if "reserves_initial_EJ" in df.columns:
        neg_reserves = (df["reserves_initial_EJ"] < 0) & df["reserves_initial_EJ"].notna()
        if neg_reserves.any():
            LOG.warning("Found %d rows with negative reserves", neg_reserves.sum())


def run(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        ensure_output_columns(conn)

        tbl = "Coal_open_mines"
        df = pd.read_sql(f"SELECT * FROM {tbl}", conn)

        # Impute fields
        df = impute_opening_year(df)
        df = impute_reserves(df)
        
        # Validate results
        validate_results(df)

        # Write back (replace keeps column order of DataFrame)
        df.to_sql(tbl, conn, if_exists="replace", index=False)

        # Coverage report
        year_cov = 100.0 * (1.0 - df["opening_year_final"].isna().mean())
        qty_cov = 100.0 * (1.0 - df["reserves_initial_EJ"].isna().mean())
        print(f"[OK] {tbl}: {len(df)} rows written. Coverage → year {year_cov:.1f}%, qty {qty_cov:.1f}%")
        LOG.info("Finished imputation for %s", tbl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data") / "Energy.db",
                        help="Path to Energy.db (default: ./data/Energy.db)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--log-file", default="impute_coal.log",
                        help="Log file path (default: impute_coal.log)")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run(args.db)


if __name__ == "__main__":
    main()
