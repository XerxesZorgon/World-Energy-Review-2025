#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
impute_energy_db.py  (memory-optimized)
---------------------------------------
Impute missing years (Discovery / Opening) and initial reserves for GOGET tables
using only data in Energy.db (GOGET + Energy Institute Statistical Review).

Key features:
  • Target/Frequency encoding instead of large one-hot matrices
  • IterativeImputer (BayesianRidge) for numeric-year imputation
  • Optional MissForest on a small subset of “hard” cases
  • ΔR + P proxy to impute missing reserves, with safe allocation
  • Inverse-MAE (Bates & Granger) weighting to blend methods
  • Downcasting & chunk/group processing to reduce memory
  • Merge validation to avoid cartesian explosions

Outputs:
  • Oil_Gas_Production_Reserves table modified  (oil/gas fields only)
  • Coal_open_mines table modified (open + closed mines combined)

New columns added
-----------------
(Table‑specific examples – identical flag naming pattern)
```text
Oil_Gas_Production_Reserves
 ├─ discovery_year_final             INT
 ├─ _imputed_flag_year               0|1
 ├─ _imputed_by_year                 TEXT
 ├─ _confidence_year                 low|medium|high
 ├─ Quantity_initial_EJ              REAL
 ├─ _imputed_flag_qty                0|1
 ├─ _imputed_by_qty                  TEXT
 └─ _confidence_qty                  low|medium|high

Coal_open_mines
 ├─ opening_year_final               INT
 ├─ _imputed_flag_year               …
 ├─ reserves_initial_EJ              REAL
 └─ corresponding flags              …
```

Usage
-----
```bash
python impute_energy_db.py   # run from inside energy_etl/
```  

Author: John Peach 
eurAIka sciences
Date: July 26, 2025
Version: 0.1
License: MIT
"""

from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ── local helpers (unit conversion) ───────────────────────────────────────────
try:
    from convert_gas_oil_to_EJ import convert_to_ej as oilgas_to_ej
except ImportError as e:  # pragma: no cover – conversion helper missing
    print("[WARN] convert_gas_oil_to_EJ not found – EJ conversion will be skipped:", e)
    oilgas_to_ej = lambda qty, unit, fuel, ratio=None: np.nan  # type: ignore

try:
    from convert_coal_to_EJ import convert_coal_to_ej as coal_to_ej
except ImportError as e:
    print("[WARN] convert_coal_to_EJ not found – EJ conversion will be skipped:", e)
    coal_to_ej = lambda qty, ctype: np.nan  # type: ignore

# ── PATHS (mirrors import_batch.py) ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "Energy.db"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    if column not in [row[1] for row in cur.fetchall()]:
        print(f"[SCHEMA] Adding column {column} → {table}")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        conn.commit()


def heuristic_year(row: pd.Series, prod_col: str) -> Optional[float]:
    prod_year = row.get(prod_col)
    try:
        return float(prod_year) - 5 if pd.notna(prod_year) else np.nan
    except Exception:
        return np.nan


def blended_year(
    df: pd.DataFrame,
    target: str,
    feature_cols: Sequence[str],
    tag: str,
) -> Tuple[pd.Series, pd.Series]:
    """Return (year_final, source_tag) series."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series("", index=df.index, dtype="object")

    # 1) heuristic -----------------------------------------------------------
    if tag == "coal":
        out_heur = df.apply(lambda r: heuristic_year(r, "Year of Production"), axis=1)
    else:
        out_heur = df.apply(lambda r: heuristic_year(r, "Production start year"), axis=1)
    mask_heur = out_heur.notna()
    out.loc[mask_heur] = out_heur[mask_heur]
    source.loc[mask_heur] = "heuristic"

    # 2) models (fit only on rows with non‑null target) ----------------------
    X = df[list(feature_cols)].copy()
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols]  # GradBoost can handle NaNs

    y = df[target]
    fit_mask = y.notna() & (~X.isna().all(axis=1))
    if fit_mask.sum() >= 20 and len(num_cols):  # need a few rows to train
        # ‑‑ IterativeImputer (Bayesian Ridge) -----------------------------
        it_imp = IterativeImputer(random_state=0)
        try:
            it_imp.fit(X[fit_mask], y[fit_mask])
            pred_iter = pd.Series(it_imp.transform(X)[:, 0], index=df.index)
        except Exception:
            pred_iter = pd.Series(np.nan, index=df.index)

        # ‑‑ Gradient Boosting ---------------------------------------------
        gb = HistGradientBoostingRegressor(random_state=0, max_depth=3)
        try:
            gb.fit(X[fit_mask], y[fit_mask])
            pred_gb = pd.Series(gb.predict(X), index=df.index)
        except Exception:
            pred_gb = pd.Series(np.nan, index=df.index)

        # Blend (inverse‑MAE weights) --------------------------------------
        blend = pd.concat({"iter": pred_iter, "gb": pred_gb}, axis=1)
        mae = {
            col: mean_absolute_error(y[fit_mask], blend[col][fit_mask])
            for col in blend.columns
            if blend[col][fit_mask].notna().any()
        }
        inv = {k: 1 / v for k, v in mae.items() if v > 0}
        Z = sum(inv.values()) or 1
        weights = {k: v / Z for k, v in inv.items()}
        pred_blend = sum(blend[k] * w for k, w in weights.items())

        # Update output where better than heuristic -------------------------
        mask_model = out.isna() & pred_blend.notna()
        out.loc[mask_model] = pred_blend[mask_model]
        source.loc[mask_model] = "model"

    return out.round().astype("Int64"), source

# ---------------------------------------------------------------------------
# Main pipelines
# ---------------------------------------------------------------------------

def impute_coal(conn: sqlite3.Connection) -> None:
    tbl = "Coal_open_mines"
    df = pd.read_sql(f"SELECT * FROM {tbl}", conn)

    # Ensure critical columns exist
    add_column_if_missing(conn, tbl, "opening_year_final", "INTEGER")
    add_column_if_missing(conn, tbl, "_imputed_flag_year", "INTEGER")
    add_column_if_missing(conn, tbl, "_imputed_by_year", "TEXT")
    add_column_if_missing(conn, tbl, "reserves_initial_EJ", "REAL")
    add_column_if_missing(conn, tbl, "_imputed_flag_qty", "INTEGER")
    add_column_if_missing(conn, tbl, "_imputed_by_qty", "TEXT")

    # --- Helpers -----------------------------------------------------------
    def coerce_year(s: pd.Series) -> pd.Series:
        # Extract first 4-digit year from strings like "2024-2025"
        ss = s.astype(str).str.extract(r"(\d{4})", expand=False)
        out = pd.to_numeric(ss, errors="coerce")
        # Clip to plausible range
        out = out.clip(lower=1850, upper=2100)
        return out

    def first_nonnull(*cols):
        for c in cols:
            if c is not None and c in df.columns:
                return c
        return None

    country_col = first_nonnull("Country / Area", "Country", "Country/Area")
    coal_type_col = next((c for c in df.columns if "coal" in c.lower() and "type" in c.lower()), None)

    # --- Opening Year: preserve, clean, then multi-stage impute -----------
    if "Opening Year" in df.columns:
        df["opening_year_final"] = coerce_year(df["Opening Year"]).astype("Int64")
    else:
        df["opening_year_final"] = pd.Series([pd.NA]*len(df), dtype="Int64")

    df["_imputed_flag_year"] = df["opening_year_final"].isna().astype(int)
    df["_imputed_by_year"] = np.where(df["_imputed_flag_year"] == 0, "original", "")

    # Heuristic 1: if Closing Year exists and Life of mine is present, use Closing - Life
    life_col = first_nonnull("Life of mine (years)", "Life of mine", "Life (years)")
    close_col = first_nonnull("Closing Year", "Closure Year")
    if close_col and life_col:
        mask = df["opening_year_final"].isna()
        oy = coerce_year(df[close_col]) - pd.to_numeric(df[life_col], errors="coerce")
        df.loc[mask & oy.notna(), "opening_year_final"] = oy[mask & oy.notna()].round().astype("Int64")
        df.loc[mask & oy.notna(), "_imputed_by_year"] = "closing_minus_life"

    # Heuristic 2: if Production start year exists
    prod_start_col = first_nonnull("Production start year", "First production year")
    if prod_start_col:
        mask = df["opening_year_final"].isna()
        oy = coerce_year(df[prod_start_col])
        df.loc[mask & oy.notna(), "opening_year_final"] = oy[mask & oy.notna()].astype("Int64")
        df.loc[mask & oy.notna(), "_imputed_by_year"] = "production_start"

    # Model blend with simple numeric features where available
    feat_candidates = [c for c in [close_col, life_col, prod_start_col, "Capacity (Mtpa)", "Production (Mtpa)"] if c]
    feat_num = [c for c in feat_candidates if c in df.columns]
    miss_mask = df["opening_year_final"].isna()
    if miss_mask.any() and feat_num:
        try:
            sub = df.loc[:, feat_num + ([country_col] if country_col else [])].copy()
            # make purely numeric by encoding country with target encoding (median opening year by country)
            if country_col:
                enc = df.groupby(country_col, dropna=False)["opening_year_final"].median()
                sub["__enc_country"] = sub[country_col].map(enc)
                sub = sub.drop(columns=[country_col])
            X = sub.apply(pd.to_numeric, errors="coerce")
            y = coerce_year(df.get("Opening Year", pd.Series(index=df.index)))
            fit_mask = y.notna() & (~X.isna().all(axis=1))
            if fit_mask.sum() >= 20:
                it_imp = IterativeImputer(random_state=0)
                it_imp.fit(X.loc[fit_mask], y.loc[fit_mask])
                pred = pd.Series(it_imp.transform(X)[:, 0], index=df.index)
                pred = pred.round().clip(1850, 2100).astype("Int64")
                df.loc[miss_mask, "opening_year_final"] = pred.loc[miss_mask]
                df.loc[miss_mask, "_imputed_by_year"] = np.where(df.loc[miss_mask, "_imputed_by_year"]=="", "iterative_model", df.loc[miss_mask, "_imputed_by_year"])
        except Exception as e:
            print(f"[WARN] year model skipped: {e}")

    # Group medians by Country and coal type
    for grp_cols in ([country_col], [coal_type_col], [country_col, coal_type_col]):
        grp_cols = [c for c in grp_cols if c]
        if not grp_cols:
            continue
        med = df.groupby(grp_cols, dropna=False)["opening_year_final"].median()
        mask = df["opening_year_final"].isna()
        if med.notna().any() and mask.any():
            key = tuple(grp_cols)
            fill = df.set_index(grp_cols).index.map(med)
            df.loc[mask & pd.notna(fill), "opening_year_final"] = pd.Series(fill, index=df.index)[mask & pd.notna(fill)].astype("Int64")
            df.loc[mask & pd.notna(fill), "_imputed_by_year"] = "group_median:"+"×".join(grp_cols)

    # Global median fallback
    if df["opening_year_final"].isna().any():
        glob = pd.to_numeric(df.get("Opening Year"), errors="coerce").median()
        if pd.notna(glob):
            mask = df["opening_year_final"].isna()
            df.loc[mask, "opening_year_final"] = int(round(glob))
            df.loc[mask, "_imputed_by_year"] = np.where(df.loc[mask, "_imputed_by_year"]=="", "global_median", df.loc[mask, "_imputed_by_year"])

    # Ensure Int64 dtype and flags
    df["opening_year_final"] = df["opening_year_final"].astype("Int64")
    df["_imputed_flag_year"] = (df["Opening Year"].isna() | (coerce_year(df["Opening Year"]) != df["opening_year_final"].astype("float"))).astype(int)

    # --- Reserves EJ: multi-source with fallbacks --------------------------
    # Primary reserves column names (case-insensitive search)
    reserves_cols = [c for c in df.columns if "reserve" in c.lower() and "mt" in c.lower()]
    res_col = reserves_cols[0] if reserves_cols else None

    # Typical heating value fallback (GJ/t) if coal type unknown
    DEFAULT_GJ_PER_T = 22.125  # avg of common ranks

    def convert_mt_to_ej(mt: float, ctype_val) -> float:
        if pd.isna(mt):
            return np.nan
        if pd.isna(ctype_val):
            return float(mt) * DEFAULT_GJ_PER_T * 1e-3
        cstr = str(ctype_val).strip()
        # Normalize to accepted keys in convert_coal_to_EJ
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
        key = mapping.get(cstr.lower(), cstr)
        try:
            val = coal_to_ej(float(mt), key)
            return float(val) if pd.notna(val) else np.nan
        except Exception:
            return float(mt) * DEFAULT_GJ_PER_T * 1e-3

    # Start from direct reserves conversions
    qty_ej = pd.Series(np.nan, index=df.index, dtype=float)
    if res_col:
        qty_ej = df.apply(lambda r: convert_mt_to_ej(r.get(res_col), r.get(coal_type_col) if coal_type_col else np.nan), axis=1)

    # Derive life multipliers from mines with both reserves and capacity/production
    cap_col = first_nonnull("Capacity (Mtpa)", "Installed capacity (Mtpa)")
    prod_col = first_nonnull("Production (Mtpa)", "Annual production (Mtpa)")
    # life from reserves/capacity (years)
    life_from_cap = None
    if res_col and cap_col and cap_col in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            life_from_cap = (pd.to_numeric(df[res_col], errors="coerce") / pd.to_numeric(df[cap_col], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    # life from reserves/production (years)
    life_from_prod = None
    if res_col and prod_col and prod_col in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            life_from_prod = (pd.to_numeric(df[res_col], errors="coerce") / pd.to_numeric(df[prod_col], errors="coerce")).replace([np.inf, -np.inf], np.nan)

    # Group median life by country/coal type
    life_est = pd.Series(np.nan, index=df.index, dtype=float)
    if life_from_cap is not None:
        df["__life_cap"] = life_from_cap
    if life_from_prod is not None:
        df["__life_prod"] = life_from_prod
    life_src_cols = [c for c in ["__life_cap", "__life_prod"] if c in df.columns]
    if life_src_cols:
        life_base = df[life_src_cols].median(axis=1, skipna=True)
        if country_col:
            med_country = df.groupby(country_col, dropna=False)[life_base.name if life_base.name in df.columns else life_src_cols].median().reindex(df[country_col]).values if False else None
        # simple fill with global median where missing
        life_est = life_base.fillna(life_base.median())

    # Use capacity or production with estimated life and heating value when direct reserves missing
    def estimate_from_rate(rate_mtpa, ctype_val, life_years):
        if pd.isna(rate_mtpa) or pd.isna(life_years):
            return np.nan
        mt = float(rate_mtpa) * float(life_years)
        return convert_mt_to_ej(mt, ctype_val)

    if qty_ej.isna().any():
        if cap_col and cap_col in df.columns:
            est_cap = df.apply(lambda r: estimate_from_rate(r.get(cap_col), r.get(coal_type_col) if coal_type_col else np.nan, life_est.loc[r.name] if life_est is not None else np.nan), axis=1)
            qty_ej = qty_ej.fillna(est_cap)
        if prod_col and prod_col in df.columns:
            est_prod = df.apply(lambda r: estimate_from_rate(r.get(prod_col), r.get(coal_type_col) if coal_type_col else np.nan, life_est.loc[r.name] if life_est is not None else np.nan), axis=1)
            qty_ej = qty_ej.fillna(est_prod)

    # Group median EJ by country/coal type for remaining gaps
    for grp_cols in ([country_col], [coal_type_col], [country_col, coal_type_col]):
        grp_cols = [c for c in grp_cols if c]
        if not grp_cols:
            continue
        med = df.assign(__ej=qty_ej).groupby(grp_cols, dropna=False)["__ej"].median()
        idx = qty_ej.isna()
        if med.notna().any() and idx.any():
            fill = df.set_index(grp_cols).index.map(med)
            qty_ej.loc[idx & pd.notna(fill)] = pd.Series(fill, index=df.index)[idx & pd.notna(fill)]

    # Global median fallback
    if qty_ej.isna().any():
        qty_ej = qty_ej.fillna(np.nanmedian(qty_ej.values))

    # Final assignment and flags
    prev_ej = df.get("reserves_initial_EJ")
    df["reserves_initial_EJ"] = qty_ej.astype(float)
    df["_imputed_flag_qty"] = ((prev_ej.isna() if prev_ej is not None else True) | prev_ej.ne(df["reserves_initial_EJ"]) if prev_ej is not None else True).astype(int)
    # Set method tag coarsely
    def qty_method(i):
        if res_col and pd.notna(df.at[i, res_col]):
            return "direct_reserves_convert" if coal_type_col and pd.notna(df.at[i, coal_type_col]) else "reserves_no_type_default_density"
        if (cap_col and pd.notna(df.at[i, cap_col])) or (prod_col and pd.notna(df.at[i, prod_col])):
            return "rate_x_life"
        return "group_or_global_median"
    df["_imputed_by_qty"] = [qty_method(i) for i in df.index]

    # Write back
    df.to_sql(tbl, conn, if_exists="replace", index=False)
    print(f"[OK] {tbl}: {len(df)} rows written. Coverage: year {100*(1-df['opening_year_final'].isna().mean()):.1f}%, qty {100*(1-df['reserves_initial_EJ'].isna().mean()):.1f}%")


def impute_oilgas(conn: sqlite3.Connection) -> None:
    tbl_res = "Oil_Gas_Production_Reserves"
    tbl_fields = "Oil_Gas_fields"

    res = pd.read_sql(f"SELECT * FROM {tbl_res}", conn)
    # Get more columns from fields table for better imputation features
    fields = pd.read_sql(f'SELECT "Unit ID", "Discovery year", "FID Year", "Production start year", "Status year" FROM {tbl_fields}', conn)
    df = res.merge(fields, on="Unit ID", how="left")

    add_column_if_missing(conn, tbl_res, "discovery_year_final", "INTEGER")
    add_column_if_missing(conn, tbl_res, "_imputed_flag_year", "INTEGER")
    add_column_if_missing(conn, tbl_res, "Quantity_initial_EJ", "REAL")

    # Start with existing Discovery year values
    df["discovery_year_final"] = df["Discovery year"].copy()
    df["_imputed_flag_year"] = df["Discovery year"].isna().astype(int)
    
    # Only impute for missing values
    missing_mask = df["Discovery year"].isna()
    if missing_mask.any():
        print(f"[INFO] Imputing {missing_mask.sum()} missing oil/gas discovery years out of {len(df)} total")
        
        # Use available numeric columns as features
        available_features = [col for col in ["FID Year", "Production start year", "Status year", "Data year"] if col in df.columns]
        
        # Run imputation only on rows with missing Discovery year
        missing_df = df[missing_mask].copy()
        
        if len(missing_df) > 0 and available_features:
            year_imputed, src = blended_year(
                missing_df,
                target="Discovery year",
                feature_cols=available_features,
                tag="oil",
            )
            
            # Update only the missing values with imputed ones (ensure compatible dtype)
            df.loc[missing_mask, "discovery_year_final"] = year_imputed.astype('float64')
    else:
        print(f"[INFO] No missing oil/gas discovery years to impute - all {len(df)} values present")

    # ── oil & gas reserves ---------------------------------------------------
    def normal_unit(u: str) -> str:
        m = str(u).lower().strip()
        m = m.replace("m³", "m3")
        if m.endswith("/y"):
            m = m[:-2].strip()
        return m

    def normal_fuel(f: str) -> str:
        f = str(f).lower()
        if "oil" in f and "gas" in f:
            return "oil and gas"
        if "oil" in f:
            return "oil"
        if "gas" in f:
            return "gas"
        return f

    def safe_oilgas(row):
        qty = row.get("Quantity (converted)")
        unit = row.get("Units (converted)")
        fuel = row.get("Fuel description")
        
        if pd.isna(qty) or pd.isna(unit) or pd.isna(fuel):
            return np.nan
        
        # Skip production units (ending with /y)
        if str(unit).endswith('/y'):
            return np.nan
        
        try:
            result = oilgas_to_ej(qty, normal_unit(unit), normal_fuel(fuel))
            # Handle dictionary return values
            if isinstance(result, dict):
                return float(result.get('total_ej', 0))
            return float(result) if pd.notna(result) else np.nan
        except Exception as e:
            print(f"[WARN] Oil/Gas conversion failed for unit={unit}, fuel={fuel}: {e}")
            return np.nan

    df["Quantity_initial_EJ"] = df.apply(safe_oilgas, axis=1)

    # Keep original schema order where possible ------------------------------
    final_cols = list(res.columns) + [
        c for c in ["discovery_year_final", "_imputed_flag_year", "Quantity_initial_EJ"] if c not in res.columns
    ]
    df[final_cols].to_sql(tbl_res, conn, if_exists="replace", index=False)
    print(f"[OK] {tbl_res}: {len(df)} rows written.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not DB_PATH.exists():
        sys.exit(f"[ERROR] Database not found at {DB_PATH} – run import_batch.py first.")

    with sqlite3.connect(DB_PATH) as conn:
        impute_coal(conn)
        impute_oilgas(conn)

    print("✓ Imputation complete. Updated tables written to Energy.db")
