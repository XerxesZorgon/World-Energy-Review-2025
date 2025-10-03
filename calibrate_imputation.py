#!/usr/bin/env python3
"""
calibrate_imputation.py
-----------------------
Post-imputation calibration to better align imputed quantity distributions
with observed ones.

Strategy (per table, per stratum):
- Work on log10 of the effective quantity column
- Compute observed mean/sd and p1–p99 quantiles per stratum
- For imputed rows: mean/variance match on log-scale; then clip to observed p1–p99
- Write back calibrated values into a calibrated quantity column
- Mark _calibrated_flag_quantity = 1 and _calibrated_method_quantity

Strata: Country/Area × discovery decade

Usage:
  python calibrate_imputation.py --tables Oil Gas Coal
  python calibrate_imputation.py --tables Coal

Notes:
- Non-destructive: writes to new column Quantity_initial_EJ_calibrated
- If observed stats are insufficient in a stratum, falls back to broader strata
- Requires _imputed_flag_quantity in the table
"""

import argparse
import sqlite3
import os
from typing import List, Tuple
import numpy as np
import pandas as pd

# Paths (align with other scripts)
SCRIPT_DIR = os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'Energy.db')

# Column resolution helpers ----------------------------------------------------

def normalize_name(s: str) -> str:
    return ''.join(ch.lower() for ch in s if ch.isalnum())


def resolve_column(cols: List[str], candidates: List[str]) -> str:
    """Resolve a column by candidate names using normalized comparison."""
    norm = {normalize_name(c): c for c in cols}
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm:
            return norm[key]
    # try partial contains
    for c in cols:
        if any(k in normalize_name(c) for k in [normalize_name(x) for x in candidates]):
            return c
    raise KeyError(f"None of {candidates} found in columns: {cols}")


def ensure_columns(conn: sqlite3.Connection, table: str, columns: List[Tuple[str, str]]):
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    existing = {row[1] for row in cur.fetchall()}
    for name, sqltype in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")


def add_decade(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    return (np.floor(s / 10) * 10).astype('Int64')


# Core calibration -------------------------------------------------------------

def calibrate_table(conn: sqlite3.Connection, table: str, method: str = "quantile_map") -> None:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    if df.empty:
        print(f"[info] {table}: empty, skipping")
        return

    # Resolve key columns
    cols = list(df.columns)
    qty_col = resolve_column(cols, [
        "Quantity_initial_EJ", "quantity_initial_ej", "quantity ej",
        "reserves_initial_EJ", "reserves initial ej", "reserves_ej"
    ])
    flag_col = resolve_column(cols, ["_imputed_flag_quantity", "imputed_flag_quantity", "_imputed_flag_qty"])
    country_col = resolve_column(cols, ["Country / Area", "country/area", "country_area", "country"])
    # discovery year may be absent; treat as optional
    try:
        year_col = resolve_column(cols, [
            "discovery_year_final", "Discovery year (final)", "discoveryyearfinal",
            "opening_year_final", "Opening Year", "openingyearfinal"
        ])
    except KeyError:
        year_col = None

    # Special handling for Coal: when calibrating reserves_initial_EJ,
    # use year flag for observed vs imputed separation (quantity flag may be 0 after reruns)
    try:
        if normalize_name(qty_col) == normalize_name("reserves_initial_EJ"):
            year_flag_col = resolve_column(cols, ["_imputed_flag_year", "imputed_flag_year"])  # may raise
            flag_col = year_flag_col
    except Exception:
        pass

    # Work frame
    work = df[[qty_col, flag_col, country_col] + ([year_col] if year_col else [])].copy()
    work = work.loc[work[qty_col].notna() & (work[qty_col] > 0)].copy()
    if work.empty:
        print(f"[warn] {table}: no positive quantities; skipping")
        return

    # log-scale
    work["logQ"] = np.log10(work[qty_col])

    # Decade
    if year_col is not None:
        work["decade"] = add_decade(work[year_col])
    else:
        work["decade"] = pd.Series([pd.NA] * len(work), dtype="Int64")

    # Strata: country × decade
    strata = [country_col, "decade"]

    # Compute observed stats per stratum
    obs = work.loc[work[flag_col] == 0].copy()
    imp = work.loc[work[flag_col] != 0].copy()

    if imp.empty:
        print(f"[info] {table}: no imputed rows; nothing to calibrate")
        return

    # Per-stratum observed stats and full lists for quantile mapping
    stats_obs = (
        obs.groupby(strata)["logQ"]
        .agg(
            mean_obs="mean",
            sd_obs="std",
            p1=lambda s: s.quantile(0.01),
            p99=lambda s: s.quantile(0.99),
            n_obs="size",
            obs_list=lambda s: s.sort_values().to_list(),
        )
        .reset_index()
    )

    # Fallback: country-only stats
    stats_obs_country = (
        obs.groupby([country_col])["logQ"]
        .agg(
            mean_obs_c="mean",
            sd_obs_c="std",
            p1_c=lambda s: s.quantile(0.01),
            p99_c=lambda s: s.quantile(0.99),
            n_obs_c="size",
            obs_list_c=lambda s: s.sort_values().to_list(),
        )
        .reset_index()
    )

    # Global stats as last resort
    mean_g = obs["logQ"].mean(); sd_g = obs["logQ"].std()
    p1_g = obs["logQ"].quantile(0.01); p99_g = obs["logQ"].quantile(0.99)

    # Merge stats into imputed
    imp2 = imp.merge(stats_obs, on=strata, how="left")
    imp2 = imp2.merge(stats_obs_country, on=country_col, how="left")

    if method == "meanvar":
        # Compute imp group stats for scaling
        stats_imp = (
            imp2.groupby(strata)["logQ"].agg(mean_imp="mean", sd_imp="std").reset_index()
        )
        imp2 = imp2.merge(stats_imp, on=strata, how="left")

        def calibrate_row_meanvar(r):
            logq = r["logQ"]
            # choose observed stats: prefer stratum, then country, then global
            mean_o = r["mean_obs"] if pd.notna(r.get("mean_obs")) else r.get("mean_obs_c")
            sd_o = r["sd_obs"] if pd.notna(r.get("sd_obs")) else r.get("sd_obs_c")
            p1 = r["p1"] if pd.notna(r.get("p1")) else r.get("p1_c")
            p99 = r["p99"] if pd.notna(r.get("p99")) else r.get("p99_c")
            if pd.isna(mean_o) or pd.isna(sd_o):
                mean_o, sd_o, p1, p99 = mean_g, sd_g, p1_g, p99_g
            mean_i = r.get("mean_imp"); sd_i = r.get("sd_imp")
            # avoid zero/NaN sd
            if pd.isna(sd_i) or sd_i <= 1e-9:
                sd_i = sd_o if pd.notna(sd_o) and sd_o > 1e-9 else 1.0
                if pd.isna(mean_i):
                    mean_i = logq
            if pd.isna(mean_i):
                mean_i = logq
            # mean/variance match
            adj = (logq - mean_i) * (sd_o / sd_i) + mean_o if sd_i > 0 and pd.notna(sd_o) else logq
            # clip to observed 1–99% to control tails
            if pd.notna(p1) and pd.notna(p99) and p1 < p99:
                adj = float(np.clip(adj, p1, p99))
            return adj

        imp2["logQ_cal"] = imp2.apply(calibrate_row_meanvar, axis=1)
        calibrated_method = "meanvar_clip_p1p99_v1"
    else:
        # Quantile mapping (v2): map each row based on its percentile within the imputed stratum
        # Build dicts for quick access to observed lists per stratum and country
        key_cols = strata
        obs_map = {tuple(row[c] for c in key_cols): row["obs_list"] for _, row in stats_obs.iterrows()}
        obs_country_map = {row[country_col]: row["obs_list_c"] for _, row in stats_obs_country.iterrows()}
        obs_global = obs["logQ"].sort_values().to_numpy()

        # Compute imputed-group percentiles for each row
        imp2["grp_n"] = imp2.groupby(strata)["logQ"].transform("size")
        imp2["grp_rank"] = imp2.groupby(strata)["logQ"].rank(method="average")
        imp2["q_imp"] = (imp2["grp_rank"]) / (imp2["grp_n"] + 1.0)
        imp2["q_imp"] = imp2["q_imp"].clip(lower=1e-6, upper=1 - 1e-6)

        # Apply per-row mapping to observed distributions
        mapped = []
        for _, r in imp2.iterrows():
            q = float(r["q_imp"]) if pd.notna(r["q_imp"]) else 0.5
            k = (r[country_col], r["decade"])
            obs_list = obs_map.get(k, None)
            if obs_list is not None and len(obs_list) >= 10:
                target_vals = np.asarray(obs_list, dtype=float)
                adj = float(np.quantile(target_vals, q))
                p1 = r.get("p1"); p99 = r.get("p99")
                if pd.notna(p1) and pd.notna(p99) and p1 < p99:
                    adj = float(np.clip(adj, p1, p99))
                mapped.append(adj)
                continue
            # country fallback
            obs_list_c = obs_country_map.get(r[country_col], None)
            if obs_list_c is not None and len(obs_list_c) >= 10:
                target_vals = np.asarray(obs_list_c, dtype=float)
                adj = float(np.quantile(target_vals, q))
                p1 = r.get("p1_c"); p99 = r.get("p99_c")
                if pd.notna(p1) and pd.notna(p99) and p1 < p99:
                    adj = float(np.clip(adj, p1, p99))
                mapped.append(adj)
                continue
            # global fallback
            if len(obs_global) >= 10:
                adj = float(np.quantile(obs_global, q))
                adj = float(np.clip(adj, p1_g, p99_g))
                mapped.append(adj)
            else:
                mapped.append(float(r["logQ"]))

        imp2["logQ_cal"] = np.array(mapped, dtype=float)
        calibrated_method = "quantile_map_v2"
    imp2["Q_cal"] = np.power(10.0, imp2["logQ_cal"]).clip(lower=0)

    # Prepare updates: set only imputed rows
    updates = imp2[[qty_col, country_col, "decade", "Q_cal"]].copy()
    # Use index alignment to original df rows
    imp_idx = imp2.index

    # Determine output calibrated column name based on quantity column
    out_cal_col = "reserves_initial_EJ_calibrated" if normalize_name(qty_col) == normalize_name("reserves_initial_EJ") else "Quantity_initial_EJ_calibrated"

    # Ensure columns exist
    ensure_columns(conn, table, [
        (out_cal_col, "REAL"),
        ("_calibrated_flag_quantity", "INTEGER"),
        ("_calibrated_method_quantity", "TEXT")
    ])

    # Write back using a temporary key
    # Create a temp table with rowid mapping
    df_with_rowid = pd.read_sql_query(f"SELECT rowid as _rid, * FROM {table}", conn)
    df_with_rowid = df_with_rowid.loc[df_with_rowid[qty_col].notna() & (df_with_rowid[qty_col] > 0)].copy()
    df_with_rowid["logQ"] = np.log10(df_with_rowid[qty_col])
    if year_col is not None:
        df_with_rowid["decade"] = add_decade(df_with_rowid[year_col])
    else:
        df_with_rowid["decade"] = pd.Series([pd.NA] * len(df_with_rowid), dtype="Int64")

    # Attach calibrated values by joining on (country, decade, logQ) & imputed flag
    # To avoid floating join issues, join by index using the same filtering order as imp2
    imp_mask = (df_with_rowid[flag_col] != 0) & df_with_rowid[qty_col].notna() & (df_with_rowid[qty_col] > 0)
    df_imp_only = df_with_rowid.loc[imp_mask].copy()
    # Align lengths
    if len(df_imp_only) != len(imp2):
        # fallback: re-merge by country + decade + nearest logQ (coarse bin)
        df_imp_only["logQ_bin"] = (df_imp_only["logQ"]*1000).round().astype(int)
        tmp = imp2.copy()
        tmp["logQ_bin"] = (tmp["logQ"]*1000).round().astype(int)
        merged = df_imp_only.merge(tmp[[country_col, "decade", "logQ_bin", "Q_cal"]], on=[country_col, "decade", "logQ_bin"], how="left")
        updates_frame = merged[["_rid", "Q_cal"]].dropna()
    else:
        updates_frame = df_imp_only[["_rid"]].copy()
        updates_frame["Q_cal"] = imp2["Q_cal"].values

    # Apply updates
    cur = conn.cursor()
    cur.execute("BEGIN TRANSACTION;")
    try:
        for rid, qcal in updates_frame[["_rid", "Q_cal"]].itertuples(index=False):
            cur.execute(
                f"UPDATE {table} SET {out_cal_col}=?, _calibrated_flag_quantity=1, _calibrated_method_quantity=? WHERE rowid=?",
                (float(qcal), calibrated_method, int(rid))
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()

    print(f"[ok] {table}: calibrated {len(updates_frame)} imputed rows → {out_cal_col}")


# CLI -------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Post-imputation calibration to align imputed distributions")
    ap.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    ap.add_argument("--tables", nargs="+", choices=["Oil", "Gas", "Coal"], default=["Oil", "Gas"], help="Tables to calibrate")
    ap.add_argument("--method", choices=["quantile_map", "meanvar"], default="quantile_map", help="Calibration method")
    return ap.parse_args()


def main():
    args = parse_args()
    db_path = args.db
    conn = sqlite3.connect(db_path)
    try:
        if "Oil" in args.tables:
            calibrate_table(conn, "Oil_fields", method=args.method)
        if "Gas" in args.tables:
            calibrate_table(conn, "Gas_fields", method=args.method)
        if "Coal" in args.tables:
            calibrate_table(conn, "Coal_open_mines", method=args.method)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
