#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_fossil_fuels.py
Create a CSV summarizing discoveries, production, cumulative series, and
reserves (= cum_discoveries - cum_production) for Oil, Natural gas, and Coal.

Usage:
  python summarize_fossil_fuels.py \
      --db /path/to/Energy.db \
      --out fossil_fuels_summary.csv \
      --format long   # or wide
      --year-min 1850 --year-max 2100
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_name(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def resolve_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first matching column (case/space-insensitive), else None."""
    norm_map = {normalize_name(c): c for c in cols}
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def apply_reserve_growth_yearly(
    df: pd.DataFrame,
    year_col: str = "Year",
    value_col: str = "Discoveries",
    growth_rate: float = 0.3,
    horizon: int = 30,
    tau: Optional[float] = None,
) -> pd.DataFrame:
    """Apply simple reserve-growth/backdating to a yearly discoveries series.

    We redistribute a fraction of future additions back to the original discovery year
    using an exponential kernel over a finite horizon. The weights sum to `growth_rate`.
    """
    if growth_rate <= 0 or horizon <= 0 or df.empty:
        return df

    ser = df.set_index(year_col)[value_col].astype(float).fillna(0.0).copy()
    years = ser.index.astype(int).to_numpy()
    if len(years) == 0:
        return df

    if tau is None:
        tau = max(1.0, horizon / 5.0)
    ks = np.arange(1, horizon + 1)
    base = np.exp(-ks / tau)
    w = base / base.sum()
    w = w * growth_rate  # total mass moved back

    # For each year t, add w[k]*D[t+k] back to year t
    ser_adj = ser.copy()
    year_set = set(years.tolist())
    for idx, y in enumerate(years):
        add = 0.0
        for k, wk in zip(ks, w):
            yk = y + k
            if yk in year_set:
                add += wk * float(ser.get(yk, 0.0))
        ser_adj[y] = float(ser.get(y, 0.0)) * (1.0 + (growth_rate * 0.0)) + add

    out = df.copy()
    out[value_col] = ser_adj.reindex(out[year_col].astype(int)).values
    return out


def read_discoveries(
    con: sqlite3.Connection,
    table: str,
    year_candidates: List[str],
    qty_candidates: List[str],
    qty_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Read and aggregate discoveries per year (sum).

    Uses a list of quantity column candidates and picks the first match case-insensitively.
    """
    cols = pd.read_sql(f"PRAGMA table_info('{table}')", con)["name"].tolist()
    year_col = resolve_column(cols, year_candidates)
    if not year_col:
        raise ValueError(f"{table} lacks required year column among candidates: {year_candidates}.")
    col_qty = resolve_column(cols, qty_candidates)
    if not col_qty:
        raise ValueError(f"{table} lacks required quantity column among candidates: {qty_candidates}.")

    df = pd.read_sql(f'SELECT "{year_col}" AS Year, "{col_qty}" AS Discoveries FROM "{table}"', con)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Discoveries"] = pd.to_numeric(df["Discoveries"], errors="coerce")
    if qty_multiplier != 1.0:
        df["Discoveries"] = df["Discoveries"] * float(qty_multiplier)
    df = df.dropna(subset=["Year"])
    # Sum per year, ignore null quantities
    df = df.groupby("Year", dropna=False, as_index=False, observed=True)["Discoveries"].sum(min_count=1)
    return df


def first_available_table(con: sqlite3.Connection, candidates: List[str]) -> Optional[str]:
    """Return the first table name that exists in the DB from candidates."""
    for t in candidates:
        try:
            cols = pd.read_sql(f"PRAGMA table_info('{t}')", con)
            if not cols.empty:
                return t
        except Exception:
            continue
    return None


def list_columns(con: sqlite3.Connection, table: str) -> List[str]:
    try:
        return pd.read_sql(f"PRAGMA table_info('{table}')", con)["name"].tolist()
    except Exception:
        return []


def read_production_oil(
    con: sqlite3.Connection,
    scope: str = "cc_ngl",
    table: Optional[str] = None,
    col: Optional[str] = None,
) -> pd.DataFrame:
    """Load oil production with scope control.

    scope:
      - 'cc'      → try columns like Crude+Condensate
      - 'cc_ngl'  → try columns like World/Total
    If table is None, default to 'Oil_production_history'.
    """
    if table is None:
        # Default EI table naming; user can override via CLI
        table = "Oil_production_history"
    try:
        # Resolve year column flexibly
        cols = pd.read_sql(f"PRAGMA table_info('{table}')", con)["name"].tolist()
        year_col = resolve_column(cols, ["Year", "year", "Data year", "Data Year", "data_year"]) or "Year"
        if year_col not in cols:
            raise KeyError("missing_year_column")
    except KeyError:
        # Fallback: use aggregated backdated table if available (Gb→EJ)
        bt_cols = pd.read_sql("PRAGMA table_info('Discoveries_Production_backdated')", con)["name"].tolist()
        ycol = resolve_column(bt_cols, ["year", "Year"]) or "year"
        pcol = resolve_column(bt_cols, ["production", "Production", "production_(gb)", "Production_(Gb)"]) or "production"
        if ycol in bt_cols and pcol in bt_cols:
            df = pd.read_sql(f"SELECT \"{ycol}\" AS Year, \"{pcol}\" AS ProductionGb FROM 'Discoveries_Production_backdated'", con)
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
            df["Production"] = pd.to_numeric(df["ProductionGb"], errors="coerce") * 6.1
            df = df.dropna(subset=["Year"]).groupby("Year", as_index=False)["Production"].sum(min_count=1)
            return df
        raise
    if col:
        candidates = [col]
    elif scope == "cc":
        candidates = [
            "Crude+Condensate",
            "Crude and Condensate",
            "Crude_and_Condensate",
            "Crude_Condensate",
            "CrudeCondensate",
        ]
    else:  # cc_ngl
        candidates = [
            "World",
            "Total",
            "Crude+Cond+NGL",
            "Crude_Cond_NGL",
        ]

    col = resolve_column(cols, candidates) or ("World" if "World" in cols else None)
    if not col:
        raise ValueError(f"No suitable production column found in {table} for scope {scope}")
    return read_production(con, table=table, year_col=year_col, prod_col=col)


def read_production(
    con: sqlite3.Connection,
    table: str,
    year_col: str = "Year",
    prod_col: str = "World",
) -> pd.DataFrame:
    """Read production per year (sum if duplicates)."""
    cols = pd.read_sql(f"PRAGMA table_info('{table}')", con)["name"].tolist()
    
    # Use case-insensitive column resolution
    actual_year_col = resolve_column(cols, [year_col])
    actual_prod_col = resolve_column(cols, [prod_col])
    
    if not actual_year_col or not actual_prod_col:
        raise ValueError(f"{table} must contain '{year_col}' and '{prod_col}'.")
    
    df = pd.read_sql(f'SELECT "{actual_year_col}" AS Year, "{actual_prod_col}" AS Production FROM "{table}"', con)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df = df.groupby("Year", dropna=False, as_index=False, observed=True)["Production"].sum(min_count=1)
    return df


def build_series(
    discoveries: pd.DataFrame,
    production: pd.DataFrame,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> pd.DataFrame:
    """Combine discoveries and production, compute cumulative and reserves."""
    # Outer join on years
    years = pd.DataFrame({"Year": pd.Index(
        sorted(set(discoveries["Year"].dropna().astype(int)).union(set(production["Year"].dropna().astype(int))))
    )})
    if year_min is not None:
        years = years[years["Year"] >= int(year_min)]
    if year_max is not None:
        years = years[years["Year"] <= int(year_max)]
    years = years.sort_values("Year").reset_index(drop=True)

    out = years.merge(discoveries, on="Year", how="left").merge(production, on="Year", how="left")
    # Fill missing with 0 for running totals; keep NaNs if both missing
    out["Discoveries"] = out["Discoveries"].fillna(0.0)
    out["Production"] = out["Production"].fillna(0.0)

    out["Cumulative_Discoveries"] = out["Discoveries"].cumsum()
    out["Cumulative_Production"]  = out["Production"].cumsum()
    out["Reserves"] = out["Cumulative_Discoveries"] - out["Cumulative_Production"]
    return out


def to_long(df_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-fuel frames to a tidy long table."""
    frames = []
    for fuel, df in df_map.items():
        tmp = df.copy()
        tmp.insert(0, "Fuel", fuel)
        frames.append(tmp)
    long_df = pd.concat(frames, ignore_index=True)
    # Order columns
    cols = ["Fuel", "Year", "Discoveries", "Cumulative_Discoveries",
            "Production", "Cumulative_Production", "Reserves"]
    return long_df[cols].sort_values(["Fuel", "Year"]).reset_index(drop=True)


def to_wide(df_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a wide table with per-fuel columns.

    Column order: Year, then Oil_* metrics, then Gas_*, then Coal_*.
    Metrics per fuel: Discoveries, Cumulative_Discoveries, Production,
    Cumulative_Production, Reserves.
    """
    # Start from complete set of years
    all_years = sorted(set().union(*[df["Year"].tolist() for df in df_map.values()]))
    wide = pd.DataFrame({"Year": all_years})
    metrics = ["Discoveries", "Cumulative_Discoveries", "Production", "Cumulative_Production", "Reserves"]

    # Merge all fuels
    for fuel, df in df_map.items():
        df2 = df.copy()
        df2.columns = ["Year"] + [f"{fuel}_{c}" for c in df.columns if c != "Year"]
        wide = wide.merge(df2, on="Year", how="left")

    # Desired fuel order
    desired_fuels = ["Oil", "Gas", "Coal"]
    fuels = [f for f in desired_fuels if f in df_map.keys()]

    # Reorder columns: Year, then for each fuel all metrics in order
    ordered_cols = ["Year"]
    for fuel in fuels:
        for metric in metrics:
            col = f"{fuel}_{metric}"
            if col in wide.columns:
                ordered_cols.append(col)

    # Keep any remaining unexpected columns at the end
    remaining = [c for c in wide.columns if c not in ordered_cols]
    ordered_cols += remaining

    return wide[ordered_cols].sort_values("Year").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Summarize discoveries, production, reserves for Oil, Gas, and Coal (EJ).")
    ap.add_argument("--db", type=Path, required=True, help="Path to Energy.db")
    ap.add_argument("--out", type=Path, default=Path("fossil_fuels_summary.csv"), help="Output CSV path")
    ap.add_argument("--format", choices=["long", "wide"], default="wide", help="Output layout")
    ap.add_argument("--year-min", type=int, default=None, help="Minimum year to include")
    ap.add_argument("--year-max", type=int, default=None, help="Maximum year to include")
    ap.add_argument("--prefer-calibrated", action="store_true", default=True, help="Prefer calibrated quantities if available (default true)")
    ap.add_argument("--strict", action="store_true", help="Error out on missing tables/columns instead of warning")
    # Oil scope and EI loader options
    ap.add_argument("--oil-scope", choices=["cc", "cc_ngl"], default="cc_ngl", help="Oil production scope: crude+condensate (cc) or crude+condensate+NGL (cc_ngl)")
    ap.add_argument("--oil-prod-table", default="Oil_production_history", help="Oil production table name (default: Oil_production_history)")
    ap.add_argument("--oil-prod-col", default=None, help="Override oil production column name (optional)")
    # Reserve growth/backdating
    ap.add_argument("--apply-reserve-growth", action="store_true", help="Apply reserve-growth/backdating to discoveries before aggregation")
    ap.add_argument("--growth-rate", type=float, default=0.3, help="Total reserve-growth fraction redistributed back to discovery year (default 0.3)")
    ap.add_argument("--growth-horizon", type=int, default=30, help="Reserve-growth horizon in years (default 30)")
    ap.add_argument("--growth-tau", type=float, default=None, help="Exponential kernel tau (defaults to horizon/5)")
    # Reconciliation report
    ap.add_argument("--recon-out", type=Path, default=Path("fossil_fuels_reconciliation.csv"), help="Path to write reconciliation report CSV")
    args = ap.parse_args()

    with sqlite3.connect(str(args.db)) as con:
        # Determine quantity columns based on preference
        oil_qty_pref = "Quantity_initial_EJ_calibrated" if args.prefer_calibrated else "Quantity_initial_EJ"
        oil_qty_fallback = "Quantity_initial_EJ" if args.prefer_calibrated else None
        gas_qty_pref = "Quantity_initial_EJ_calibrated" if args.prefer_calibrated else "Quantity_initial_EJ"
        gas_qty_fallback = "Quantity_initial_EJ" if args.prefer_calibrated else None
        coal_qty_pref = "reserves_initial_EJ_calibrated" if args.prefer_calibrated else "reserves_initial_EJ"
        coal_qty_fallback = "reserves_initial_EJ" if args.prefer_calibrated else None

        frames: Dict[str, pd.DataFrame] = {}

        # ---- Oil ----
        try:
            oil_tbl = first_available_table(con, [
                # Prioritize aggregated, backdated series if present
                "Discoveries_Production_backdated",
                "Oil_fields",
                "oil_fields",
                "Oil_Gas_fields_Oil",
                "Oil_Gas_fields_oil",
                "Oil_Gas_fields",
            ])
            if not oil_tbl:
                raise ValueError("No Oil fields table found among candidates")
            if oil_tbl == "Discoveries_Production_backdated":
                # Table stores oil discoveries in gigabarrels
                oil_disc = read_discoveries(
                    con,
                    table=oil_tbl,
                    year_candidates=["year", "Year"],
                    qty_candidates=["discoveries_(gb)", "Discoveries_(Gb)", "discoveries_gb"],
                    qty_multiplier=6.1,  # 1 Gbbl ≈ 6.1 EJ
                )
            else:
                oil_disc = read_discoveries(
                    con,
                    table=oil_tbl,
                    year_candidates=[
                        "discovery_year_final",
                        "discovery_year",
                        "Discovery year",
                        "Year of Discovery",
                        "Year",
                        "Data year",
                    ],
                    qty_candidates=[
                        # Field-level initial quantities / URR
                        "Quantity_initial_EJ_calibrated",
                        "Quantity_initial_EJ",
                        "Quantity_EJ_initial",
                        "Quantity_EJ",
                        "quantity_initial_ej_calibrated",
                        "quantity_initial_ej",
                        "quantity_ej_initial",
                        "quantity_ej",
                        "Estimated_ultimate_recovery_EJ",
                        "URR_EJ",
                        # Aggregated discoveries series
                        "Discoveries",
                        "Discoveries_EJ",
                        "discoveries_ej",
                        "Value_EJ",
                        "Value",
                        "Quantity",
                        "Quantity_EJ",
                    ],
                )
            if args.apply_reserve_growth:
                oil_disc = apply_reserve_growth_yearly(
                    oil_disc, year_col="Year", value_col="Discoveries",
                    growth_rate=args.growth_rate, horizon=args.growth_horizon, tau=args.growth_tau,
                )
            oil_prod = read_production_oil(
                con, scope=args.oil_scope, table=args.oil_prod_table, col=args.oil_prod_col,
            )
            frames["Oil"] = build_series(oil_disc, oil_prod, args.year_min, args.year_max)
        except Exception as e:
            msg = f"[warn] Oil summary skipped: {e}"
            if args.strict:
                raise
            else:
                print(msg)

        # ---- Natural gas ----
        try:
            # Try to get gas discoveries from various sources
            gas_disc = None
            gas_tbl = first_available_table(con, [
                "Gas_fields",
                "gas_fields",
                "Oil_Gas_fields_Gas",
                "Oil_Gas_fields_gas",
                "Oil_Gas_fields",
            ])
            if gas_tbl:
                try:
                    gas_disc = read_discoveries(
                        con,
                        table=gas_tbl,
                        year_candidates=[
                            "discovery_year_final",
                            "discovery_year",
                            "Discovery year",
                            "Year of Discovery",
                            "Year",
                            "Data year",
                        ],
                        qty_candidates=[
                            "Quantity_initial_EJ_calibrated",
                            "Quantity_initial_EJ",
                            "Quantity_EJ_initial",
                            "Quantity_EJ",
                            "quantity_initial_ej_calibrated",
                            "quantity_initial_ej",
                            "quantity_ej_initial",
                            "quantity_ej",
                            "Estimated_ultimate_recovery_EJ",
                            "URR_EJ",
                            "Discoveries",
                            "Discoveries_EJ",
                            "discoveries_ej",
                            "Value_EJ",
                            "Value",
                            "Quantity",
                            "Quantity_EJ",
                        ],
                    )
                    if args.apply_reserve_growth:
                        gas_disc = apply_reserve_growth_yearly(
                            gas_disc, year_col="Year", value_col="Discoveries",
                            growth_rate=args.growth_rate, horizon=args.growth_horizon, tau=args.growth_tau,
                        )
                except Exception:
                    gas_disc = None
            
            # If no discoveries found, create empty discoveries DataFrame
            if gas_disc is None:
                print("[info] Gas discoveries not available; using empty discoveries series")
                gas_disc = pd.DataFrame({"Year": [], "Discoveries": []})
            
            gas_prod = read_production(con, table="Gas_production_history", year_col="Year", prod_col="World")
            frames["Gas"] = build_series(gas_disc, gas_prod, args.year_min, args.year_max)
        except Exception as e:
            msg = f"[warn] Gas summary failed: {e}"
            if args.strict:
                raise
            else:
                print(msg)
                # Create minimal Gas frame with production-only if possible
                try:
                    gas_disc = pd.DataFrame({"Year": [], "Discoveries": []})
                    gas_prod = read_production(con, table="Gas_production_history", year_col="Year", prod_col="World")
                    frames["Gas"] = build_series(gas_disc, gas_prod, args.year_min, args.year_max)
                except Exception:
                    print("[warn] Gas production also failed; creating empty Gas frame")
                    frames["Gas"] = pd.DataFrame({"Year": [], "Discoveries": [], "Cumulative_Discoveries": [], "Production": [], "Cumulative_Production": [], "Reserves": []})

        # ---- Coal ----
        try:
            # Try to get coal discoveries from various sources
            coal_disc = None
            coal_tbl = first_available_table(con, [
                "Coal_open_mines",
                "Coal_missing",
                "coal_open_mines",
                "coal_missing",
                "Coal_discoveries",
            ])
            if coal_tbl:
                try:
                    coal_disc = read_discoveries(
                        con,
                        table=coal_tbl,
                        year_candidates=[
                            "opening_year_final",
                            "Opening Year",
                            "Year of Production",
                            "Year",
                        ],
                        qty_candidates=[
                            "reserves_initial_EJ_calibrated",
                            "reserves_initial_EJ",
                            "Reserves_final",
                            "Reserves_imp",
                            "reserves_ej",
                            "Quantity_initial_EJ",
                            "Quantity_EJ",
                        ],
                    )
                    # Reserve growth not typically applied to coal; keep off intentionally
                except Exception:
                    coal_disc = None
            
            # If no discoveries found, create empty discoveries DataFrame
            if coal_disc is None:
                print("[info] Coal discoveries not available; using empty discoveries series")
                coal_disc = pd.DataFrame({"Year": [], "Discoveries": []})
            
            coal_prod = read_production(con, table="Coal_production_history", year_col="Year", prod_col="World")
            frames["Coal"] = build_series(coal_disc, coal_prod, args.year_min, args.year_max)
        except Exception as e:
            msg = f"[warn] Coal summary failed: {e}"
            if args.strict:
                raise
            else:
                print(msg)
                # Create minimal Coal frame with production-only if possible
                try:
                    coal_disc = pd.DataFrame({"Year": [], "Discoveries": []})
                    coal_prod = read_production(con, table="Coal_production_history", year_col="Year", prod_col="World")
                    frames["Coal"] = build_series(coal_disc, coal_prod, args.year_min, args.year_max)
                except Exception:
                    print("[warn] Coal production also failed; creating empty Coal frame")
                    frames["Coal"] = pd.DataFrame({"Year": [], "Discoveries": [], "Cumulative_Discoveries": [], "Production": [], "Cumulative_Production": [], "Reserves": []})

    # Assemble output
    if args.format == "long":
        out_df = to_long(frames)
    else:
        out_df = to_wide(frames)

    # Final tidy up: numeric formatting and sorting
    numeric_cols = out_df.columns.drop(["Fuel", "Year"], errors="ignore")
    out_df[numeric_cols] = out_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    out_df = out_df.sort_values(list({"Year", "Fuel"} & set(out_df.columns))).reset_index(drop=True)

    # Write CSV
    out_df.to_csv(args.out, index=False)
    # Console summary
    if "Fuel" in out_df.columns:
        summary = out_df.groupby("Fuel")[["Discoveries", "Production"]].sum().round(2)
        print("Totals (EJ) — long format:\n", summary.to_string())
    else:
        print(f"Wrote wide table with columns: {', '.join(out_df.columns)}")
    print(f"Saved → {args.out.resolve()}")

    # Reconciliation report (implied reserves vs optional EI reserves, if present)
    try:
        recon_rows: List[Dict[str, object]] = []
        for fuel, df in frames.items():
            tmp = df.copy()
            # Optional EI reserves lookup
            ei_tbl_candidates = {
                "Oil": ["Oil_reserves_history", "EI_oil_reserves", "Oil_proved_reserves"],
                "Gas": ["Gas_reserves_history", "EI_gas_reserves", "Gas_proved_reserves"],
                "Coal": ["Coal_reserves_history", "EI_coal_reserves", "Coal_proved_reserves"],
            }.get(fuel, [])
            ei_series = None
            for tname in ei_tbl_candidates:
                try:
                    cols = pd.read_sql(f"PRAGMA table_info('{tname}')", con)["name"].tolist()
                    ycol = resolve_column(cols, ["Year"])
                    vcol = resolve_column(cols, ["World", "Total"]) if cols else None
                    if ycol and vcol:
                        ei = pd.read_sql(f'SELECT "{ycol}" AS Year, "{vcol}" AS EI_Reserves FROM "{tname}"', con)
                        ei["Year"] = pd.to_numeric(ei["Year"], errors="coerce").astype("Int64")
                        ei_series = ei
                        break
                except Exception:
                    continue

            tmp = tmp.merge(ei_series if ei_series is not None else pd.DataFrame(columns=["Year", "EI_Reserves"]), on="Year", how="left")
            tmp["Delta_vs_EI"] = tmp["Reserves"] - tmp.get("EI_Reserves")
            tmp["NegativeFlag"] = (tmp["Reserves"] < 0).astype(int)
            tmp.insert(0, "Fuel", fuel)
            recon_rows.append(tmp)

        recon_df = pd.concat(recon_rows, ignore_index=True)
        recon_df.to_csv(args.recon_out, index=False)
        print(f"Reconciliation saved → {args.recon_out.resolve()}")
    except Exception as e:
        print(f"[warn] Reconciliation report failed: {e}")


if __name__ == "__main__":
    main()
