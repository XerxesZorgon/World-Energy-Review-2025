#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_modified_discoveries.py

Construct a modified discoveries-by-year series (Oil, Gas, Coal) for 1900..latest:
- Aggregate observed (GOGET-derived) discoveries by year from Energy.db.
- Apply a simple reserve-growth "backdating" multiplier by field age (USGS/Arrington inspired).
- Define target cumulative = EI Production (EJ) cumulative + EI Proved Reserves (EJ).
- Solve a nonnegative least-squares problem to allocate "missing mass" M_t so
  Reserves_hat_t ≈ EI reserves path, while respecting a 1960s peak and low values circa 1900.
- Output a long CSV: Fuel, Year, D_obs_EJ, D_backdated_EJ, D_added_EJ, D_hat_EJ,
  Cum_D_hat_EJ, Cum_Prod_EJ, Reserves_hat_EJ, Reserves_EI_EJ.

Dependencies: numpy, pandas; scipy (optional, for bounded least squares; a fallback is provided).

Key assumptions & references:
- GOGET coverage thresholds cause early undercounting (≥25 mmboe reserves or ≥1 mmboe/yr). (GEM GOGET docs)
- Reserve growth/backdating is real and material; we use a simple, tunable growth curve. (USGS, Arrington)
- EI definitions & conversions:
    oil production includes crude + condensate + NGLs + oil sands (Energy Institute);
    1 barrel ≈ 5.8 MMBtu ≈ 6.119 GJ (EIA/BP);
    gas: 1 bcm = 36 PJ LHV (IEA).
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Try SciPy bounded least squares; fall back to a simple projected solver if unavailable.
try:
    from scipy.optimize import lsq_linear
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------------- configuration defaults ----------------------

DEFAULT_START_YEAR = 1900
# Backdating parameters (growth multiplier approaches 1 + alpha as age -> ∞)
PARAMS = {
    "oil": {"alpha": 0.50, "tau": 30.0},  # modest long-run +50% with ~30y time constant
    "gas": {"alpha": 0.60, "tau": 35.0},
    "coal": {"alpha": 0.00, "tau": 30.0}, # keep 0 by default for coal
}
# 1960s discovery peak prior
PRIOR_PEAK_YEAR = 1965
PRIOR_PEAK_WIDTH = 12   # years (controls the Gaussian-like prior weights)
PRIOR_LAMBDA = 1e-2     # regularization strength for prior on M_t (can raise to tighten)

# Unit conversions (cited):
KB_PER_DAY_TO_EJ_PER_YEAR = 1000 * 365 * 5.8 * 1.055056e9 / 1e18  # kb/d -> EJ/yr  (EIA/Bureau of Mines)
# Oil: 1 barrel ≈ 6.119 GJ (EIA/BP), 1 Gb = 1e9 barrels
GB_TO_EJ = 6.119  # EJ per Gb (since 1 Gb * 6.119 GJ/bbl * 1e9 bbl = 6.119e18 J = 6.119 EJ)
TCM_TO_EJ = 36.0  # EJ per tcm (IEA LHV basis)
# Coal energy density default (EJ per Gt). 1 t * 24 GJ/t = 24e9 J = 2.4e-8 EJ; per Gt => 24 EJ/Gt
COAL_EJ_PER_GT = 24.0


# ---------------------- helpers ----------------------

def years_from_columns(df: pd.DataFrame, exclude_cols=("Country", "Country/Area", "Fuel")) -> pd.Index:
    years = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        try:
            y = int(c)
            if 1800 <= y <= 2100:
                years.append(y)
        except Exception:
            pass
    return pd.Index(sorted(years))


def get_latest_year(*series: pd.DataFrame) -> int:
    last = DEFAULT_START_YEAR
    for s in series:
        if s is None or s.empty:
            continue
        if "Year" in s.columns:
            last = max(last, int(pd.to_numeric(s["Year"], errors="coerce").dropna().max()))
    return last


def gaussian_weights(years: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Calculate normalized Gaussian weights centered at mu.
    
    Args:
        years: Array of year values
        mu: Center year for peak
        sigma: Width parameter (years)
        
    Returns:
        Normalized weights summing to 1
    """
    sigma = max(sigma, 1e-6)  # Ensure minimum sigma to avoid division by zero
    w = np.exp(-0.5 * ((years - mu) / sigma)**2)
    return w / max(w.sum(), 1e-12)


def backdating_multiplier(age: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    """
    Calculate reserve growth multiplier based on field age.
    
    Args:
        age: Array of field ages in years
        alpha: Maximum growth factor (0-1 typical)
        tau: Time constant for growth curve (years)
        
    Returns:
        Growth multipliers: 1 + alpha*(1 - exp(-age/tau))
    """
    return 1.0 + alpha * (1.0 - np.exp(-np.maximum(age, 0.0) / max(tau, 1e-6)))


def lower_triangular_cumsum_matrix(T: int) -> np.ndarray:
    # A is T×T with ones on and below diagonal so (A @ x)[t] = sum_{i<=t} x[i]
    A = np.tril(np.ones((T, T), dtype=float))
    return A


def solve_missing_mass_nnls(A: np.ndarray, y: np.ndarray, m_prior: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve min ||A m - y||^2 + lam ||m - m_prior||^2, with m >= 0.
    If SciPy is available, use bounded least squares (lsq_linear), else fallback to a simple projected GD.
    """
    # Augment for Tikhonov: [A; sqrt(lam)*I] m ≈ [y; sqrt(lam)*m_prior]
    T = A.shape[1]
    if lam > 0:
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(T)])
        b_aug = np.hstack([y, np.sqrt(lam) * m_prior])
    else:
        A_aug, b_aug = A, y

    if HAVE_SCIPY:
        res = lsq_linear(A_aug, b_aug, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
        m = res.x
        m[m < 0] = 0.0
        return m

    # Fallback: simple projected gradient descent
    m = np.maximum(0.0, m_prior.copy())
    lr = 1e-3
    for _ in range(20000):
        grad = 2 * (A_aug.T @ (A_aug @ m - b_aug))
        m -= lr * grad
        m[m < 0] = 0.0
        # crude stopping
        if np.linalg.norm(grad) < 1e-6:
            break
    return m


# ---------------------- validation functions ----------------------

def validate_discovery_data(df: pd.DataFrame, fuel: str) -> pd.DataFrame:
    """Validate and clean discovery data"""
    if df.empty:
        print(f"Warning: No {fuel} discovery data found")
        return df
    
    # Check for negative discoveries
    if 'Q_EJ' in df.columns:
        neg_count = (df['Q_EJ'] < 0).sum()
        if neg_count > 0:
            print(f"Warning: {neg_count} negative {fuel} discoveries found, clipping to 0")
            df['Q_EJ'] = df['Q_EJ'].clip(lower=0)
    
    print(f"Loaded {len(df)} {fuel} discovery records")
    return df


def validate_params(args):
    """Validate input parameters"""
    if args.oil_alpha < 0 or args.gas_alpha < 0 or args.coal_alpha < 0:
        raise ValueError("Alpha parameters must be non-negative")
    if args.oil_tau <= 0 or args.gas_tau <= 0 or args.coal_tau <= 0:
        raise ValueError("Tau parameters must be positive")
    if args.prior_lambda < 0:
        raise ValueError("Prior lambda must be non-negative")
    if args.start_year < 1800 or args.start_year > 2100:
        raise ValueError("Start year must be between 1800 and 2100")
    if args.peak_year < args.start_year:
        raise ValueError("Peak year must be >= start year")
    if args.peak_width <= 0:
        raise ValueError("Peak width must be positive")


# ---------------------- DB readers & converters ----------------------

def read_oil_discoveries(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read oil field discoveries from Oil_fields, convert to EJ."""
    try:
        # Oil_fields: discovery_year_final, Quantity_initial_EJ_calibrated (fallback Quantity_initial_EJ)
        cols = pd.read_sql("PRAGMA table_info('Oil_fields')", conn)["name"].tolist()
        qty_col = "Quantity_initial_EJ_calibrated" if "Quantity_initial_EJ_calibrated" in cols else "Quantity_initial_EJ"
        df = pd.read_sql(f"""
            SELECT discovery_year_final AS Year, {qty_col} AS Q_EJ
            FROM Oil_fields
            WHERE discovery_year_final IS NOT NULL
        """, conn)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Q_EJ"] = pd.to_numeric(df["Q_EJ"], errors="coerce")
        result = df.dropna(subset=["Year", "Q_EJ"])
        return validate_discovery_data(result, "Oil")
    except sqlite3.Error as e:
        print(f"Warning: Could not read oil discoveries: {e}")
        return pd.DataFrame(columns=['Year', 'Q_EJ'])


def read_gas_discoveries(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read gas field discoveries from Gas_fields, convert to EJ."""
    try:
        q = "SELECT * FROM Gas_fields"
        df = pd.read_sql(q, conn)
        if "Quantity_initial_EJ" in df.columns:
            df = df[["discovery_year_final", "Quantity_initial_EJ"]].rename(columns={
                "discovery_year_final": "Year", "Quantity_initial_EJ": "Q_EJ"})
        else:
            # fallback column names
            df = df.rename(columns={"Discovery Year": "Year", "Reserves (Tcf)": "Q_Tcf"})
            df["Q_EJ"] = df["Q_Tcf"] * TCM_TO_EJ
        result = df[["Year", "Q_EJ"]].dropna()
        return validate_discovery_data(result, "Gas")
    except sqlite3.Error as e:
        print(f"Warning: Could not read gas discoveries: {e}")
        return pd.DataFrame(columns=['Year', 'Q_EJ'])


def read_coal_discoveries(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read coal mine discoveries from Coal_open_mines, convert to EJ."""
    try:
        q = "SELECT * FROM Coal_open_mines"
        df = pd.read_sql(q, conn)
        if "reserves_initial_EJ" in df.columns:
            df = df[["opening_year_final", "reserves_initial_EJ"]].rename(columns={
                "opening_year_final": "Year", "reserves_initial_EJ": "Q_EJ"})
        else:
            # fallback: assume Mt
            df = df.rename(columns={"Opening Year": "Year", "Reserves (Mt)": "Q_Mt"})
            df["Q_EJ"] = df["Q_Mt"] * COAL_EJ_PER_GT / 1000  # Mt -> Gt -> EJ
        result = df[["Year", "Q_EJ"]].dropna()
        return validate_discovery_data(result, "Coal")
    except sqlite3.Error as e:
        print(f"Warning: Could not read coal discoveries: {e}")
        return pd.DataFrame(columns=['Year', 'Q_EJ'])


def oil_production_ej(conn: sqlite3.Connection) -> pd.DataFrame:
    # EI_oil_production: wide table, 'Country' row "Total World" in kb/d
    wide = pd.read_sql('SELECT * FROM "EI_oil_production"', conn)
    row = wide[wide["Country"].str.lower().isin(["total world", "world"])]
    if row.empty:
        raise RuntimeError("EI_oil_production: 'Total World' row not found.")
    years = years_from_columns(wide, exclude_cols=("Country",))
    kbpd = row[years.astype(str).tolist()].iloc[0].astype(float).values
    prod_ej = kbpd * KB_PER_DAY_TO_EJ_PER_YEAR
    return pd.DataFrame({"Year": years.astype(int), "Production": prod_ej})


def gas_production_ej(conn: sqlite3.Connection) -> pd.DataFrame:
    # EI_gas_production_EJ: wide table, 'Country' row "Total World" already in EJ
    wide = pd.read_sql('SELECT * FROM "EI_gas_production_EJ"', conn)
    row = wide[wide[wide.columns[0]].str.lower().isin(["total world", "world"])]
    if row.empty:
        raise RuntimeError("EI_gas_production_EJ: 'Total World' row not found.")
    years = years_from_columns(wide, exclude_cols=(wide.columns[0],))
    prod = pd.to_numeric(row[years.astype(str).tolist()].iloc[0], errors="coerce").values
    return pd.DataFrame({"Year": years.astype(int), "Production": prod})


def coal_production_ej(conn: sqlite3.Connection) -> pd.DataFrame:
    # Coal_production_history: (Year, World) expected in EJ
    df = pd.read_sql('SELECT "Year", "World" as Production FROM "Coal_production_history"', conn)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    return df.dropna(subset=["Year", "Production"]).astype({"Year": int})


def oil_reserves_ej(conn: sqlite3.Connection) -> pd.DataFrame:
    # EI_oil_proved_reserves: wide; detect units (likely billion barrels); convert to EJ
    wide = pd.read_sql('SELECT * FROM "EI_oil_proved_reserves"', conn)
    row = wide[wide["Country"].str.lower().isin(["total world", "world"])]
    if row.empty:
        raise RuntimeError("EI_oil_proved_reserves: 'Total World' row not found.")
    years = years_from_columns(wide, exclude_cols=("Country",))
    vals = pd.to_numeric(row[years.astype(str).tolist()].iloc[0], errors="coerce").values
    # crude detection: typical world proved oil ≈ 1,500–1,800 Gb. If range ~[500, 5000], treat as Gb.
    if np.nanmax(vals) > 100 and np.nanmax(vals) < 10000:
        ej = vals * GB_TO_EJ
    else:
        # assume already EJ if numbers are O(10^3) EJ; if O(10^6), could be Mb etc. Allow override via CLI in future.
        ej = vals
    return pd.DataFrame({"Year": years.astype(int), "Reserves_EI": ej})


def gas_reserves_ej(conn: sqlite3.Connection) -> pd.DataFrame:
    wide = pd.read_sql('SELECT * FROM "EI_gas_proved_reserves"', conn)
    row = wide[wide[wide.columns[0]].str.lower().isin(["total world", "world"])]
    if row.empty:
        raise RuntimeError("EI_gas_proved_reserves: 'Total World' row not found.")
    years = years_from_columns(wide, exclude_cols=(wide.columns[0],))
    vals = pd.to_numeric(row[years.astype(str).tolist()].iloc[0], errors="coerce").values
    # typical tcm range ≈ 150–250, so detect and convert
    if np.nanmax(vals) > 50 and np.nanmax(vals) < 500:
        ej = vals * TCM_TO_EJ
    else:
        ej = vals
    return pd.DataFrame({"Year": years.astype(int), "Reserves_EI": ej})


def coal_reserves_ej_snapshot(conn: sqlite3.Connection) -> Tuple[int, float]:
    # EI_coal_reserves: snapshot by country (metric tonnes). Sum totals, convert to EJ with default GJ/t.
    df = pd.read_sql('SELECT * FROM "EI_coal_reserves"', conn)
    # Heuristic: if there is a "Total" column in Mt; else sum "Total" (Mt) column over countries.
    cand_cols = [c for c in df.columns if c.lower() in ("total", "total (mt)", "total_mt", "reserves_total")]
    if not cand_cols:
        raise RuntimeError("EI_coal_reserves: cannot find a 'Total' column.")
    mt = pd.to_numeric(df[cand_cols[0]], errors="coerce").dropna().sum()  # Mt
    gt = mt / 1000.0
    ej = gt * COAL_EJ_PER_GT
    # No year in table; assume last production year as anchor
    return (None, ej)


# ---------------------- core workflow per fuel ----------------------

def aggregate_and_backdate(df_fields: pd.DataFrame, last_year: int, alpha: float, tau: float) -> pd.DataFrame:
    """
    df_fields: rows (Year, Q_EJ) at field level or pre-aggregated with 'Year'.
    Returns per-year frame with D_obs_EJ and D_backdated_EJ.
    """
    # If this is field-level, aggregate by year:
    df = df_fields.groupby("Year", as_index=False)["Q_EJ"].sum().rename(columns={"Q_EJ": "D_obs_EJ"})
    # Backdate via a simple age-based multiplier applied to each field contribution:
    # Approximate by distributing year sums with the multiplier computed at age = last_year - year.
    years = df["Year"].astype(int).values
    age = last_year - years
    mult = backdating_multiplier(age.astype(float), alpha=alpha, tau=tau)
    df["D_backdated_EJ"] = df["D_obs_EJ"] * mult
    return df


def build_target_and_fit(fuel: str,
                         df_back: pd.DataFrame,
                         prod: pd.DataFrame,
                         reserves: Optional[pd.DataFrame],
                         start_year: int,
                         last_year: int,
                         prior_peak_year: int,
                         prior_peak_width: float,
                         prior_lambda: float) -> pd.DataFrame:
    """
    Given backdated discoveries (by year), production (by year) and reserves (by year, optional),
    estimate missing mass M_t >= 0 that reconciles to EI reserves while favoring a 1960s peak.
    """
    # Make a full year grid
    years = np.arange(start_year, last_year + 1, dtype=int)
    out = pd.DataFrame({"Fuel": fuel, "Year": years})
    # Merge observed/backdated
    out = out.merge(df_back[["Year", "D_obs_EJ", "D_backdated_EJ"]], on="Year", how="left")
    out[["D_obs_EJ", "D_backdated_EJ"]] = out[["D_obs_EJ", "D_backdated_EJ"]].fillna(0.0)

    # Production (fill missing with 0)
    out = out.merge(prod, on="Year", how="left")
    out["Production"] = out["Production"].fillna(0.0)

    # EI reserves (oil/gas): align years we have; DO NOT back-fill to years outside EI coverage
    if reserves is not None and not reserves.empty:
        out = out.merge(reserves, on="Year", how="left")
        # Only forward-fill within reasonable bounds, don't back-fill to early years
        # EI data typically starts around 1980, so don't extend reserves targets to 1900s
        first_valid_year = reserves["year"].min() if "year" in reserves.columns else 1980
        out.loc[out["Year"] < first_valid_year, "Reserves_EI"] = np.nan
        # Forward-fill only (no back-fill to avoid unrealistic early targets)
        out["Reserves_EI"] = out["Reserves_EI"].ffill()
    else:
        out["Reserves_EI"] = np.nan

    # Build target cumulative where reserves available: C_target_t = CumProd_t + Reserves_EI_t
    out["Cum_Prod_EJ"] = out["Production"].cumsum()
    if out["Reserves_EI"].notna().any():
        out["Target_CumDisc_EJ"] = out["Cum_Prod_EJ"] + out["Reserves_EI"]
    else:
        out["Target_CumDisc_EJ"] = np.nan  # (for coal, handled later)

    # Base cumulative (backdated only)
    out["Cum_D_backdated_EJ"] = out["D_backdated_EJ"].cumsum()

    # If we have a reserves path (oil/gas): build linear system on years where EI reserves known
    if out["Target_CumDisc_EJ"].notna().any():
        mask = out["Target_CumDisc_EJ"].notna().values
        y = out.loc[mask, "Target_CumDisc_EJ"].values - out.loc[mask, "Cum_D_backdated_EJ"].values
        # y is the cumulative missing mass we need: cum(M)_t ≈ y_t, with M >= 0
        y = np.maximum(y, 0.0)

        # cumulative operator for selected rows
        T = len(out)
        A_full = lower_triangular_cumsum_matrix(T)   # T×T
        A = A_full[mask, :]

        # Prior weights (peak ~1965; low at 1900)
        w = gaussian_weights(out["Year"].values.astype(float), mu=prior_peak_year, sigma=prior_peak_width)
        m_prior = (w * (y[-1] if len(y) else 0.0))  # scale prior roughly to total missing mass

        # Solve for M (nonnegative)
        M = solve_missing_mass_nnls(A, y, m_prior, lam=prior_lambda)
        out["D_added_EJ"] = M
    else:
        # Coal: no EI reserves time series in DB — anchor to snapshot: total missing needed at end year
        # Compute EI coal reserves snapshot in EJ if possible by summing table; else set to NaN and skip.
        # We approximate target as: at last_year, Cum(D_hat) - Cum(Prod) = R_snapshot.
        # Allocate M with the prior weights only.
        # NOTE: We do not read coal reserves snapshot here to avoid an extra DB call; caller should pass it if desired.
        total_missing = 0.0
        out["D_added_EJ"] = 0.0
        # If caller passed a scalar reserves anchor via reserves DataFrame with single year, it would have been merged.

    # Final series
    out["D_hat_EJ"] = out["D_backdated_EJ"] + out["D_added_EJ"]
    out["Cum_D_hat_EJ"] = out["D_hat_EJ"].cumsum()
    out["Reserves_hat_EJ"] = out["Cum_D_hat_EJ"] - out["Cum_Prod_EJ"]

    return out


# ---------------------- main ----------------------

def main():
    p = argparse.ArgumentParser(description="Build modified discoveries-by-year (Oil, Gas, Coal) from Energy.db")
    p.add_argument("--db", type=Path, required=True, help="Path to Energy.db")
    p.add_argument("--out", type=Path, default=Path("modified_discoveries.csv"), help="Output CSV (long format)")
    p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--peak-year", type=int, default=PRIOR_PEAK_YEAR)
    p.add_argument("--peak-width", type=float, default=PRIOR_PEAK_WIDTH)
    p.add_argument("--prior-lambda", type=float, default=PRIOR_LAMBDA, help="Regularization for M_t prior")
    # per-fuel backdating overrides
    p.add_argument("--oil-alpha", type=float, default=PARAMS["oil"]["alpha"])
    p.add_argument("--oil-tau", type=float, default=PARAMS["oil"]["tau"])
    p.add_argument("--gas-alpha", type=float, default=PARAMS["gas"]["alpha"])
    p.add_argument("--gas-tau", type=float, default=PARAMS["gas"]["tau"])
    p.add_argument("--coal-alpha", type=float, default=PARAMS["coal"]["alpha"])
    p.add_argument("--coal-tau", type=float, default=PARAMS["coal"]["tau"])
    # coal handling
    p.add_argument("--coal-anchor-ej", type=float, default=None,
                   help="Optional EI coal reserves anchor (EJ) at last year; if omitted, uses EI_coal_reserves snapshot if available.")
    args = p.parse_args()
    
    # Validate parameters
    validate_params(args)
    print(f"Starting discovery processing with parameters:")
    print(f"  Start year: {args.start_year}")
    print(f"  Peak year: {args.peak_year} (width: {args.peak_width})")
    print(f"  Oil: alpha={args.oil_alpha}, tau={args.oil_tau}")
    print(f"  Gas: alpha={args.gas_alpha}, tau={args.gas_tau}")
    print(f"  Coal: alpha={args.coal_alpha}, tau={args.coal_tau}")

    with sqlite3.connect(str(args.db)) as conn:
        # Read discoveries (fields)
        oil_fields = read_oil_discoveries(conn)
        gas_fields = read_gas_discoveries(conn)
        coal_fields = read_coal_discoveries(conn)

        # Read production
        oil_prod = oil_production_ej(conn)
        gas_prod = gas_production_ej(conn)
        coal_prod = coal_production_ej(conn)

        # Reserves (EI)
        oil_resv = oil_reserves_ej(conn)
        gas_resv = gas_reserves_ej(conn)

        # last year across series
        last_year = max(get_latest_year(oil_prod, gas_prod, coal_prod),
                        get_latest_year(oil_resv, gas_resv))
        start_year = args.start_year

        # Aggregate & backdate
        oil_back = aggregate_and_backdate(oil_fields, last_year, alpha=args.oil_alpha, tau=args.oil_tau)
        gas_back = aggregate_and_backdate(gas_fields, last_year, alpha=args.gas_alpha, tau=args.gas_tau)
        coal_back = aggregate_and_backdate(coal_fields, last_year, alpha=args.coal_alpha, tau=args.coal_tau)

        # Fit oil & gas with EI reserves paths
        oil_out = build_target_and_fit("Oil", oil_back, oil_prod, oil_resv,
                                       start_year, last_year,
                                       args.peak_year, args.peak_width, args.prior_lambda)
        gas_out = build_target_and_fit("Gas", gas_back, gas_prod, gas_resv,
                                       start_year, last_year,
                                       args.peak_year, args.peak_width, args.prior_lambda)

        # Coal: no reserves time series in DB; anchor to snapshot if available/desired
        coal_anchor_ej = args.coal_anchor_ej
        if coal_anchor_ej is None:
            try:
                _, coal_anchor_ej = coal_reserves_ej_snapshot(conn)
            except Exception:
                coal_anchor_ej = None

        # Build coal target: define Target_CumDisc at last_year only, else use prior
        coal_out = pd.DataFrame({"Fuel": "Coal",
                                 "Year": np.arange(start_year, last_year + 1, dtype=int)})
        coal_out = coal_out.merge(coal_back[["Year", "D_obs_EJ", "D_backdated_EJ"]], on="Year", how="left")
        coal_out[["D_obs_EJ", "D_backdated_EJ"]] = coal_out[["D_obs_EJ", "D_backdated_EJ"]].fillna(0.0)
        coal_out = coal_out.merge(coal_prod, on="Year", how="left")
        coal_out["Production"] = coal_out["Production"].fillna(0.0)
        coal_out["Cum_Prod_EJ"] = coal_out["Production"].cumsum()
        coal_out["Cum_D_backdated_EJ"] = coal_out["D_backdated_EJ"].cumsum()

        # If we have an anchor EJ for reserves at last_year, solve a single-target NNLS to hit it.
        if coal_anchor_ej is not None:
            target_end = coal_out["Cum_Prod_EJ"].iloc[-1] + float(coal_anchor_ej)
            y = np.array([max(0.0, target_end - coal_out["Cum_D_backdated_EJ"].iloc[-1])])
            # Build A as row vector of ones for cumulative at end-year
            T = len(coal_out)
            A = np.ones((1, T), dtype=float)
            # Prior weights (peak ~1965) scaled to y
            w = gaussian_weights(coal_out["Year"].values.astype(float),
                                 mu=args.peak_year, sigma=args.peak_width)
            m_prior = w * y[0]
            M = solve_missing_mass_nnls(A, y, m_prior, lam=args.prior_lambda)
            coal_out["D_added_EJ"] = M
        else:
            coal_out["D_added_EJ"] = 0.0

        coal_out["D_hat_EJ"] = coal_out["D_backdated_EJ"] + coal_out["D_added_EJ"]
        coal_out["Cum_D_hat_EJ"] = coal_out["D_hat_EJ"].cumsum()
        coal_out["Reserves_hat_EJ"] = coal_out["Cum_D_hat_EJ"] - coal_out["Cum_Prod_EJ"]
        coal_out["Reserves_EI"] = np.nan
        if coal_anchor_ej is not None:
            coal_out.loc[coal_out.index[-1], "Reserves_EI"] = coal_anchor_ej

        # Concatenate & restrict to >= start_year
        frames = [oil_out, gas_out, coal_out]
        out = pd.concat(frames, ignore_index=True)
        out = out[out["Year"] >= start_year].copy()

        # Order columns and write
        cols = ["Fuel", "Year",
                "D_obs_EJ", "D_backdated_EJ", "D_added_EJ", "D_hat_EJ",
                "Cum_D_hat_EJ", "Production", "Cum_Prod_EJ",
                "Reserves_hat_EJ", "Reserves_EI"]
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        out = out[cols].sort_values(["Fuel", "Year"]).reset_index(drop=True)
        out.to_csv(args.out, index=False)

        print(f"[OK] Wrote modified discoveries to: {args.out.resolve()}")
        for fuel in ["Oil", "Gas", "Coal"]:
            last = out[out["Fuel"] == fuel].iloc[-1:]
            if not last.empty:
                yr = int(last["Year"].iloc[0])
                res = float(last["Reserves_hat_EJ"].iloc[0])
                print(f"  {fuel}: last year {yr}, Reserves_hat ≈ {res:,.1f} EJ")


if __name__ == "__main__":
    main()
