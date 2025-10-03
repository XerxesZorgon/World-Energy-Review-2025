#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_reserves.py
---------------
Fix negative reserves in fossil_fuels_summary.csv using multiple approaches:

1. **Reserve Growth/Backdating**: Apply reserve growth to redistribute future discoveries back to earlier years
2. **EI Reserves Benchmarking**: Use EI proved reserves data to calibrate and validate estimates
3. **Smoothing**: Apply moving averages to reduce year-to-year volatility
4. **Floor Constraints**: Ensure reserves never go below zero

Usage:
  python fix_reserves.py
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

def load_ei_reserves(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load EI proved reserves data for benchmarking with proper unit conversions."""
    reserves_data = {}
    
    with sqlite3.connect(db_path) as con:
        # Oil reserves (thousand million barrels → EJ)
        # Conversion: 1 Gbbl ≈ 6.1 EJ
        try:
            # Get all columns to find 'Total World' row
            oil_df = pd.read_sql("SELECT * FROM EI_oil_proved_reserves", con)
            
            # Find the 'Total World' row (case-insensitive)
            world_mask = oil_df.apply(lambda row: any(
                str(val).lower().strip() == 'total world' 
                for val in row if pd.notna(val)
            ), axis=1)
            
            if world_mask.any():
                world_row = oil_df[world_mask].iloc[0]
                # Extract year columns (numeric columns)
                year_cols = [col for col in oil_df.columns if str(col).isdigit()]
                
                oil_reserves = []
                for year_col in year_cols:
                    year = int(year_col)
                    reserves_gbbl = pd.to_numeric(world_row[year_col], errors='coerce')
                    if pd.notna(reserves_gbbl):
                        reserves_ej = reserves_gbbl * 6.1  # Convert Gbbl to EJ
                        oil_reserves.append({'year': year, 'reserves_EJ': reserves_ej})
                
                reserves_data['Oil'] = pd.DataFrame(oil_reserves)
                print(f"[info] Oil reserves: {len(reserves_data['Oil'])} years loaded")
            else:
                print("[warn] 'Total World' row not found in EI oil reserves")
                reserves_data['Oil'] = pd.DataFrame(columns=['year', 'reserves_EJ'])
                
        except Exception as e:
            print(f"[warn] Could not load EI oil reserves: {e}")
            reserves_data['Oil'] = pd.DataFrame(columns=['year', 'reserves_EJ'])
        
        # Gas reserves (trillion cubic meters → EJ)
        # Conversion: 1 tcm ≈ 38.7 EJ
        try:
            gas_df = pd.read_sql("SELECT * FROM EI_gas_proved_reserves", con)
            
            # Find the 'Total World' row
            world_mask = gas_df.apply(lambda row: any(
                str(val).lower().strip() == 'total world' 
                for val in row if pd.notna(val)
            ), axis=1)
            
            if world_mask.any():
                world_row = gas_df[world_mask].iloc[0]
                year_cols = [col for col in gas_df.columns if str(col).isdigit()]
                
                gas_reserves = []
                for year_col in year_cols:
                    year = int(year_col)
                    reserves_tcm = pd.to_numeric(world_row[year_col], errors='coerce')
                    if pd.notna(reserves_tcm):
                        reserves_ej = reserves_tcm * 38.7  # Convert tcm to EJ
                        gas_reserves.append({'year': year, 'reserves_EJ': reserves_ej})
                
                reserves_data['Gas'] = pd.DataFrame(gas_reserves)
                print(f"[info] Gas reserves: {len(reserves_data['Gas'])} years loaded")
            else:
                print("[warn] 'Total World' row not found in EI gas reserves")
                reserves_data['Gas'] = pd.DataFrame(columns=['year', 'reserves_EJ'])
                
        except Exception as e:
            print(f"[warn] Could not load EI gas reserves: {e}")
            reserves_data['Gas'] = pd.DataFrame(columns=['year', 'reserves_EJ'])
        
        # Coal reserves (million tonnes → EJ)
        # Conversion: 1 Mt coal ≈ 0.022 EJ (assuming average energy density)
        try:
            coal_df = pd.read_sql("SELECT * FROM EI_coal_reserves", con)
            
            # Coal only has 2020 data with world total = 1074108 Mt
            if '2020' in coal_df.columns:
                # Find world total value
                world_total_mt = None
                for _, row in coal_df.iterrows():
                    val = pd.to_numeric(row['2020'], errors='coerce')
                    if pd.notna(val) and val > 1000000:  # Looking for the large world total
                        world_total_mt = val
                        break
                
                if world_total_mt is not None:
                    reserves_ej = world_total_mt * 0.022  # Convert Mt to EJ
                    reserves_data['Coal'] = pd.DataFrame([{
                        'year': 2020, 
                        'reserves_EJ': reserves_ej
                    }])
                    print(f"[info] Coal reserves: 2020 data loaded ({world_total_mt} Mt → {reserves_ej:.1f} EJ)")
                else:
                    print("[warn] Could not find world total in EI coal reserves")
                    reserves_data['Coal'] = pd.DataFrame(columns=['year', 'reserves_EJ'])
            else:
                print("[warn] 2020 column not found in EI coal reserves")
                reserves_data['Coal'] = pd.DataFrame(columns=['year', 'reserves_EJ'])
                
        except Exception as e:
            print(f"[warn] Could not load EI coal reserves: {e}")
            reserves_data['Coal'] = pd.DataFrame(columns=['year', 'reserves_EJ'])
    
    return reserves_data

def _detect_cumulative(series: pd.Series) -> bool:
    """Heuristic to detect cumulative series: mostly non-decreasing and large magnitude."""
    s = pd.to_numeric(series, errors='coerce')
    if s.isna().all() or len(s) < 3:
        return False
    diffs = s.diff()
    nondec_share = (diffs.fillna(0) >= 0).mean()
    return nondec_share > 0.8

def _to_flows_from_cumulative(series: pd.Series) -> pd.Series:
    """Convert cumulative series to flows using first difference (first year = value)."""
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    flows = s.diff()
    if len(s) > 0:
        flows.iloc[0] = s.iloc[0]
    return flows.fillna(0.0)

def _align_ei_to_years(years: pd.Series, ei_df: pd.DataFrame) -> pd.Series:
    """Return EI reserves aligned to requested years with interpolation/ffill where needed."""
    if ei_df is None or ei_df.empty:
        return pd.Series(index=years.index, dtype=float)
    tmp = ei_df.sort_values('year')
    tmp = tmp.dropna(subset=['year', 'reserves_EJ'])
    if tmp.empty:
        return pd.Series(index=years.index, dtype=float)
    # Build a continuous index covering min..max of either series
    yr_min = min(int(years.min()), int(tmp['year'].min()))
    yr_max = max(int(years.max()), int(tmp['year'].max()))
    idx = pd.Index(range(yr_min, yr_max + 1), name='Year')
    series = pd.Series(tmp['reserves_EJ'].values, index=tmp['year'].astype(int)).reindex(idx)
    series = series.interpolate().ffill().bfill()
    # Map back to requested years
    out = pd.Series(series.reindex(years.astype(int)).values, index=years.index)
    return pd.to_numeric(out, errors='coerce')

def load_backdated_discoveries(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load world backdated discoveries (Oil/Gas/Coal) from Discoveries_Production_backdated.
    Tries to be robust to wide/long schemas and units.
    Returns dict: fuel -> DataFrame(year, discoveries_EJ)
    """
    out: Dict[str, pd.DataFrame] = {'Oil': pd.DataFrame(), 'Gas': pd.DataFrame(), 'Coal': pd.DataFrame()}
    if not Path(db_path).exists():
        return out

def load_backdated_oil_columns(db_path: str) -> Optional[pd.DataFrame]:
    """Load backdated oil series from Discoveries_Production_backdated and convert Gb → EJ.
    Expected columns (robust to case/whitespace):
      - Year
      - Discoveries (Gb)
      - Discoveries Cumulative (Gb)
      - Production
      - Production Cumulative
      - Discovery Production Difference
      - Production / Cumulative Production (P/Q)  [ignored]

    Returns a DataFrame with columns:
      Year,
      Oil_discoveries_backdated,
      Oil_discoveries_cumulative_backdated,
      Oil_production_backdated,
      Oil_production_cumulative_backdated,
      Oil_reserves_backdated
    """
    if not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(db_path) as con:
            t = pd.read_sql("SELECT * FROM Discoveries_Production_backdated", con)
    except Exception as e:
        print(f"[warn] Could not read Discoveries_Production_backdated: {e}")
        return None
    # Normalize column names
    norm = {c: str(c).strip().lower() for c in t.columns}
    inv = {v: k for k, v in norm.items()}
    def col_like(keywords):
        for k, v in norm.items():
            s = v
            if all(kw in s for kw in keywords):
                return k
        return None
    year_c = col_like(['year']) or inv.get('year')
    disc_gb_c = col_like(['discoveries', '(gb)'])
    disc_cum_gb_c = col_like(['discoveries', 'cumulative'])
    prod_c = col_like(['production'])
    prod_cum_c = col_like(['production', 'cumulative'])
    diff_c = col_like(['discovery', 'production', 'difference'])

    if not year_c:
        print('[warn] Year column not found in Discoveries_Production_backdated')
        return None

    # Extract and convert (assume Gb → EJ for all 5 numeric columns)
    out = pd.DataFrame()
    out['Year'] = pd.to_numeric(t[year_c], errors='coerce')
    def to_ej(s):
        return pd.to_numeric(s, errors='coerce') * 6.1
    if disc_gb_c and disc_gb_c in t.columns:
        out['Oil_discoveries_backdated'] = to_ej(t[disc_gb_c])
    if disc_cum_gb_c and disc_cum_gb_c in t.columns:
        out['Oil_discoveries_cumulative_backdated'] = to_ej(t[disc_cum_gb_c])
    if prod_c and prod_c in t.columns:
        out['Oil_production_backdated'] = to_ej(t[prod_c])
    if prod_cum_c and prod_cum_c in t.columns:
        out['Oil_production_cumulative_backdated'] = to_ej(t[prod_cum_c])
    if diff_c and diff_c in t.columns:
        out['Oil_reserves_backdated'] = to_ej(t[diff_c])
    out.dropna(subset=['Year'], inplace=True)
    out.sort_values('Year', inplace=True)
    return out
    try:
        with sqlite3.connect(db_path) as con:
            t = pd.read_sql("SELECT * FROM Discoveries_Production_backdated", con)
    except Exception as e:
        print(f"[warn] Could not load Discoveries_Production_backdated: {e}")
        return out

    # Heuristics: check for long format
    cols_lower = {c.lower(): c for c in t.columns}
    def getc(name):
        return cols_lower.get(name, None)

    year_col = next((c for c in t.columns if str(c).lower() in ('year', 'yr')), None)
    fuel_col = next((c for c in t.columns if 'fuel' in str(c).lower()), None)
    qty_cols = [c for c in t.columns if c not in (year_col, fuel_col) and not str(c).lower().startswith('unit')]

    def to_ej(val: float, fuel: str, unit_hint: Optional[str]) -> float:
        if pd.isna(val):
            return np.nan
        # Unit hint parsing
        u = (unit_hint or '').lower()
        if 'ej' in u:
            return float(val)
        if fuel == 'Oil':
            # assume Gbbl if not EJ
            return float(val) * 6.1
        if fuel == 'Gas':
            # assume tcm
            return float(val) * 38.7
        if fuel == 'Coal':
            # assume Mt
            return float(val) * 0.022
        return float(val)

    if year_col and fuel_col and any('quantity' in str(c).lower() for c in t.columns):
        # Likely long format
        qty_col = next(c for c in t.columns if 'quantity' in str(c).lower())
        unit_col = next((c for c in t.columns if 'unit' in str(c).lower()), None)
        # Filter world if present
        world_mask = pd.Series(True, index=t.index)
        for c in t.columns:
            if 'world' == str(c).lower():
                world_mask &= t[c].astype(str).str.lower().str.contains('total world|world')
        tl = t.loc[world_mask, [year_col, fuel_col, qty_col] + ([unit_col] if unit_col else [])].copy()
        tl.rename(columns={year_col: 'year', fuel_col: 'Fuel', qty_col: 'val'}, inplace=True)
        tl['Fuel'] = tl['Fuel'].astype(str).str.title().str.replace('And ', ' & ', regex=False)
        for fuel in ('Oil', 'Gas', 'Coal'):
            sub = tl[tl['Fuel'].str.contains(fuel, case=False, na=False)].copy()
            if sub.empty:
                continue
            unit_hint = sub[unit_col].iloc[0] if unit_col and unit_col in sub.columns and len(sub) else ''
            sub['discoveries_EJ'] = [to_ej(v, fuel, unit_hint) for v in sub['val']]
            sub = sub[['year', 'discoveries_EJ']].dropna()
            sub['year'] = pd.to_numeric(sub['year'], errors='coerce')
            out[fuel] = sub.dropna()
    else:
        # Likely wide format; try to find rows per fuel and numeric year columns
        year_cols = [c for c in t.columns if str(c).isdigit()]
        if year_cols:
            # Find a label column
            label_col = next((c for c in t.columns if c not in year_cols and t[c].dtype == object), None)
            for fuel in ('Oil', 'Gas', 'Coal'):
                if label_col and label_col in t.columns:
                    mask = t[label_col].astype(str).str.lower().str.contains(fuel.lower())
                    if not mask.any():
                        continue
                    row = t[mask].iloc[0]
                    unit_hint = str(row[label_col])
                    records = []
                    for yc in year_cols:
                        v = pd.to_numeric(row[yc], errors='coerce')
                        if pd.notna(v):
                            records.append({'year': int(yc), 'discoveries_EJ': to_ej(v, fuel, unit_hint)})
                    out[fuel] = pd.DataFrame(records)

    # Basic smoothing to reduce jaggedness and emphasize mid-60s peak shape if needed
    for fuel in ('Oil', 'Gas', 'Coal'):
        df = out.get(fuel)
        if df is not None and not df.empty:
            df.sort_values('year', inplace=True)
            s = pd.Series(df['discoveries_EJ'].values).rolling(3, center=True, min_periods=1).mean()
            df['discoveries_EJ'] = s.values
            out[fuel] = df

    return out

def apply_reserve_growth(discoveries: pd.Series, years: pd.Series, 
                        growth_rate: float = 0.3, horizon: int = 30) -> pd.Series:
    """Apply reserve growth by redistributing future discoveries back to earlier years."""
    if discoveries.empty or growth_rate <= 0:
        return discoveries
    
    # Create exponential decay weights
    ks = np.arange(1, horizon + 1)
    tau = horizon / 5.0
    weights = np.exp(-ks / tau)
    weights = weights / weights.sum() * growth_rate
    
    # Apply backdating
    discoveries_adj = discoveries.copy()
    year_to_idx = {year: idx for idx, year in enumerate(years)}
    
    for idx, year in enumerate(years):
        backdate_amount = 0.0
        for k, weight in enumerate(ks):
            future_year = year + k
            if future_year in year_to_idx:
                future_idx = year_to_idx[future_year]
                backdate_amount += weight * discoveries.iloc[future_idx]
        
        # Add backdated amount to current year
        discoveries_adj.iloc[idx] += backdate_amount
    
    return discoveries_adj

def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply moving average smoothing to reduce volatility."""
    return series.rolling(window=window, center=True, min_periods=1).mean()

def calibrate_with_ei(computed_reserves: pd.Series, years: pd.Series, 
                     ei_reserves: pd.DataFrame, fuel: str) -> pd.Series:
    """Deprecated in EI-anchored mode. Retained for fallback without EI data."""
    if ei_reserves is None or ei_reserves.empty:
        return computed_reserves
    # Prefer strict EI anchoring; keep scale calibration only if requested in future.
    aligned = _align_ei_to_years(years, ei_reserves)
    return pd.to_numeric(aligned, errors='coerce')

def fix_fuel_reserves(df: pd.DataFrame, fuel: str, ei_data: Dict[str, pd.DataFrame], backdated: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """Fix reserves for a specific fuel type."""
    print(f"\n=== Fixing {fuel} Reserves ===")
    
    # Extract fuel-specific columns
    disc_col = f"{fuel}_Discoveries"
    prod_col = f"{fuel}_Production"
    reserves_col = f"{fuel}_Reserves"
    
    if not all(col in df.columns for col in [disc_col, prod_col, reserves_col]):
        print(f"[warn] Missing columns for {fuel}, skipping")
        return df
    
    # Get the data
    years = pd.to_numeric(df['Year'], errors='coerce')
    discoveries = pd.to_numeric(df[disc_col], errors='coerce').fillna(0.0)
    production = pd.to_numeric(df[prod_col], errors='coerce').fillna(0.0).clip(lower=0)
    original_reserves = df[reserves_col].copy()
    
    print(f"Original reserves: min={original_reserves.min():.1f}, max={original_reserves.max():.1f}")
    print(f"Negative reserves: {(original_reserves < 0).sum()} out of {len(original_reserves)} years")
    
    # Ensure discoveries are annual flows, not cumulative
    # For Oil, use the backdated columns that were already loaded and merged into the DataFrame
    if fuel == 'Oil' and f"{fuel}_discoveries_backdated" in df.columns:
        backdated_col = df[f"{fuel}_discoveries_backdated"]
        if not backdated_col.isna().all():
            print("Using Oil backdated discoveries from merged columns")
            discoveries_flows = pd.to_numeric(backdated_col, errors='coerce').fillna(0.0)
            print(f"Backdated {fuel} discoveries: {discoveries_flows.sum():.1f} EJ total, peak year {years[discoveries_flows.idxmax()]}")
        else:
            print("Oil backdated column exists but is empty, using original discovery data")
            if _detect_cumulative(discoveries):
                print("[info] Detected cumulative discoveries; converting to annual flows via differencing")
                discoveries_flows = _to_flows_from_cumulative(discoveries)
            else:
                discoveries_flows = discoveries
    elif backdated and backdated.get(fuel) is not None and not backdated[fuel].empty:
        print("Using backdated world discoveries from DB")
        bd = backdated[fuel]
        # align to CSV years
        bd_series = pd.Series(bd['discoveries_EJ'].values, index=bd['year'].astype(int))
        aligned = bd_series.reindex(years.astype(int))
        aligned = aligned.interpolate().fillna(0.0)  # Use 0 for missing years, not forward/backward fill
        discoveries_flows = pd.to_numeric(aligned.values, errors='coerce')
        discoveries_flows = pd.Series(discoveries_flows, index=df.index).fillna(0.0)
        print(f"Backdated {fuel} discoveries: {discoveries_flows.sum():.1f} EJ total, peak year {years[discoveries_flows.idxmax()]}")
    else:
        if _detect_cumulative(discoveries):
            print("[info] Detected cumulative discoveries; converting to annual flows via differencing")
            discoveries_flows = _to_flows_from_cumulative(discoveries)
        else:
            discoveries_flows = discoveries

    # Prefer EI-anchored reserves if available
    ei_df = ei_data.get(fuel, pd.DataFrame())
    if ei_df is not None and not ei_df.empty:
        print("Anchoring reserves to EI world totals with shape-preserving discovery calibration")
        reserves_ei_aligned = _align_ei_to_years(years, ei_df)
        
        # Shape-preserving approach: use backdated discovery shape, scale to match EI
        if backdated and backdated.get(fuel) is not None and not backdated[fuel].empty:
            print(f"Using backdated {fuel} discovery shape as template")
            
            # Get backdated discovery flows (annual, not cumulative) - these start from zero
            bd_flows = pd.Series(discoveries_flows)
            
            # Find optimal scale factor by minimizing difference between:
            # - EI reserves at multiple years  
            # - Reserves computed from scaled backdated discoveries + production
            def compute_reserves_from_scaled_discoveries(scale_factor):
                scaled_discoveries = bd_flows * scale_factor
                # Build cumulative from zero (not inheriting any prior values)
                cum_d = scaled_discoveries.cumsum()
                cum_p = production.cumsum()
                return cum_d - cum_p
            
            # Try different scale factors and find best match to EI
            scales = np.logspace(-2, 2, 100)  # 0.01 to 100
            best_scale = 1.0
            best_error = float('inf')
            
            for scale in scales:
                computed_reserves = compute_reserves_from_scaled_discoveries(scale)
                # Compare to EI reserves where both exist and are positive
                mask = (pd.notna(computed_reserves) & 
                       pd.notna(reserves_ei_aligned) & 
                       (reserves_ei_aligned > 0))
                
                if mask.sum() >= 3:  # Need at least 3 points for comparison
                    # Use relative error to handle different magnitudes
                    rel_error = np.abs((computed_reserves[mask] - reserves_ei_aligned[mask]) / 
                                     reserves_ei_aligned[mask]).mean()
                    if rel_error < best_error:
                        best_error = rel_error
                        best_scale = scale
            
            print(f"Optimal scale factor for {fuel}: {best_scale:.4f} (error: {best_error:.4f})")
            
            # Apply optimal scaling to annual discoveries
            discoveries_adj = bd_flows * best_scale
            
            # Build cumulative discoveries starting from zero (first year = first discovery)
            # This completely ignores any pre-existing cumulative values from the CSV
            cum_discoveries_adj = discoveries_adj.cumsum()
            
            # For reserves, blend scaled discoveries with EI anchoring
            # Use EI where available, computed reserves elsewhere
            reserves_from_scaled = cum_discoveries_adj - production.cumsum()
            reserves_final = reserves_ei_aligned.copy()
            
            # Fill missing EI years with computed reserves
            missing_ei = pd.isna(reserves_final)
            reserves_final[missing_ei] = reserves_from_scaled[missing_ei]
            reserves_final = reserves_final.clip(lower=0)
            
            print(f"Final {fuel} cumulative discoveries: {cum_discoveries_adj.iloc[0]:.1f} EJ (1900) → {cum_discoveries_adj.iloc[-1]:.1f} EJ (2024)")
            
        else:
            # No backdated data: fall back to pure EI back-calculation
            print(f"No backdated {fuel} data; using EI back-calculation")
            delta_r = pd.to_numeric(reserves_ei_aligned, errors='coerce').diff()
            if len(delta_r) > 0:
                delta_r.iloc[0] = reserves_ei_aligned.iloc[0]
            discoveries_adj = (delta_r.fillna(0.0) + production).clip(lower=0)
            cum_discoveries_adj = discoveries_adj.cumsum()
            reserves_final = reserves_ei_aligned.clip(lower=0)
    else:
        # Fallback: compute reserves from discoveries and production
        print("No EI data available; using adjusted discoveries method")
        # Optionally apply mild reserve growth only in fallback path to avoid inflation
        discoveries_adj = apply_reserve_growth(discoveries_flows, years, growth_rate=0.1, horizon=20)
        cum_discoveries_adj = discoveries_adj.cumsum()
        cum_production = production.cumsum()
        reserves_raw = cum_discoveries_adj - cum_production
        # If EI missing, scale to sensible world targets for Gas/Coal
        target_totals = {
            'Gas': 7500.0,     # EJ
            'Coal': 25000.0,   # EJ (midpoint of 23,000–27,600)
        }
        if fuel in target_totals and pd.notna(reserves_raw.iloc[-1]) and reserves_raw.iloc[-1] > 0:
            scale = target_totals[fuel] / float(reserves_raw.iloc[-1])
            if np.isfinite(scale) and scale > 0:
                discoveries_adj = discoveries_adj * scale
                cum_discoveries_adj = discoveries_adj.cumsum()
                reserves_raw = cum_discoveries_adj - cum_production
        reserves_smoothed = smooth_series(reserves_raw, window=5)
        reserves_final = np.maximum(reserves_smoothed, 0.0)
    
    print(f"Final reserves: min={reserves_final.min():.1f}, max={reserves_final.max():.1f}")
    print(f"Negative reserves after fix: {(reserves_final < 0).sum()}")
    
    # Add new columns to DataFrame
    df[f"{fuel}_Discoveries_Adjusted"] = discoveries_adj
    df[f"{fuel}_Cumulative_Discoveries_Adjusted"] = cum_discoveries_adj
    df[f"{fuel}_Reserves_Fixed"] = reserves_final
    
    # Diagnostic: check if cumulative starts small as expected
    if len(cum_discoveries_adj) > 0:
        first_val = cum_discoveries_adj.iloc[0]
        last_val = cum_discoveries_adj.iloc[-1]
        print(f"{fuel} cumulative discoveries: {first_val:.1f} EJ (first) → {last_val:.1f} EJ (last)")
        if first_val > 1000:
            print(f"[WARNING] {fuel} cumulative discoveries start too high! Expected small values in 1900.")
    
    # Calculate improvement metrics
    neg_before = (original_reserves < 0).sum()
    neg_after = (reserves_final < 0).sum()
    print(f"Improvement: {neg_before} → {neg_after} negative values ({neg_before - neg_after} fixed)")
    
    return df

def main():
    """Main function to fix reserves in fossil_fuels_summary.csv"""
    
    # File paths
    csv_path = Path("fossil_fuels_summary.csv")
    db_path = Path("data/Energy.db")
    output_path = Path("fossil_fuels_summary_fixed_v2.csv")
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
    
    if not db_path.exists():
        print(f"Warning: {db_path} not found, proceeding without EI calibration")
    
    print("Loading fossil fuels summary...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Clip to 1900-2024 as requested
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[(df['Year'] >= 1900) & (df['Year'] <= 2024)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Clipped to 1900-2024: {len(df)} rows remaining")
    
    # Load EI reserves data
    print("Loading EI reserves data...")
    ei_data = load_ei_reserves(str(db_path)) if db_path.exists() else {}
    backdated = load_backdated_discoveries(str(db_path)) if db_path.exists() else {}

    # Merge oil backdated columns (Gb→EJ) just after Year
    oil_backdated = load_backdated_oil_columns(str(db_path)) if db_path.exists() else None
    if oil_backdated is not None and not oil_backdated.empty:
        df = df.merge(oil_backdated, on='Year', how='left')
        # Reorder to place these five oil columns right after Year
        after_year = [
            'Oil_discoveries_backdated',
            'Oil_discoveries_cumulative_backdated',
            'Oil_production_backdated',
            'Oil_production_cumulative_backdated',
            'Oil_reserves_backdated',
        ]
        cols = list(df.columns)
        if 'Year' in cols:
            rest = [c for c in cols if c not in ['Year'] + after_year]
            df = df[['Year'] + after_year + rest]
    
    for fuel, ei_df in ei_data.items():
        if not ei_df.empty:
            print(f"EI {fuel} reserves: {len(ei_df)} years ({ei_df['year'].min()}-{ei_df['year'].max()})")
        else:
            print(f"EI {fuel} reserves: No data available")
    
    # Fix reserves for each fuel
    fuels = ['Oil', 'Gas', 'Coal']
    for fuel in fuels:
        df = fix_fuel_reserves(df, fuel, ei_data, backdated)

    # Reorder columns to keep Oil, Gas, Coal groups together
    def group_cols(prefix):
        return [c for c in df.columns if c.startswith(prefix + '_')]
    base_cols = [c for c in df.columns if not any(c.startswith(p + '_') for p in fuels)]
    new_order = base_cols + group_cols('Oil') + group_cols('Gas') + group_cols('Coal')
    df = df.reindex(columns=new_order)
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Summary report
    print("\n=== SUMMARY REPORT ===")
    for fuel in fuels:
        reserves_col = f"{fuel}_Reserves"
        fixed_col = f"{fuel}_Reserves_Fixed"
        
        if reserves_col in df.columns and fixed_col in df.columns:
            orig_neg = (df[reserves_col] < 0).sum()
            fixed_neg = (df[fixed_col] < 0).sum()
            print(f"{fuel:5s}: {orig_neg:3d} → {fixed_neg:3d} negative values")
    
    print(f"\nFixed data saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
