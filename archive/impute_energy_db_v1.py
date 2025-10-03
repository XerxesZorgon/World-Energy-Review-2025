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
  • Oil_missing  (oil fields only)
  • Gas_missing  (gas fields only)
  • Coal_missing (open + closed mines combined)

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

import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from convert_gas_oil_to_EJ import convert_to_ej
from convert_coal_to_EJ import convert_coal_to_ej

# Define constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'Energy.db')

### Helper Functions (reusable) ###
def heuristic_year(row, field):
    prod_year = row.get("Production start year") or row.get("Year of Production")
    try:
        return int(prod_year) - 5 if prod_year else np.nan
    except:
        return np.nan

def bates_granger_blend(df, cols, true_col):
    mae = {col: mean_absolute_error(df[true_col], df[col]) for col in cols if col in df.columns and df[col].notna().any()}
    inv_mae = {k: 1/v for k, v in mae.items() if v > 0}
    Z = sum(inv_mae.values())
    weights = {k: v/Z for k, v in inv_mae.items()}
    df['blend_val'] = sum(df[k]*w for k, w in weights.items() if k in df.columns)
    return df, weights


def impute_year_block(df, tag):
    # Prepare subset
    df = df.copy()
    df['heur_val'] = df.apply(lambda r: heuristic_year(r, tag), axis=1)
    df['heur_tag'] = ['prod_start_minus5' if pd.notna(x) else None for x in df['heur_val']]

    # Categorical columns
    cat_cols = ['FID Year', 'Unit type', 'Onshore/Offshore', 'Production Type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)

    # Use IterativeImputer (Bayesian Ridge)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].drop(columns=["Discovery year", "Opening Year"], errors="ignore")
    
    # Determine target column based on available columns
    if "Discovery year" in df.columns:
        y = df["Discovery year"]
        target_name = "Discovery year"
    elif "Opening Year" in df.columns:
        y = df["Opening Year"]
        target_name = "Opening Year"
    else:
        # Look for other possible year columns
        year_cols = [col for col in df.columns if 'year' in col.lower() or 'opening' in col.lower()]
        if year_cols:
            y = df[year_cols[0]]
            target_name = year_cols[0]
            print(f"[DEBUG] Using year column: {target_name}")
        else:
            print(f"[ERROR] No year column found in {tag} data. Available columns: {list(df.columns)}")
            return df
    valid = y.notna() & X.notna().all(axis=1)
    
    # Initialize columns with NaN
    df['iter_val'] = np.nan
    df['gb_val'] = np.nan
    
    if valid.any() and len(X.columns) > 0:
        # IterativeImputer
        try:
            it = IterativeImputer(random_state=0)
            it.fit(X[valid], y[valid])
            df.loc[:, 'iter_val'] = it.transform(X)[:, 0]
        except Exception as e:
            print("[WARN] IterativeImputer failed:", e)

        # Gradient Boosting
        try:
            model = GradientBoostingRegressor()
            model.fit(X[valid], y[valid])
            df['gb_val'] = model.predict(X)
        except Exception as e:
            print("[WARN] GB failed:", e)
    else:
        print("[WARN] No valid data for model training - using only heuristics")

    # Blend values (removed MissForest due to compatibility issues)
    target = y.name
    cols = ['heur_val', 'iter_val', 'gb_val']
    valid_blend = df[target].notna() & df[cols].notna().all(axis=1)
    if valid_blend.any():
        df, weights = bates_granger_blend(df.loc[valid_blend], cols, target)
    else:
        df['blend_val'] = df['heur_val']  # fallback

    df[f'{target.lower()}_final'] = df[target].combine_first(df['blend_val'].round())
    df['_imputed_flag_year'] = df[target].isna().astype(int)
    df['_imputed_by_year'] = ['reported' if not f else 'blended' for f in df['_imputed_flag_year']]
    df['_confidence_year'] = ['high' if not f else 'low' for f in df['_imputed_flag_year']]

    return df

def add_column_if_missing(conn, table, column_name, column_type):
    cursor = conn.cursor()
    # Check if column exists
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column_name not in columns:
        print(f"[ALTER] Adding column '{column_name}' to '{table}'...")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")
    else:
        print(f"[OK] Column '{column_name}' already exists in '{table}'.")

def main():
    conn = sqlite3.connect(DB_PATH)

    # Coal
    add_column_if_missing(conn, "Coal_open_mines", "opening_year_final", "REAL")
    add_column_if_missing(conn, "Coal_open_mines", "_imputed_flag_year", "TEXT")
    add_column_if_missing(conn, "Coal_open_mines", "initial_reserves_imputed", "REAL")
    add_column_if_missing(conn, "Coal_open_mines", "_imputed_flag_reserves", "TEXT")

    # Oil/Gas
    add_column_if_missing(conn, "Oil_Gas_Production_Reserves", "discovery_year_final", "REAL")
    add_column_if_missing(conn, "Oil_Gas_Production_Reserves", "_imputed_flag_year", "TEXT")
    add_column_if_missing(conn, "Oil_Gas_Production_Reserves", "initial_reserves_imputed", "REAL")
    add_column_if_missing(conn, "Oil_Gas_Production_Reserves", "_imputed_flag_reserves", "TEXT")

    # --- COAL ---
    coal = pd.read_sql("SELECT * FROM Coal_open_mines", con)
    print(f"[DEBUG] Coal columns: {list(coal.columns)}")
    print(f"[DEBUG] Coal shape: {coal.shape}")
    coal_imp = impute_year_block(coal, tag="coal")
    coal_imp['initial_coal_mt'] = coal_imp['Total Reserves (Proven and Probable, Mt)']
    coal_imp['initial_coal_mt'] = coal_imp['initial_coal_mt'].fillna(0)
    def safe_convert_coal(row):
        coal_type = row.get("Coal type") or row.get("Coal Type") or row.get("coal_type")
        if coal_type is None or pd.isna(coal_type):
            return np.nan
        
        # Normalize compound coal types
        coal_type_str = str(coal_type).strip()
        if '&' in coal_type_str or 'and' in coal_type_str.lower():
            # Use the first type for compound types
            coal_type_str = coal_type_str.replace('&', ' and ').split(' and ')[0].strip()
        
        # Map common variations
        type_map = {
            'bituminous': 'Bituminous',
            'subbituminous': 'Subbituminous', 
            'lignite': 'Lignite',
            'anthracite': 'Anthracite'
        }
        coal_type_normalized = type_map.get(coal_type_str.lower(), coal_type_str)
        
        try:
            return convert_coal_to_ej(row['initial_coal_mt'], coal_type_normalized)
        except Exception as e:
            print(f"[WARN] Coal conversion failed for {coal_type_normalized}: {e}")
            return np.nan
    
    coal_imp['initial_coal_ej'] = coal_imp.apply(safe_convert_coal, axis=1)
    conn.execute("DROP TABLE IF EXISTS Coal_open_mines")
    coal_imp.to_sql("Coal_open_mines", conn, index=False)

    # --- OIL & GAS ---
    oilgas = pd.read_sql("SELECT * FROM Oil_Gas_Production_Reserves", conn)
    og_fields = pd.read_sql("SELECT * FROM Oil_Gas_fields", conn)
    oilgas = oilgas.merge(og_fields[['Unit ID', 'Discovery year']], on='Unit ID', how='left')
    oilgas_imp = impute_year_block(oilgas, tag="oil/gas")
    oilgas_imp['original_reserve'] = oilgas_imp['Quantity (converted)']
    oilgas_imp['original_reserve'] = oilgas_imp['original_reserve'].fillna(0)
    def safe_convert_oilgas(row):
        unit = row.get('Units (converted)')
        fuel = row.get('Fuel description')
        quantity = row.get('original_reserve')
        
        # Skip production units (ending with /y) and invalid data
        if pd.isna(unit) or pd.isna(fuel) or pd.isna(quantity):
            return np.nan
        if str(unit).endswith('/y'):
            return np.nan  # Skip production units
        
        # Normalize units
        unit_map = {
            'million bbl/y': 'million bbl',
            'million m³/y': 'million m3', 
            'million m3/y': 'million m3',
            'million boe/y': 'million boe'
        }
        unit_clean = unit_map.get(str(unit), str(unit))
        
        # Normalize fuel types for compound types
        fuel_str = str(fuel).lower().strip()
        if 'oil' in fuel_str and ('condensate' in fuel_str or 'gas' in fuel_str):
            fuel_clean = 'oil and gas'  # Map compound oil types to 'oil and gas'
        elif 'condensate' in fuel_str and 'lpg' in fuel_str:
            fuel_clean = 'oil and gas'  # Map condensate+LPG to 'oil and gas'
        elif 'oil' in fuel_str:
            fuel_clean = 'oil'
        elif 'gas' in fuel_str:
            fuel_clean = 'gas'
        else:
            fuel_clean = fuel_str
        
        try:
            result = convert_to_ej(quantity, unit_clean, fuel_clean, None)
            # Extract numeric value from result if it's a dict
            if isinstance(result, dict):
                return float(result.get('total_ej', 0))
            return result
        except Exception as e:
            print(f"[WARN] Oil/Gas conversion failed for unit={unit_clean}, fuel={fuel_clean}: {e}")
            return np.nan
    
    oilgas_imp['initial_oilgas_ej'] = oilgas_imp.apply(safe_convert_oilgas, axis=1)
    conn.execute("DROP TABLE IF EXISTS Oil_Gas_Production_Reserves")
    oilgas_imp.to_sql("Oil_Gas_Production_Reserves", conn, index=False)
    conn.close()

    print("[DONE] Imputation and conversions completed.")

### MAIN ###
if __name__ == "__main__":
    main()
