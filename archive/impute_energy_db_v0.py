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

import os
print("[DEBUG PATH] Running script at:", os.path.abspath(__file__))
import sys; sys.stdout.flush()

print("[DEBUG] impute_energy_db.py script STARTED") 
import sys; sys.stdout.flush()

import os
import sys
import math
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import argparse

# --- PATH CONSTANTS (import_batch.py style) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'Energy.db')

# --- ENVIRONMENT TEST: Toy fit for HistGradientBoostingRegressor ---
print("[ENV TEST] numpy version:", np.__version__)
try:
    import sklearn
    print("[ENV TEST] scikit-learn version:", sklearn.__version__)
except ImportError:
    print("[ENV TEST] scikit-learn not found!")
    sys.exit(1)
try:
    X_toy = np.arange(10).reshape(-1, 1).astype(float)
    y_toy = np.arange(10).astype(float)
    hgb = HistGradientBoostingRegressor()
    hgb.fit(X_toy, y_toy)
    print("[ENV TEST] Toy fit succeeded.")
except Exception as e:
    print(f"[ENV TEST] Toy fit failed: {e}")
    sys.exit(1)

# ── optional extras ───────────────────────────────────────────────────────────
try:
    from missingforest import MissForest
    HAS_MISSFOREST = True
except ImportError:
    HAS_MISSFOREST = False

try:
    import category_encoders as ce
    HAS_CE = True
except ImportError:
    HAS_CE = False

# ── local helpers (unit conversion) ───────────────────────────────────────────
try:
    from convert_gas_oil_to_EJ import convert_to_ej as _oilgas_to_ej
except ImportError:
    try:
        from convert_gas_oil_to_EJ import convert_to_ej as _oilgas_to_ej
    except ImportError:
        def _oilgas_to_ej(qty, unit, fuel_type="oil", gas_ratio=None):
            raise ImportError("convert_gas_oil_to_EJ.py not found or convert_to_ej unavailable")

try:
    from convert_coal_to_EJ import convert_coal_to_ej as _coal_to_ej
except ImportError:
    try:
        from convert_coal_to_EJ import convert_coal_to_ej as _coal_to_ej
    except ImportError:
        def _coal_to_ej(qty, coal_type):
            raise ImportError("convert_coal_to_EJ.py not found or convert_coal_to_ej unavailable")

# ── configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 0
CHUNK_SIZE = 10_000  

def parse_args():
    p = argparse.ArgumentParser(description="Impute missing energy data into SQLite DB")
    p.add_argument("--db", type=Path, default=DB_PATH, help="Path to Energy.db (default: auto-detect)")
    return p.parse_args()

# ── utilities ─────────────────────────────────────────────────────────────────

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Save RAM by down‑casting numeric dtypes."""
    for col in df.select_dtypes(include=["float64", "int64"]):
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def target_encode(train_X, y, pred_X):
    """Return (X_train_enc, X_pred_enc). Falls back to frequency‑encode."""
    cat_cols = train_X.select_dtypes(include=["object", "category"]).columns
    if not len(cat_cols):
        return train_X, pred_X  # nothing to encode

    fit_mask = y.notna()
    if HAS_CE:
        enc = ce.TargetEncoder(cols=cat_cols, smoothing=10, min_samples_leaf=5)
        # Fit only on non-missing y
        enc.fit(train_X.loc[fit_mask], y.loc[fit_mask])
        train_enc = enc.transform(train_X)
        pred_enc = enc.transform(pred_X)
        return train_enc, pred_enc

    # fallback
    freqs = {c: train_X.loc[fit_mask, c].value_counts(normalize=True) for c in cat_cols}
    train_enc = train_X.copy()
    pred_enc = pred_X.copy()
    for c in cat_cols:
        train_enc[c] = train_X[c].map(freqs[c]).fillna(0)
        pred_enc[c] = pred_X[c].map(freqs[c]).fillna(0)
    return train_enc, pred_enc


def mae_blend(df: pd.DataFrame, cand_cols: list, true_col: str) -> pd.Series:
    """Inverse-MAE Bates & Granger blend across candidate columns (NaN-safe)."""
    eps = 1e-9
    weights = {}

    for c in cand_cols:
        if c not in df.columns:
            continue
        # rows where both y_true and y_pred are present
        valid = df[true_col].notna() & df[c].notna()
        if valid.any():
            mae = mean_absolute_error(df.loc[valid, true_col], df.loc[valid, c])
            weights[c] = 1.0 / (mae + eps)          # inverse-MAE weight
        else:
            # if no overlap, give the column zero weight
            weights[c] = 0.0

    if not any(weights.values()):
        # nothing usable – return a NaN series
        return pd.Series(np.nan, index=df.index)

    total = sum(weights.values())
    out = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if w:                                    # skip zero-weight cols
            out += df[col].fillna(0) * (w / total)
    return out

# ── heuristic helpers ─────────────────────────────────────────────────────────

def safe_numeric(val):
    try:
        return pd.to_numeric(val, errors='coerce')
    except Exception:
        return np.nan

def heuristic_discovery(row):
    """Return (year, tag) tuple for oil/gas discovery year heuristics."""
    prod_start = safe_numeric(row.get("Production start year"))
    if pd.notna(prod_start):
        return prod_start - 5, "prod_start_minus5"
    fid_year = safe_numeric(row.get("FID Year"))
    if pd.notna(fid_year):
        return fid_year - 3, "fid_minus3"
    status_year = safe_numeric(row.get("Status year"))
    if pd.notna(status_year) and isinstance(row.get("Status"), str):
        if "discover" in row["Status"].lower():
            return status_year, "status_year"
    return np.nan, "none"


def heuristic_opening(row):
    yop = safe_numeric(row.get("Year of Production"))
    if pd.notna(yop):
        return yop, "first_prod"
    closing = safe_numeric(row.get("Closing Year"))
    lom = safe_numeric(row.get("Reported Life of Mine"))
    if pd.notna(closing) and pd.notna(lom):
        return closing - lom, "closing_minus_lom"
    return np.nan, "none"

# ── main processing blocks ────────────────────────────────────────────────────


def impute_year(df: pd.DataFrame, target_col: str, feat_cols: list, tag: str):
    """
    Impute year columns using heuristics, iterative imputer, and gradient boosting.
    Includes both numeric and encoded categorical features. Robustly builds output columns.
    """
    print("[impute_year DIAG] Function called with tag:", tag)
    import sys; sys.stdout.flush()
    out = df.copy()

    # Coerce year columns to numeric
    out[target_col] = pd.to_numeric(out[target_col], errors='coerce')

    # Heuristic imputation
    print(f"• {tag}: running heuristic…")
    heur_vals, heur_tags = zip(*out.apply(heuristic_opening if tag == "coal" else heuristic_discovery, axis=1))
    out["heur_val"] = heur_vals
    out["heur_tag"] = heur_tags

    # Prepare features: numeric + selected categoricals
    from sklearn.preprocessing import OrdinalEncoder
    # Select categorical columns to encode (avoid high-cardinality)
    cat_candidates = [
        'FID Year', 'Production start year', 'Unit type', 'Onshore/Offshore',
        'Production Type', 'Basin', 'Country/Area', 'Fuel description'
    ]
    cat_cols = [col for col in cat_candidates if col in feat_cols and col in out.columns and out[col].nunique() < 100 and out[col].notna().sum() > 0]
    num_cols = [col for col in feat_cols if col in out.columns and pd.api.types.is_numeric_dtype(out[col])]
    all_feat_cols = num_cols + cat_cols
    print(f"[impute_year DIAG] Numeric features: {num_cols}")
    print(f"[impute_year DIAG] Categorical features: {cat_cols}")
    X = out[all_feat_cols].copy() if all_feat_cols else pd.DataFrame(index=out.index)
    # Encode categoricals
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
    # IterativeImputer
    iter_val = np.full(len(out), np.nan)
    if num_cols:
        try:
            iter_imp = IterativeImputer(estimator=BayesianRidge(), random_state=RANDOM_STATE, max_iter=15)
            subcols = [target_col] + num_cols
            arr = iter_imp.fit_transform(out[subcols])
            iter_val = arr[:, 0]
            print(f"[impute_year DIAG] IterativeImputer succeeded.")
        except Exception as e:
            print(f"[impute_year DIAG] IterativeImputer failed: {e}")
    else:
        print("[impute_year DIAG] No numeric features for IterativeImputer.")
    out["iter_val"] = iter_val
    # Gradient Boosting
    gb_val = np.full(len(out), np.nan)
    if X.shape[1] > 0:
        from sklearn.ensemble import HistGradientBoostingRegressor
        gb = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        y = out[target_col]
        fit_mask = y.notna() & (~X.isna().any(axis=1))
        valid_X = X.loc[fit_mask]
        valid_y = y[fit_mask]
        if valid_X.shape[1] == 0 or valid_X.shape[0] == 0:
            print("[WARN] No usable features found for GradientBoosting; skipping.")
        else:
            try:
                gb.fit(valid_X, valid_y)
                gb_val = gb.predict(X)
                print(f"[impute_year DIAG] GradientBoosting fit succeeded.")
            except Exception as e:
                print(f"[impute_year DIAG] GradientBoosting fit failed: {e}")
    else:
        print("[impute_year DIAG] No features for GradientBoosting.")
    out["gb_val"] = gb_val
    # Build final columns
    if tag == "oil/gas":
        final_col = "discovery year_final"
    elif tag == "coal":
        final_col = "opening year_final"
    else:
        final_col = f"{target_col.lower()}_final"
    out[final_col] = out["heur_val"]
    out[final_col] = out[final_col].combine_first(pd.Series(iter_val, index=out.index))
    out[final_col] = out[final_col].combine_first(pd.Series(gb_val, index=out.index))
    # Ensure imputed years are integers (or NA)
    out[final_col] = out[final_col].round().astype("Int64")
    # Imputation flags/metadata
    out["_imputed_flag_year"] = out[target_col].isna() & out[final_col].notna()
    def _by(row):
        if not pd.isna(row["heur_val"]): return "heur"
        if not pd.isna(row["iter_val"]): return "iter"
        if not pd.isna(row["gb_val"]): return "gb"
        return None
    def _conf(row):
        if not pd.isna(row["heur_val"]): return "high"
        if not pd.isna(row["iter_val"]): return "medium"
        if not pd.isna(row["gb_val"]): return "low"
        return None
    # Check for infinite values
    print(f"[DIAGNOSTIC] Infinite values in features: {(~np.isfinite(valid_X)).sum().sum()}")
    print(f"[DIAGNOSTIC] Infinite values in target: {(~np.isfinite(valid_y)).sum()}")
    # Print dtypes
    print(f"[DIAGNOSTIC] Dtypes of features:\n{valid_X.dtypes}")
    print(f"[DIAGNOSTIC] Dtype of target: {valid_y.dtype}")
    # Print head
    print(f"[DIAGNOSTIC] valid_X.head():\n{valid_X.head()}")
    print(f"[DIAGNOSTIC] valid_y.head():\n{valid_y.head()}")
    # Assert all finite and numeric
    if valid_X.isna().any().any() or valid_y.isna().any() or (~np.isfinite(valid_X)).any().any() or (~np.isfinite(valid_y)).any():
        print(f"[ERROR] NaNs or infinite values remain in features or target before fit. Printing first 5 problematic rows:")
        print(valid_X[(valid_X.isna().any(axis=1)) | (~np.isfinite(valid_X)).any(axis=1)].head())
        print(valid_y[(valid_y.isna()) | (~np.isfinite(valid_y))].head())
        print(f"Aborting fit for {tag} due to persistent NaNs or infinite values.")
        out["gb_val"] = np.nan
    elif not all(np.issubdtype(dt, np.number) for dt in valid_X.dtypes):
        print(f"[ERROR] Non-numeric dtypes found in features. Aborting fit for {tag}.")
        out["gb_val"] = np.nan
    elif not np.issubdtype(valid_y.dtype, np.number):
        print(f"[ERROR] Non-numeric dtype found in target. Aborting fit for {tag}.")
        out["gb_val"] = np.nan
    else:
        print(f"[DIAGNOSTIC] Dtypes before conversion:\n{valid_X.dtypes}\nTarget: {valid_y.dtype}")
        try:
            valid_X_arr = valid_X.astype(np.float64)
            valid_y_arr = valid_y.astype(np.float64)
        except Exception as e:
            print(f"[ERROR] Failed to convert to float64: {e}")
            print("valid_X sample:\n", valid_X.head())
            print("valid_y sample:\n", valid_y.head())
            out["gb_val"] = np.nan
            return out
            print(f"[DIAGNOSTIC] Dtypes after conversion:\n{valid_X_arr.dtypes if hasattr(valid_X_arr, 'dtypes') else type(valid_X_arr)}\nTarget: {valid_y_arr.dtype}")
            print(f"[DIAGNOSTIC] valid_X_arr.shape: {valid_X_arr.shape}, valid_y_arr.shape: {valid_y_arr.shape}")
            print(f"[DIAGNOSTIC] Any NaN in valid_X_arr: {np.isnan(valid_X_arr).any()}, Any NaN in valid_y_arr: {np.isnan(valid_y_arr).any()}")
            print(f"[DIAGNOSTIC] Any inf in valid_X_arr: {np.isinf(valid_X_arr).any()}, Any inf in valid_y_arr: {np.isinf(valid_y_arr).any()}")
            print(f"[DIAGNOSTIC] First 10 rows of valid_X_arr:\n{valid_X_arr[:10]}")
            print(f"[DIAGNOSTIC] First 10 values of valid_y_arr:\n{valid_y_arr[:10]}")
            # Print unique values and types for each feature and target (first 20 rows)
            for col in valid_X.columns:
                print(f"[DIAGNOSTIC] Unique values in feature '{col}': {pd.unique(valid_X[col].head(20))}")
                print(f"[DIAGNOSTIC] Types in feature '{col}': {[type(x) for x in valid_X[col].head(20)]}")
            print(f"[DIAGNOSTIC] Unique values in target: {pd.unique(valid_y.head(20))}")
            print(f"[DIAGNOSTIC] Types in target: {[type(x) for x in valid_y.head(20)]}")
            # Try minimal fit on first 10 rows
            try:
                gb.fit(valid_X_arr[:10], valid_y_arr[:10])
                print(f"[DIAGNOSTIC] Minimal fit on first 10 rows succeeded.")
            except Exception as e:
                print(f"[ERROR] Minimal fit on first 10 rows failed: {e}")
                out["gb_val"] = np.nan
                return out
            # Try LinearRegression as a control
            from sklearn.linear_model import LinearRegression
            try:
                lr = LinearRegression()
                lr.fit(valid_X_arr, valid_y_arr)
                print(f"[DIAGNOSTIC] LinearRegression fit succeeded.")
            except Exception as e:
                print(f"[ERROR] LinearRegression fit failed: {e}")
            # Print all values in first 50 rows
            print(f"[DIAGNOSTIC] First 50 rows of valid_X_arr:\n{valid_X_arr[:50]}")
            print(f"[DIAGNOSTIC] First 50 values of valid_y_arr:\n{valid_y_arr[:50]}")
            print(f"[DIAGNOSTIC] np.isnan(valid_X_arr).sum(): {np.isnan(valid_X_arr).sum()}")
            print(f"[DIAGNOSTIC] np.isinf(valid_X_arr).sum(): {np.isinf(valid_X_arr).sum()}")
            print(f"[DIAGNOSTIC] np.isnan(valid_y_arr).sum(): {np.isnan(valid_y_arr).sum()}")
            print(f"[DIAGNOSTIC] np.isinf(valid_y_arr).sum(): {np.isinf(valid_y_arr).sum()}")
            # Fit row by row to find problematic row
            for i in range(1, len(valid_X_arr)+1):
                try:
                    gb.fit(valid_X_arr[:i], valid_y_arr[:i])
                except Exception as e:
                    print(f"[ERROR] GradientBoosting fit failed at row {i}: {e}")
                    print(f"Row {i} values X: {valid_X_arr[i-1]}")
                    print(f"Row {i} value y: {valid_y_arr[i-1]}")
                    out["gb_val"] = np.nan
                    return out
            # Print dtypes and sample of raw DataFrame before conversion
            print(f"[RAW DIAG] valid_X dtypes: {valid_X.dtypes}")
            print(f"[RAW DIAG] valid_y dtype: {valid_y.dtype}")
            print(f"[RAW DIAG] First 10 rows of valid_X:\n{valid_X.head(10)}")
            print(f"[RAW DIAG] First 10 values of valid_y:\n{valid_y.head(10)}")
            # Export full columns to CSV for inspection
            try:
                valid_X.to_csv('raw_X_columns.csv', index=False)
                valid_y.to_csv('raw_y_column.csv', index=False, header=True)
                print(f"[RAW DIAG] Exported valid_X to raw_X_columns.csv and valid_y to raw_y_column.csv")
            except Exception as e:
                print(f"[RAW DIAG] Error exporting columns to CSV: {e}")
                out["gb_val"] = np.nan
                return out
            # Force cast to float64 and print errors if any
            try:
                valid_X_cast = valid_X.astype('float64')
                print(f"[CAST DIAG] valid_X dtypes after cast: {valid_X_cast.dtypes}")
            except Exception as e:
                print(f"[CAST DIAG] Error casting valid_X to float64: {e}")
                out["gb_val"] = np.nan
                return out
            try:
                valid_y_cast = valid_y.astype('float64')
                print(f"[CAST DIAG] valid_y dtype after cast: {valid_y_cast.dtype}")
            except Exception as e:
                print(f"[CAST DIAG] Error casting valid_y to float64: {e}")
                out["gb_val"] = np.nan
                return out
            # Ensure arrays are numpy arrays for NaN checks
            valid_X_arr_np = np.asarray(valid_X_cast)
            valid_y_arr_np = np.asarray(valid_y_cast)
            print(f"[DIAGNOSTIC] Type of valid_X_arr: {type(valid_X_arr_np)}")
            print(f"[DIAGNOSTIC] Type of valid_y_arr: {type(valid_y_arr_np)}")
            print(f"[DIAGNOSTIC] Shape of valid_X_arr: {valid_X_arr_np.shape}")
            print(f"[DIAGNOSTIC] Shape of valid_y_arr: {valid_y_arr_np.shape}")
            print(f"[DIAGNOSTIC] Contents of valid_X_arr: {valid_X_arr_np}")
            print(f"[DIAGNOSTIC] Contents of valid_y_arr: {valid_y_arr_np}")
            print(f"[DIAGNOSTIC] valid_X_arr NaNs: {np.isnan(valid_X_arr_np).sum()}, all NaN: {np.isnan(valid_X_arr_np).all()}, infs: {np.isinf(valid_X_arr_np).sum()}, non-NaN count: {np.count_nonzero(~np.isnan(valid_X_arr_np))}")
            print(f"[DIAGNOSTIC] valid_y_arr NaNs: {np.isnan(valid_y_arr_np).sum()}, all NaN: {np.isnan(valid_y_arr_np).all()}, infs: {np.isinf(valid_y_arr_np).sum()}, non-NaN count: {np.count_nonzero(~np.isnan(valid_y_arr_np))}")
            # Force-fill NaN and inf with -1
            print(f"[DIAGNOSTIC] NaNs in valid_X_arr_np before fill: {np.isnan(valid_X_arr_np).sum()}")
            print(f"[DIAGNOSTIC] infs in valid_X_arr_np before fill: {np.isinf(valid_X_arr_np).sum()}")
            print(f"[DIAGNOSTIC] NaNs in valid_y_arr_np before fill: {np.isnan(valid_y_arr_np).sum()}")
            print(f"[DIAGNOSTIC] infs in valid_y_arr_np before fill: {np.isinf(valid_y_arr_np).sum()}")
            valid_X_arr_np = np.where(np.isnan(valid_X_arr_np) | np.isinf(valid_X_arr_np), -1, valid_X_arr_np)
            valid_y_arr_np = np.where(np.isnan(valid_y_arr_np) | np.isinf(valid_y_arr_np), -1, valid_y_arr_np)
            print(f"[DIAGNOSTIC] NaNs in valid_X_arr_np after fill: {np.isnan(valid_X_arr_np).sum()}")
            print(f"[DIAGNOSTIC] infs in valid_X_arr_np after fill: {np.isinf(valid_X_arr_np).sum()}")
            print(f"[DIAGNOSTIC] NaNs in valid_y_arr_np after fill: {np.isnan(valid_y_arr_np).sum()}")
            print(f"[DIAGNOSTIC] infs in valid_y_arr_np after fill: {np.isinf(valid_y_arr_np).sum()}")
            if valid_X_arr_np.shape[0] == 0 or valid_y_arr_np.shape[0] == 0:
                print(f"[WARNING] No valid rows remain for GradientBoosting fit in {tag} after force-fill. Skipping model fit.")
                out["gb_val"] = np.nan
                return out
            # Deep diagnostic: print dtype and any remaining NaN indices/values
            print(f"[DEEP DIAG] valid_X_arr_np dtype: {valid_X_arr_np.dtype}")
            print(f"[DEEP DIAG] valid_y_arr_np dtype: {valid_y_arr_np.dtype}")
            x_nan_idx = np.argwhere(np.isnan(valid_X_arr_np))
            y_nan_idx = np.argwhere(np.isnan(valid_y_arr_np))
            if x_nan_idx.size > 0:
                print(f"[DEEP DIAG] NaN found in valid_X_arr_np at indices: {x_nan_idx}")
                for idx in x_nan_idx:
                    print(f"[DEEP DIAG] valid_X_arr_np{tuple(idx)} = {valid_X_arr_np[tuple(idx)]}")
                out["gb_val"] = np.nan
                return out
            if y_nan_idx.size > 0:
                print(f"[DEEP DIAG] NaN found in valid_y_arr_np at indices: {y_nan_idx}")
                for idx in y_nan_idx:
                    print(f"[DEEP DIAG] valid_y_arr_np{tuple(idx)} = {valid_y_arr_np[tuple(idx)]}")
                out["gb_val"] = np.nan
                return out
            print(f"• {tag}: Fitting GradientBoosting on {len(valid_X_arr_np)} rows, {valid_X_arr_np.shape[1]} features.")
            gb.fit(valid_X_arr_np, valid_y_arr_np)
            # For prediction, fill any remaining NaNs with -1 and convert to float64
            pred_X = X_enc[valid_X.columns].fillna(-1).astype(np.float64)
            out["gb_val"] = gb.predict(pred_X)

    # ─ MissForest (optional)
    if HAS_MISSFOREST and len(out) < MAX_MISSFOREST_ROWS:
        print(f"• {tag}: MissForest…")
        mf = MissForest(random_state=RANDOM_STATE, max_iter=6)
        mf_arr = mf.fit_transform(out[[target_col] + feat_cols])
        out["mf_val"] = mf_arr[:, 0]
    else:
        out["mf_val"] = np.nan

    # ─ ensemble blend
    cand = ["heur_val", "iter_val", "gb_val", "mf_val"]
    out["blend_val"] = mae_blend(out, cand, target_col)

    # ─ final selection hierarchy
    out["final_val"] = (out["blend_val"]
                         .fillna(out["heur_val"])
                         .fillna(out["iter_val"])
                         .fillna(out["gb_val"])
                         .fillna(out["mf_val"]))

    # ─ flags
    if "final_val" in out.columns:
        out["final_val"] = out["final_val"].round().astype("Int64")
    else:
        out["final_val"] = pd.Series([pd.NA]*len(out), dtype="Int64")
    if target_col in out.columns:
        out["_imputed_flag_year"] = out[target_col].isna().astype(int)
    else:
        out["_imputed_flag_year"] = 1
    if target_col in out.columns and "heur_val" in out.columns and "heur_tag" in out.columns:
        out["_imputed_by_year"] = np.where(out[target_col].notna(), "reported",
                                            np.where(out["heur_val"].notna(), out["heur_tag"], "model"))
    else:
        out["_imputed_by_year"] = "model"
    if target_col in out.columns and "heur_val" in out.columns:
        out["_confidence_year"] = np.select(
            [out[target_col].notna(), out["heur_val"].notna()],
            ["high", "medium"],
            default="low")
    else:
        out["_confidence_year"] = "low"

    # Diagnostics before renaming
    print("[impute_year DIAG] Columns before rename:", out.columns.tolist())
    print("[impute_year DIAG] Head before rename:\n", out.head(2))
    import sys; sys.stdout.flush()
    # Explicit column naming for oil/gas and coal
    if tag == "oil/gas":
        out.rename(columns={"final_val": "discovery year_final"}, inplace=True)
        expected_cols = ["discovery year_final", "_imputed_flag_year", "_imputed_by_year", "_confidence_year"]
    elif tag == "coal":
        out.rename(columns={"final_val": "opening year_final"}, inplace=True)
        expected_cols = ["opening year_final", "_imputed_flag_year", "_imputed_by_year", "_confidence_year"]
    else:
        # fallback to previous behavior
        out.rename(columns={"final_val": f"{target_col.lower()}_final"}, inplace=True)
        expected_cols = [f"{target_col.lower()}_final", "_imputed_flag_year", "_imputed_by_year", "_confidence_year"]
    print("[impute_year DIAG] Columns after rename:", out.columns.tolist())
    print("[impute_year DIAG] Head after rename:\n", out.head(2))
    import sys; sys.stdout.flush()
    # Ensure all expected columns exist, even if empty
    for col in expected_cols:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA]*len(out))
    print("[impute_year DIAG] Final columns before return:", out.columns.tolist())
    print("[impute_year DIAG] Final head before return:\n", out.head(2))
    import sys; sys.stdout.flush()
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    db_path = args.db
    warnings.filterwarnings("ignore")
    print("Connecting to", db_path)
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"✖ Could not connect to database at {db_path}: {e}")
        sys.exit(1)

    # --- OIL / GAS ----------------------------------------------------------------
    try:
        og_fields = pd.read_sql("SELECT * FROM Oil_Gas_fields", conn)
    except Exception as e:
        print(f"✖ Could not load Oil_Gas_fields: {e}")
        return
    # og_fields should have correct dtypes from the DB; assert and print dtypes for diagnostics
    print("[DEBUG] Before og_fields dtypes")
    print("[DIAG] og_fields dtypes:")
    print(og_fields.dtypes)
    print("[DEBUG] After og_fields dtypes")
    import sys; sys.stdout.flush()

    feat_og = [c for c in [
        "FID Year", "Production start year", "Status year",
        "Onshore/Offshore", "Fuel description", "Unit type",
        "Production Type", "Basin", "Country/Area"
    ] if c in og_fields.columns]
    if not feat_og:
        print("Warning: No features found for oil/gas imputation.")

    print("[DEBUG UNIQUE] This is before impute_year")
    print("[MAIN DIAG] About to call impute_year for Oil/Gas fields")
    import sys; sys.stdout.flush()
    og_fields_imp = impute_year(og_fields, "Discovery year", feat_og, tag="oil/gas")
    print("[MAIN DIAG] Returned from impute_year")
    import sys; sys.stdout.flush()
    print("[DIAG] og_fields_imp columns:", og_fields_imp.columns.tolist())
    print("[DIAG] og_fields_imp head:\n", og_fields_imp.head(2))
    # Now select columns, but only if all present
    required_cols = ["Unit ID", "discovery year_final", "_imputed_flag_year", "_imputed_by_year", "_confidence_year"]
    missing_cols = [c for c in required_cols if c not in og_fields_imp.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns in og_fields_imp: {missing_cols}")
        print("[DEBUG] Exiting due to missing columns")
        import sys; sys.stdout.flush()
        sys.exit(1)
    # Rename columns for uniqueness before merge
    og_fields_imp = og_fields_imp.rename(columns={
        "discovery year_final": "discovery_year_final_og_fields",
        "_imputed_flag_year": "_imputed_flag_year_og_fields",
        "_imputed_by_year": "_imputed_by_year_og_fields",
        "_confidence_year": "_confidence_year_og_fields"
    })
    try:
        og_res = pd.read_sql("SELECT * FROM Oil_Gas_Production_Reserves", conn)
    except Exception as e:
        print(f"✖ Could not load Oil_Gas_Production_Reserves: {e}")
        return
    # Drop from og_fields_imp any columns (except 'Unit ID') that already exist in og_res
    intersection = set(og_fields_imp.columns) & set(og_res.columns)
    intersection.discard('Unit ID')
    print('[DIAG] Columns in both og_res and og_fields_imp before merge:', intersection)
    og_fields_imp = og_fields_imp[[c for c in og_fields_imp.columns if c == 'Unit ID' or c not in og_res.columns]]
    og_res = og_res.merge(og_fields_imp, on="Unit ID", how="left")

    # Coerce discovery year columns to integer
    print("[DIAG] og_res columns:", og_res.columns.tolist())
    # Check for duplicate columns before writing to SQL
    dup_cols = og_res.columns[og_res.columns.duplicated()].tolist()
    if dup_cols:
        print(f"[ERROR] Duplicate columns before SQL write: {dup_cols}")
        # Drop all but the first occurrence of each duplicate
        og_res = og_res.loc[:, ~og_res.columns.duplicated()]
        print("[DIAG] Dropped duplicate columns. Columns now:", og_res.columns.tolist())
    # Print all columns that start with 'discovery year_final'
    discovery_cols = [c for c in og_res.columns if c.startswith('discovery year_final')]
    print('[DIAG] Columns matching discovery year_final:', discovery_cols)
    for col in ["discovery year_final", "discovery_year_final_og_fields"]:
        if col in og_res.columns:
            print(f"[DIAG] Type of og_res[{col}]:", type(og_res[col]))
            if isinstance(og_res[col], pd.Series):
                og_res[col] = pd.to_numeric(og_res[col], errors='coerce')
                print(f"[DIAG] {col} dtype after numeric coercion: {og_res[col].dtype}")
                # Show problematic values
                nonint_mask = (~og_res[col].isna()) & (og_res[col] % 1 != 0)
                if nonint_mask.any():
                    print(f"[DIAG] {col} has non-integer values:")
                    print(og_res.loc[nonint_mask, col].head(10))
                else:
                    og_res[col] = og_res[col].astype('Int64')
                    print(f"[DIAG] {col} successfully coerced to Int64. Nulls: {og_res[col].isna().sum()}")
            else:
                print(f"[ERROR] og_res[{col}] is not a Series! Type: {type(og_res[col])}")
    # --- Diagnostics before SQL write ---
    print(f"[DIAG] About to write Oil_Gas_Production_Reserves: shape={og_res.shape}, columns={og_res.columns.tolist()}")
    try:
        og_res.to_sql("Oil_Gas_Production_Reserves", conn, if_exists="replace", index=False)
        print(f"[SUCCESS] Oil_Gas_Production_Reserves written to DB: {og_res.shape[0]} rows, {og_res.shape[1]} columns.")
    except Exception as e:
        print(f"[ERROR] Failed to write Oil_Gas_Production_Reserves to DB: {e}")
        import sys; sys.stdout.flush()
        raise

    # Check required columns for conversion
    if not ("Quantity (converted)" in og_res.columns and "Units (converted)" in og_res.columns):
        print("Warning: Required columns for oil/gas conversion missing in Oil_Gas_Production_Reserves.")

    # Normalize fuel types to supported values
    def normalize_fuel_type(val):
        if not isinstance(val, str):
            return None
        v = val.strip().lower()
        if v in ["oil", "crude oil", "oil field", "oil only", "oilfield", "oil and condensate", "condensate"]:
            return "oil"
        if v in ["gas", "natural gas", "gas field", "gas only", "gasfield"]:
            return "gas"
        if v in ["oil and gas", "oil & gas", "oil/gas", "oil+gas", "oil and gas field"]:
            return "oil and gas"
        # fallback: try to map anything with both words to oil and gas
        if "oil" in v and "gas" in v:
            return "oil and gas"
        if "oil" in v:
            return "oil"
        if "gas" in v:
            return "gas"
        return None
    og_res["Fuel description (normalized)"] = og_res["Fuel description"].apply(normalize_fuel_type)

    # Only keep reserve records, and skip units ending with '/y' (production units)
    reserve_mask = (
        og_res["Record type"].str.lower().str.contains("reserve")
        if "Record type" in og_res.columns else pd.Series([True]*len(og_res))
    )
    not_production_unit_mask = ~og_res["Units (converted)"].astype(str).str.endswith("/y")
    mask = reserve_mask & not_production_unit_mask

    # --- MAP Units (converted) to allowed units ---
    def map_units(unit):
        u = str(unit).strip().lower()
        if u in ["million bbl", "million bbl/y"]:
            return "million bbl"
        if u in ["million m³", "million m³/y", "million m3", "million m3/y"]:
            return "million m3"
        if u in ["million boe", "million boe/y"]:
            return "million boe"
        return u
    og_res["Units (converted) mapped"] = og_res["Units (converted)"].apply(map_units)

    # DIAGNOSTICS: Check normalization and mask
    print("[DIAG] Fuel description value_counts:")
    print(og_res["Fuel description"].value_counts(dropna=False))
    print("[DIAG] Fuel description (normalized) value_counts:")
    print(og_res["Fuel description (normalized)"].value_counts(dropna=False))
    print(f"[DIAG] Number of rows passing mask (should be reserves): {mask.sum()} / {len(mask)}")
    print("[DIAG] Sample of mask-passing rows:")
    print(og_res.loc[mask, ["Unit ID", "Fuel description", "Fuel description (normalized)", "Quantity (converted)", "Units (converted)"]].head(10))
    print("[DIAG] Unique Units (converted) for mask-passing rows:")
    print(og_res.loc[mask, "Units (converted)"].unique())
    print("[DIAG] Unique Units (converted) mapped for mask-passing rows:")
    print(og_res.loc[mask, "Units (converted) mapped"].unique())
    # Try a sample call to _oilgas_to_ej
    sample_row = og_res.loc[mask].iloc[0]
    print("[DIAG] Sample call to _oilgas_to_ej:")
    print("quantity:", sample_row["Quantity (converted)"])
    print("unit:", sample_row["Units (converted) mapped"])
    print("fuel_type:", sample_row["Fuel description (normalized)"])
    try:
        result = _oilgas_to_ej(sample_row["Quantity (converted)"], sample_row["Units (converted) mapped"], sample_row["Fuel description (normalized)"])
        print("[DIAG] _oilgas_to_ej result:", result)
    except Exception as e:
        print("[DIAG] _oilgas_to_ej raised exception:", e)

    print(f"Converting Quantity (converted) → EJ for {mask.sum()} reserve records...")
    # Only apply conversion to rows where mask is True to avoid ValueError for unsupported units
    og_res["Quantity_initial_EJ"] = np.nan
    def safe_oilgas_to_ej(r):
        fuel_type = r.get("Fuel description (normalized)")
        if fuel_type is None:
            return np.nan
        return _oilgas_to_ej(r.get("Quantity (converted)"), r.get("Units (converted) mapped"), fuel_type)
    # DIAGNOSTICS: Print sample rows for conversion failure
    print("[DIAG] Sample for Quantity_initial_EJ assignment:")
    print(og_res.loc[mask, ["Quantity (converted)", "Units (converted) mapped", "Fuel description (normalized)"]].head(10))
    og_res.loc[mask, "Quantity_initial_EJ"] = og_res.loc[mask].apply(safe_oilgas_to_ej, axis=1)
    # Ensure numeric dtype for aggregation
    og_res["Quantity_initial_EJ"] = pd.to_numeric(og_res["Quantity_initial_EJ"], errors='coerce')
    print("[DIAG] Quantity_initial_EJ value counts after assignment:")
    print(og_res["Quantity_initial_EJ"].value_counts(dropna=False).head(10))

    og_res["_imputed_flag_qty"] = og_res["Quantity_initial_EJ"].isna().astype(int)
    og_res["_imputed_by_qty"] = np.where(og_res["_imputed_flag_qty"] == 0, "reported", "proxy_equal")
    og_res["_confidence_qty"] = np.where(og_res["_imputed_flag_qty"] == 0, "high", "low")

    na_mask = og_res["Quantity_initial_EJ"].isna()
    if na_mask.any():
        mean_by_country = (og_res.loc[~na_mask]
                                .groupby("Country/Area")["Quantity_initial_EJ"].mean())
        og_res.loc[na_mask, "Quantity_initial_EJ"] = og_res.loc[na_mask, "Country/Area"].map(mean_by_country)

    print("Oil/Gas rows imputed:", na_mask.sum())
    print("[DIAG] About to write Oil_Gas_Production_Reserves to SQL...")
    # Drop duplicate columns before writing to SQL
    og_res = og_res.loc[:,~og_res.columns.duplicated()]
    # Dtypes should match DB schema; print for diagnostics
    print("[DIAG] og_res dtypes before SQL write:")
    print(og_res.dtypes)
    og_res.to_sql("Oil_Gas_Production_Reserves", conn, if_exists="replace", index=False)
    print("[DIAG] Finished writing Oil_Gas_Production_Reserves to SQL.")

    # --- COAL --------------------------------------------------------------------
    print("[COAL DIAG] Entering coal imputation block...")
    coal = None
    try:
        coal = pd.read_sql("SELECT * FROM Coal_open_mines", conn)
        print("[DIAG] coal dtypes:")
        print(coal.dtypes)
        # --- CLEAN Opening Year ---
        import re
        def extract_year(val):
            if pd.isna(val):
                return np.nan
            match = re.search(r"(\d{4})", str(val))
            if match:
                return int(match.group(1))
            return np.nan
        coal["Opening Year_clean"] = coal["Opening Year"].apply(extract_year)
        n_non_numeric = coal["Opening Year_clean"].isna().sum()
        print(f"[CLEAN DIAG] Non-numeric Opening Year entries after extraction: {n_non_numeric}")
        if n_non_numeric > 0:
            print("[CLEAN DIAG] Example problematic Opening Year values:")
            print(coal.loc[coal["Opening Year_clean"].isna(), "Opening Year"].head())
        coal["Opening Year"] = coal["Opening Year_clean"]
        del coal["Opening Year_clean"]
        # Opening Year and other columns should be correct type; add assert
        assert np.issubdtype(coal['Opening Year'].dtype, np.number), "Opening Year should be numeric"
        feat_coal = [c for c in [
            "Closing Year", "Mine type", "Status", "Mining method",
            "Country / Area", "Capacity (Mtpa)", "Production (Mtpa)",
        ] if c in coal.columns]
        if not feat_coal:
            print("Warning: No features found for coal imputation.")
        # Encode categoricals for features
        def encode_categoricals(df):
            cat_cols = df.select_dtypes(include="object").columns
            if len(cat_cols) == 0:
                return df
            for col in cat_cols:
                df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
            return df
        coal_encoded = coal.copy()
        coal_encoded = encode_categoricals(coal_encoded)
        # Impute Opening Year using robust target assignment
        y = coal_encoded["Opening Year"]
        X = coal_encoded[feat_coal]
        print(f"[COAL DIAG] Imputing Opening Year — missing: {y.isna().sum()}, total: {len(y)}")
        coal_imp = impute_year(coal_encoded, "Opening Year", feat_coal, tag="coal")
        # Merge imputed columns back using 'GEM Mine ID' to avoid length mismatch
        coal["GEM Mine ID"] = coal["GEM Mine ID"].astype(str)
        coal_imp["GEM Mine ID"] = coal_imp["GEM Mine ID"].astype(str)
        coal_merged = coal.merge(
            coal_imp[["GEM Mine ID", "opening year_final", "_imputed_flag_year", "_imputed_by_year", "_confidence_year"]],
            on="GEM Mine ID", how="left", suffixes=("", "_imputed")
        )
        print(f"[COAL DIAG] After merge: coal_merged shape: {coal_merged.shape}, columns: {list(coal_merged.columns)}")
        print("[COAL DIAG] opening year_final head:\n", coal_merged["opening year_final"].head())
        print("[COAL DIAG] opening year_final value counts:\n", coal_merged["opening year_final"].value_counts(dropna=False))
        print("[COAL DIAG] _imputed_flag_year value counts:\n", coal_merged["_imputed_flag_year"].value_counts(dropna=False))
        coal = coal_merged
    except Exception as e:
        print(f"[COAL DIAG ERROR] Exception in coal imputation block: {e}")
        import traceback; traceback.print_exc()

    if coal is not None:
        feat_coal = [c for c in [
            "Closing Year", "Mine type", "Status", "Mining method",
            "Country / Area", "Capacity (Mtpa)", "Production (Mtpa)",
        ] if c in coal.columns]
        if not feat_coal:
            print("Warning: No features found for coal imputation.")

        # Normalize coal types to supported values
        def normalize_coal_type(val):
            if not isinstance(val, str):
                return None
            v = val.strip().lower()
            if v in ["bituminous", "bituminous coal", "bituminous and subbituminous"]:
                return "Bituminous"
            if v in ["subbituminous", "subbituminous coal", "subbituminous / lignite", "bituminous and subbituminous"]:
                return "Subbituminous"
            if v in ["lignite", "lignite coal", "subbituminous / lignite"]:
                return "Lignite"
            # fallback: if the word is present
            if "bituminous" in v:
                return "Bituminous"
            if "subbituminous" in v:
                return "Subbituminous"
            if "lignite" in v:
                return "Lignite"
            return None
        coal_type_col = None
        for col in ["Coal Type", "coal_type", "Type", "Coal_Type"]:
            if col in coal.columns:
                coal_type_col = col
                break
        if not coal_type_col:
            print("Warning: No coal type column found for conversion. Conversion will fail.")
        else:
            coal["coal_type_normalized"] = coal[coal_type_col].apply(normalize_coal_type)
            return "Subbituminous"
        if v in ["lignite", "lignite coal", "subbituminous / lignite"]:
            return "Lignite"
        # fallback: if the word is present
        if "bituminous" in v:
            return "Bituminous"
        if "subbituminous" in v:
            return "Subbituminous"
        if "lignite" in v:
            return "Lignite"
        return None
    coal["coal_type_normalized"] = coal[coal_type_col].apply(normalize_coal_type) if coal_type_col else None

    print("Converting Total Reserves → EJ…")
    def safe_coal_to_ej(r):
        coal_type = r.get("coal_type_normalized")
        if coal_type is None:
            return np.nan
        return _coal_to_ej(r.get("Total Reserves (Proven and Probable, Mt)"), coal_type)
    coal["reserves_initial_EJ"] = coal.apply(safe_coal_to_ej, axis=1)
    # Ensure numeric dtype for aggregation
    coal["reserves_initial_EJ"] = pd.to_numeric(coal["reserves_initial_EJ"], errors='coerce')
    coal["_imputed_flag_qty"] = coal["reserves_initial_EJ"].isna().astype(int)
    coal["_imputed_by_qty"] = np.where(coal["_imputed_flag_qty"] == 0, "reported", "proxy_equal")
    coal["_confidence_qty"] = np.where(coal["_imputed_flag_qty"] == 0, "high", "low")

    na_mask_c = coal["reserves_initial_EJ"].isna()
    if na_mask_c.any():
        mean_country = coal.loc[~na_mask_c].groupby("Country / Area")["reserves_initial_EJ"].mean()
        coal.loc[na_mask_c, "reserves_initial_EJ"] = coal.loc[na_mask_c, "Country / Area"].map(mean_country)

    print("Coal rows imputed:", na_mask_c.sum())
    print("Replacing Coal_open_mines table…")
    coal.to_sql("Coal_open_mines", conn, if_exists="replace", index=False)

    conn.close()
    print("✓ Imputation complete. Updated tables written to Energy.db")


if __name__ == "__main__":
    main()
