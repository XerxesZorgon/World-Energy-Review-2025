#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
richards_discovery_fit.py

Fit 1–3 Richards peaks to discovery data and pick the best model by AICc.
You can choose whether the *fit* is done on cumulative or rate data;
the script will *always* produce both cumulative and rate plots for review/presentation.

Inputs:
  - CSV with columns: year, value (rename via --year-col / --value-col)
  - --input {rate|cumulative} tells us how to interpret your CSV values
  - --fit-on {rate|cumulative|auto} chooses which space to fit in (default: cumulative)
        auto: if input=rate → fits on rate; if input=cumulative → fits on cumulative

Outputs:
  - Console summary (best model, parameters, metrics for BOTH spaces)
  - PNG plots: cumulative (for presentation) and rate (diagnostic)
  - CSV of per-peak parameters
  - JSON report

Usage example:
  python richards_discovery_fit.py data.csv --year-col year --value-col discovery \
        --input rate --fit-on cumulative --title "Global oil discoveries"
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
from scipy.optimize import least_squares
from scipy.signal import find_peaks
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path

# ----------------- Constants -----------------

# Bounds for parameter optimization
TIME_BUFFER = 20.0
MIN_NU = 0.05
MAX_K = 2.0
MAX_NU = 10.0
MIN_QINF = 1e-9
MIN_K = 1e-6
NU_TOLERANCE = 1e-12
EXP_CLIP_MIN = 1e-300
EXP_CLIP_MAX = 1e300

# ----------------- Richards model -----------------

def richards_cumulative(t, Qinf, k, t0, nu):
    """
    Richards cumulative function: Q(t) = Qinf / (1 + nu*exp(-k*(t - t0)))**(1/nu)
    
    Args:
        t: Time array
        Qinf: Ultimate cumulative value
        k: Growth rate parameter
        t0: Time of inflection point
        nu: Shape parameter
    
    Returns:
        Cumulative values at time t
    """
    if abs(nu) < NU_TOLERANCE:
        raise ValueError(f"nu parameter too close to zero: {nu}")
    
    x = np.clip(np.exp(-k * (t - t0)), EXP_CLIP_MIN, EXP_CLIP_MAX)
    return Qinf / np.power(1.0 + nu * x, 1.0/nu)

def richards_rate(t, Qinf, k, t0, nu):
    """
    Richards rate function: R(t) = k*Qinf*exp(-k*(t - t0)) / (1 + nu*exp(-k*(t - t0)))**(1 + 1/nu)
    
    Args:
        t: Time array
        Qinf: Ultimate cumulative value
        k: Growth rate parameter
        t0: Time of inflection point
        nu: Shape parameter
    
    Returns:
        Rate values at time t
    """
    if abs(nu) < NU_TOLERANCE:
        raise ValueError(f"nu parameter too close to zero: {nu}")
    
    x = np.clip(np.exp(-k * (t - t0)), EXP_CLIP_MIN, EXP_CLIP_MAX)
    denom = np.power(1.0 + nu * x, 1.0 + 1.0/nu)
    return (k * Qinf * x) / denom

def mixture_cumulative(t, params, n_peaks):
    y = np.zeros_like(t, dtype=float)
    for i in range(n_peaks):
        Qinf, k, t0, nu = params[4*i:4*i+4]
        y += richards_cumulative(t, Qinf, k, t0, nu)
    return y

def mixture_rate(t, params, n_peaks):
    y = np.zeros_like(t, dtype=float)
    for i in range(n_peaks):
        Qinf, k, t0, nu = params[4*i:4*i+4]
        y += richards_rate(t, Qinf, k, t0, nu)
    return y

# ----------------- Metrics -----------------

def r2_score(y, yhat):
    """Calculate R-squared coefficient of determination."""
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def rmse(y, yhat):
    """Calculate Root Mean Square Error."""
    return float(np.sqrt(np.mean((y - yhat)**2)))

def aic(n, rss, kparams):
    """Calculate Akaike Information Criterion."""
    return n*np.log(rss/n) + 2*kparams

def aicc(n, rss, kparams):
    """Calculate corrected Akaike Information Criterion."""
    a = aic(n, rss, kparams)
    denom = (n - kparams - 1)
    return np.inf if denom <= 0 else a + (2*kparams*(kparams+1))/denom

def bic(n, rss, kparams):
    """Calculate Bayesian Information Criterion."""
    return n*np.log(rss/n) + kparams*np.log(n)

# ----------------- Inits & bounds -----------------

def total_area_trapz(t, y):
    """Calculate total area under curve using trapezoidal rule."""
    try:
        return np.trapezoid(y, t)
    except AttributeError:
        return np.trapz(y, t)

def guess_peaks_from_data(t, y, n_peaks):
    """
    Guess peak locations from data for initial parameter estimation.
    
    Args:
        t: Time array
        y: Data values
        n_peaks: Number of peaks to find
    
    Returns:
        List of indices for peak locations
    """
    if len(y) < 3: 
        return [max(0, len(y)//2)]
    
    if n_peaks <= 0:
        return [len(y)//2]
    
    # Try to find peaks in the data
    try:
        distance = max(1, len(y)//(n_peaks+1))
        idxs, _ = find_peaks(y, distance=distance)
    except Exception:
        idxs = []
    
    if len(idxs) == 0:
        # No peaks found, distribute evenly
        if n_peaks == 1:
            return [len(t)//2]
        else:
            centers = np.linspace(0, len(t)-1, n_peaks+2, dtype=int)[1:-1]
            return [max(0, min(len(t)-1, c)) for c in centers]
    
    if len(idxs) >= n_peaks:
        # More peaks than needed, select the highest ones
        try:
            order = np.argsort(y[idxs])[::-1]
            picks = idxs[order[:n_peaks]]
            return sorted(picks)
        except IndexError:
            # Fallback if indexing fails
            return sorted(idxs[:n_peaks])
    else:
        # Fewer peaks than needed, use what we have and add more
        result = list(idxs)
        needed = n_peaks - len(idxs)
        if needed > 0:
            # Add evenly spaced points
            remaining_space = np.setdiff1d(np.arange(len(t)), idxs)
            if len(remaining_space) >= needed:
                additional = np.linspace(0, len(remaining_space)-1, needed, dtype=int)
                result.extend(remaining_space[additional])
            else:
                # Just add some reasonable points
                for i in range(needed):
                    idx = (i + 1) * len(t) // (needed + 1)
                    result.append(min(idx, len(t)-1))
        
        return sorted(result)

def build_initial_params_for_rate(t, y_rate, n_peaks, custom_t0_values=None):
    """
    Build initial parameters for rate-based fitting.
    
    Args:
        t: Time array
        y_rate: Rate data
        n_peaks: Number of peaks
        custom_t0_values: Optional list of custom t0 values for each peak
    
    Returns:
        Array of initial parameters [Qinf, k, t0, nu] for each peak
    """
    t = np.asarray(t, float)
    y_rate = np.asarray(y_rate, float)
    
    if len(t) != len(y_rate):
        raise ValueError(f"Time and rate arrays must have same length: {len(t)} vs {len(y_rate)}")
    
    if len(t) == 0:
        raise ValueError("Empty time array")
    
    if n_peaks <= 0:
        raise ValueError(f"Number of peaks must be positive: {n_peaks}")
    
    # Validate custom t0 values if provided
    if custom_t0_values is not None:
        if len(custom_t0_values) != n_peaks:
            raise ValueError(f"Number of custom t0 values ({len(custom_t0_values)}) must match number of peaks ({n_peaks})")
        t_min, t_max = float(np.min(t)), float(np.max(t))
        for i, t0_val in enumerate(custom_t0_values):
            if t0_val < t_min - 50 or t0_val > t_max + 50:  # Allow some buffer
                raise ValueError(f"Custom t0 value {i+1} ({t0_val}) is outside reasonable range [{t_min-50}, {t_max+50}]")
    
    span = max(5.0, t[-1] - t[0])
    area = max(1e-6, total_area_trapz(t, y_rate))
    Q_each = area / n_peaks
    k_default = 2.0/span
    
    # Get t0 values - either custom or estimated from peaks
    if custom_t0_values is not None:
        t0s = list(custom_t0_values)
    else:
        try:
            peak_idxs = guess_peaks_from_data(t, y_rate, n_peaks)
        except Exception as e:
            raise ValueError(f"Error finding peaks: {e}")
        
        if len(peak_idxs) != n_peaks:
            raise ValueError(f"Expected {n_peaks} peak indices, got {len(peak_idxs)}")
        
        # Validate peak indices
        for i, idx in enumerate(peak_idxs):
            if idx < 0 or idx >= len(t):
                raise ValueError(f"Peak index {i} out of range: {idx} (array length: {len(t)})")
        
        t0s = [t[i] for i in peak_idxs]
    
    params = []
    for i in range(n_peaks): 
        params += [Q_each, k_default, t0s[i], 1.0]
    
    return np.array(params, float)

def build_initial_params_for_cum(t, y_cum, n_peaks, custom_t0_values=None):
    # Use second finite diff as a proxy for curvature to guess centers
    t = np.asarray(t, float); y = np.asarray(y_cum, float)
    span = max(5.0, t[-1] - t[0])
    rate_proxy = np.gradient(y, t)
    return build_initial_params_for_rate(t, np.maximum(rate_proxy, 0.0), n_peaks, custom_t0_values)

def build_bounds(t, n_peaks, custom_t0_values=None):
    """Build parameter bounds for optimization."""
    tmin, tmax = float(np.min(t)), float(np.max(t))
    lb, ub = [], []
    for i in range(n_peaks):
        lb += [MIN_QINF, MIN_K]
        ub += [np.inf, MAX_K]
        
        # Handle t0 bounds - fix if custom values provided
        if custom_t0_values is not None and i < len(custom_t0_values):
            # Fix t0 to the custom value by setting tight bounds
            t0_val = float(custom_t0_values[i])
            lb += [t0_val - 1e-6, MIN_NU]  # Very tight bound around custom value
            ub += [t0_val + 1e-6, MAX_NU]
        else:
            # Normal t0 bounds
            lb += [tmin - TIME_BUFFER, MIN_NU]
            ub += [tmax + TIME_BUFFER, MAX_NU]
    
    return np.array(lb, float), np.array(ub, float)

# ----------------- Fitting core -----------------

def fit_in_space(t, y_target, n_peaks, space="rate", restarts=12, seed=42, custom_t0_values=None):
    """
    Fit parameters so that mixture_{space}(t, params) ~= y_target.
    space ∈ {"rate","cumulative"}.
    """
    rng = np.random.default_rng(seed)
    lb, ub = build_bounds(t, n_peaks, custom_t0_values)
    if space == "rate":
        p0 = build_initial_params_for_rate(t, y_target, n_peaks, custom_t0_values)
        model = lambda tt, pp: mixture_rate(tt, pp, n_peaks)
    else:
        p0 = build_initial_params_for_cum(t, y_target, n_peaks, custom_t0_values)
        model = lambda tt, pp: mixture_cumulative(tt, pp, n_peaks)

    def residuals(p): return model(t, p) - y_target

    def jitter(p, scale=0.35):
        pj = np.copy(p)
        span = (t[-1]-t[0]) or 1.0
        for i in range(n_peaks):
            Q, k, t0, nu = pj[4*i:4*i+4]
            pj[4*i+0] = Q * np.exp(rng.normal(0, scale))
            pj[4*i+1] = k * np.exp(rng.normal(0, scale))
            
            # Only jitter t0 if not using custom values
            if custom_t0_values is None or i >= len(custom_t0_values):
                pj[4*i+2] = t0 + rng.normal(0, scale*0.2*span)
            # else: keep t0 fixed at custom value
            
            pj[4*i+3] = max(0.06, nu + rng.normal(0, scale))
        return pj

    best = None
    inits = [p0] + [jitter(p0) for _ in range(restarts)]
    for pinit in inits:
        pinit = np.minimum(np.maximum(pinit, lb+1e-12), ub-1e-12)
        res = least_squares(residuals, pinit, bounds=(lb, ub),
                            loss="soft_l1", f_scale=1.0, max_nfev=6000)
        rss = float(np.sum(res.fun**2))
        if (best is None) or (rss < best["rss"]):
            best = dict(params=res.x, rss=rss, success=res.success,
                        message=str(res.message), nfev=int(res.nfev))

    # Metrics in fit-space
    yhat_fit = model(t, best["params"])
    n = len(t); kparams = len(best["params"])
    metrics = {
        "RSS": best["rss"],
        "R2": float(r2_score(y_target, yhat_fit)),
        "RMSE": rmse(y_target, yhat_fit),
        "AIC": float(aic(n, best["rss"], kparams)),
        "AICc": float(aicc(n, best["rss"], kparams)),
        "BIC": float(bic(n, best["rss"], kparams)),
        "n": int(n),
        "kparams": int(kparams),
        "success": bool(best["success"]),
        "message": best["message"],
        "nfev": best["nfev"],
    }
    return best["params"], metrics

@dataclass
class FitResult:
    n_peaks: int
    params: List[float]
    fit_space: str
    metrics_fit_space: Dict[str, float]
    metrics_rate: Dict[str, float]
    metrics_cum: Dict[str, float]
    yhat_rate: List[float]
    yhat_cum: List[float]

def evaluate_both_spaces(t, params, n_peaks, y_rate, y_cum, fit_space):
    yhat_r = mixture_rate(t, params, n_peaks)
    yhat_c = mixture_cumulative(t, params, n_peaks)
    # Compute metrics in BOTH spaces for fair comparison and diagnostics
    n_r, n_c = len(y_rate), len(y_cum)
    kparams = len(params)
    rss_r = float(np.sum((y_rate - yhat_r)**2))
    rss_c = float(np.sum((y_cum - yhat_c)**2))
    metrics_rate = {
        "RSS": rss_r, "R2": float(r2_score(y_rate, yhat_r)), "RMSE": rmse(y_rate, yhat_r),
        "AIC": float(aic(n_r, rss_r, kparams)), "AICc": float(aicc(n_r, rss_r, kparams)),
        "BIC": float(bic(n_r, rss_r, kparams)), "n": int(n_r), "kparams": int(kparams)
    }
    metrics_cum = {
        "RSS": rss_c, "R2": float(r2_score(y_cum, yhat_c)), "RMSE": rmse(y_cum, yhat_c),
        "AIC": float(aic(n_c, rss_c, kparams)), "AICc": float(aicc(n_c, rss_c, kparams)),
        "BIC": float(bic(n_c, rss_c, kparams)), "n": int(n_c), "kparams": int(kparams)
    }
    return yhat_r, yhat_c, metrics_rate, metrics_cum

def fit_best_model(t, y_rate, y_cum, fit_on="cumulative", max_peaks=3, restarts=12, seed=42, custom_t0_values=None):
    """
    Try 1..max_peaks. For each peak count:
      - Fit in the requested space (rate or cumulative)
      - Evaluate metrics in BOTH spaces
    Model selection: choose smallest AICc in the *fit space*.
    
    Args:
        t: Time array
        y_rate: Rate data
        y_cum: Cumulative data
        fit_on: Space to fit in ('rate' or 'cumulative')
        max_peaks: Maximum number of peaks to try
        restarts: Number of random restarts
        seed: Random seed
        custom_t0_values: Optional list of custom t0 values for each peak
    
    Returns:
        Tuple of (best_result, all_results)
    """
    # Input validation
    if len(t) != len(y_rate) or len(t) != len(y_cum):
        raise ValueError("Input arrays must have same length")
    if max_peaks < 1 or max_peaks > 5:
        raise ValueError("max_peaks must be between 1 and 5")
    if len(t) < 6:
        raise ValueError("Need at least 6 data points for fitting")
    
    results: List[FitResult] = []
    fit_space = "rate" if fit_on == "rate" else "cumulative"

    for n in range(1, max_peaks+1):
        target = y_rate if fit_space == "rate" else y_cum
        
        # Use custom t0 values only if the number matches current peak count
        t0_vals = custom_t0_values[:n] if custom_t0_values and len(custom_t0_values) >= n else None
        
        params, metrics_fit = fit_in_space(t, target, n_peaks=n, space=fit_space,
                                           restarts=restarts, seed=seed, custom_t0_values=t0_vals)
        yhat_r, yhat_c, metr_r, metr_c = evaluate_both_spaces(
            t, params, n, y_rate, y_cum, fit_space
        )
        results.append(FitResult(
            n_peaks=n,
            params=list(params),
            fit_space=fit_space,
            metrics_fit_space=metrics_fit,
            metrics_rate=metr_r,
            metrics_cum=metr_c,
            yhat_rate=list(yhat_r),
            yhat_cum=list(yhat_c),
        ))

    # pick best by AICc in the chosen fit space
    best = min(results, key=lambda r: r.metrics_fit_space["AICc"])
    return best, results

# ----------------- Plotting -----------------

def plot_cumulative(t, y_cum, fit: FitResult, title, out_path):
    t = np.asarray(t, float)
    order = np.argsort(t)
    yhat = np.array(fit.yhat_cum)[order]
    y = np.asarray(y_cum, float)[order]
    plt.figure(figsize=(10, 6))
    plt.plot(t[order], y, marker='o', linestyle='', alpha=0.7, label="Cumulative (data)")
    plt.plot(t[order], yhat, linewidth=2.5, label=f"Best cumulative fit (n={fit.n_peaks})")
    # components
    p = np.array(fit.params, float)
    for i in range(fit.n_peaks):
        comp = richards_cumulative(t, *p[4*i:4*i+4])[order]
        plt.plot(t[order], comp, linestyle='--', alpha=0.9, label=f"Peak {i+1}")
    m = fit.metrics_cum
    txt = f"Fit on: {fit.fit_space}\nR²={m['R2']:.4f}  RMSE={m['RMSE']:.4g}  AICc={m['AICc']:.2f}"
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes, va='top',
                   bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))
    plt.title(title + " — Cumulative")
    plt.xlabel("Year"); plt.ylabel("Cumulative discoveries")
    plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

def plot_rate(t, y_rate, fit: FitResult, title, out_path):
    t = np.asarray(t, float)
    order = np.argsort(t)
    yhat = np.array(fit.yhat_rate)[order]
    y = np.asarray(y_rate, float)[order]
    plt.figure(figsize=(10, 6))
    plt.plot(t[order], y, marker='o', linestyle='', alpha=0.7, label="Rate (data)")
    plt.plot(t[order], yhat, linewidth=2.2, label=f"Best rate from fit (n={fit.n_peaks})")
    # components
    p = np.array(fit.params, float)
    for i in range(fit.n_peaks):
        comp = richards_rate(t, *p[4*i:4*i+4])[order]
        plt.plot(t[order], comp, linestyle='--', alpha=0.9, label=f"Peak {i+1}")
    m = fit.metrics_rate
    txt = f"Fit on: {fit.fit_space}\nR²={m['R2']:.4f}  RMSE={m['RMSE']:.4g}  AICc={m['AICc']:.2f}"
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes, va='top',
                   bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))
    plt.title(title + " — Rate")
    plt.xlabel("Year"); plt.ylabel("Annual discoveries")
    plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

# ----------------- I/O -----------------

def load_series(csv_path, year_col, value_col) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load time series data from CSV file with proper error handling.
    
    Args:
        csv_path: Path to CSV file
        year_col: Name of year/time column
        value_col: Name of value column
    
    Returns:
        Tuple of (time_array, value_array)
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_path}: {e}")
    
    if year_col not in df.columns:
        raise ValueError(f"Year column '{year_col}' not found. Available columns: {list(df.columns)}")
    
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found. Available columns: {list(df.columns)}")
    
    # Select and clean data
    df_clean = df[[year_col, value_col]].copy()
    
    # Remove rows where either column is NaN or empty
    df_clean = df_clean.dropna()
    
    # Remove rows with non-numeric values
    try:
        df_clean[year_col] = pd.to_numeric(df_clean[year_col], errors='coerce')
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
        df_clean = df_clean.dropna()
    except Exception as e:
        raise ValueError(f"Error converting columns to numeric: {e}")
    
    if len(df_clean) == 0:
        raise ValueError("No valid numeric data rows after cleaning")
    
    # Sort by year
    df_clean = df_clean.sort_values(year_col)
    
    t = df_clean[year_col].to_numpy(float)
    y = df_clean[value_col].to_numpy(float)
    
    # Additional validation
    if len(t) < 6:
        raise ValueError(f"Need at least 6 data points for fitting, got {len(t)}")
    
    if np.any(y < 0):
        raise ValueError("Negative values found in data. Richards model requires non-negative values.")
    
    if np.all(y == 0):
        raise ValueError("All values are zero. Cannot fit Richards model.")
    
    return t, y

def ensure_rate_and_cum(t, y, interpretation: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (y_rate, y_cum) from input y and interpretation ('rate' or 'cumulative').
    Negative tiny rates (from gradient) are clamped to zero.
    """
    interpretation = interpretation.lower()
    if interpretation == "rate":
        y_rate = np.asarray(y, float)
        # Use scipy's cumtrapz for efficient cumulative integration
        y_cum = np.concatenate([[0], cumtrapz(y_rate, t)])
    elif interpretation == "cumulative":
        y_cum = np.asarray(y, float)
        y_rate = np.gradient(y_cum, t)
        y_rate = np.where(y_rate < 0, 0.0, y_rate)
    else:
        raise ValueError("--input must be 'rate' or 'cumulative'")
    return y_rate, y_cum

# ----------------- Core Processing Function -----------------

def process_richards_fit(csv_path, year_col, value_col, input_type, fit_on, max_peaks, restarts, seed, title, output_dir, progress_callback=None, custom_t0_values=None):
    """
    Core processing function that can be called from both CLI and GUI.
    
    Args:
        csv_path: Path to CSV file
        year_col: Year column name
        value_col: Value column name
        input_type: 'rate' or 'cumulative'
        fit_on: 'rate', 'cumulative', or 'auto'
        max_peaks: Maximum number of peaks to test
        restarts: Number of random restarts
        seed: Random seed
        title: Plot title
        output_dir: Directory for output files
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with results and output file paths
    """
    try:
        if progress_callback:
            progress_callback("Loading data...")
        
        # Load data
        t, y_raw = load_series(csv_path, year_col, value_col)
        y_rate, y_cum = ensure_rate_and_cum(t, y_raw, input_type)
        
        # pick fit space
        if fit_on == "auto":
            fit_on = "rate" if input_type == "rate" else "cumulative"
        
        # Sanity filters
        mask = np.isfinite(t) & np.isfinite(y_rate) & np.isfinite(y_cum)
        t, y_rate, y_cum = t[mask], y_rate[mask], y_cum[mask]
        if len(t) < 6:
            raise ValueError("Need at least 6 valid points.")
        
        if progress_callback:
            progress_callback("Fitting models...")
        
        # Fit and model selection
        best, all_results = fit_best_model(
            t, y_rate, y_cum, fit_on, max_peaks=max(1, min(5, max_peaks)),
            restarts=restarts, seed=seed, custom_t0_values=custom_t0_values
        )
        
        if progress_callback:
            progress_callback("Saving results...")
        
        # Generate output file paths
        base_name = Path(csv_path).stem
        out_plot_cum = os.path.join(output_dir, f"{base_name}_richards_fit_cumulative.png")
        out_plot_rate = os.path.join(output_dir, f"{base_name}_richards_fit_rate.png")
        out_params = os.path.join(output_dir, f"{base_name}_richards_params.csv")
        out_json = os.path.join(output_dir, f"{base_name}_richards_model.json")
        
        # Save parameters table
        rows = []
        for i in range(best.n_peaks):
            Qinf, k, t0, nu = best.params[4*i:4*i+4]
            rows.append(dict(peak=i+1, Qinf=Qinf, k=k, t0=t0, nu=nu))
        pd.DataFrame(rows).to_csv(out_params, index=False)
        
        # Save JSON report
        report = {
            "fit_space": best.fit_space,
            "n_peaks": best.n_peaks,
            "params": best.params,
            "metrics_fit_space": best.metrics_fit_space,
            "metrics_rate": best.metrics_rate,
            "metrics_cumulative": best.metrics_cum,
            "columns_per_peak": ["Qinf","k","t0","nu"]
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        plot_cumulative(t, y_cum, best, title, out_plot_cum)
        plot_rate(t, y_rate, best, title, out_plot_rate)
        
        if progress_callback:
            progress_callback("Complete!")
        
        return {
            "best_result": best,
            "all_results": all_results,
            "output_files": {
                "cumulative_plot": out_plot_cum,
                "rate_plot": out_plot_rate,
                "parameters_csv": out_params,
                "model_json": out_json
            }
        }
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        raise

# ----------------- GUI Application -----------------

class FileSelectionDialog:
    def __init__(self, parent=None):
        self.selected_file = None
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Select CSV File - Richards Discovery Fit")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # Make dialog modal
        self.root.transient(parent)
        self.root.grab_set()
        
        self.setup_file_dialog()
        
        # Center the dialog
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
    def setup_file_dialog(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Richards Discovery Fit", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                                text="Select a CSV file containing discovery data for analysis.",
                                font=('Arial', 11))
        instructions.grid(row=1, column=0, pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="CSV File Selection", padding="15")
        file_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_path_var = tk.StringVar()
        
        # File path display
        path_frame = ttk.Frame(file_frame)
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        path_frame.columnconfigure(0, weight=1)
        
        ttk.Label(path_frame, text="Selected file:").grid(row=0, column=0, sticky=tk.W)
        self.path_entry = ttk.Entry(path_frame, textvariable=self.file_path_var, 
                                   state="readonly", width=60)
        self.path_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Browse button
        browse_btn = ttk.Button(file_frame, text="Browse for CSV File...", 
                               command=self.browse_file)
        browse_btn.grid(row=1, column=0, pady=(0, 15))
        
        # Preview frame
        preview_frame = ttk.LabelFrame(file_frame, text="File Preview", padding="10")
        preview_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        file_frame.rowconfigure(2, weight=1)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=8, width=70,
                                                     state="disabled")
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(10, 0))
        
        self.continue_btn = ttk.Button(button_frame, text="Continue to Options", 
                                      command=self.continue_to_options, state="disabled")
        self.continue_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file with discovery data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=self.root
        )
        if filename:
            self.file_path_var.set(filename)
            self.selected_file = filename
            self.continue_btn.config(state="normal")
            self.show_preview(filename)
            
    def show_preview(self, filename):
        try:
            # Read first few lines for preview
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(10)]
            
            preview_text = "\n".join(lines)
            if len(lines) == 10:
                preview_text += "\n... (file continues)"
                
            self.preview_text.config(state="normal")
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, preview_text)
            self.preview_text.config(state="disabled")
            
        except Exception as e:
            self.preview_text.config(state="normal")
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, f"Error reading file: {str(e)}")
            self.preview_text.config(state="disabled")
            
    def continue_to_options(self):
        if self.selected_file:
            self.root.destroy()
            
    def cancel(self):
        self.selected_file = None
        self.root.destroy()
        
    def show(self):
        self.root.wait_window()
        return self.selected_file

class ProcessingOptionsDialog:
    def __init__(self, csv_file_path, parent=None):
        self.csv_file_path = csv_file_path
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title(f"Processing Options - {os.path.basename(csv_file_path)}")
        self.root.geometry("700x800")
        self.root.resizable(True, True)
        
        # Variables
        self.year_col = tk.StringVar(value="Year")
        self.value_col = tk.StringVar()
        self.input_type = tk.StringVar(value="rate")
        self.fit_on = tk.StringVar(value="cumulative")
        self.max_peaks = tk.IntVar(value=3)
        self.restarts = tk.IntVar(value=14)
        self.seed = tk.IntVar(value=42)
        self.title = tk.StringVar(value="Richards Fit to Discovery Data")
        self.output_dir = tk.StringVar(value=os.path.dirname(csv_file_path))
        self.use_custom_t0 = tk.BooleanVar(value=False)
        self.custom_t0_values = tk.StringVar(value="1950, 1970, 1990")
        
        # Detect available columns
        self.available_columns = self.detect_columns()
        if self.available_columns:
            # Set default value column to first discovery column found
            discovery_cols = [col for col in self.available_columns if 'discover' in col.lower()]
            if discovery_cols:
                self.value_col.set(discovery_cols[0])
            else:
                self.value_col.set(self.available_columns[1] if len(self.available_columns) > 1 else "")
        
        self.setup_options_dialog()
        
        # Center the dialog
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
    def detect_columns(self):
        try:
            df = pd.read_csv(self.csv_file_path, nrows=0)  # Just read headers
            return list(df.columns)
        except Exception:
            return []
            
    def setup_options_dialog(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Title
        title_label = ttk.Label(main_frame, text="Richards Discovery Fit - Processing Options", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # File info
        file_info = ttk.Label(main_frame, text=f"File: {os.path.basename(self.csv_file_path)}",
                             font=('Arial', 10, 'italic'))
        file_info.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1
        
        # Column selection frame
        col_frame = ttk.LabelFrame(main_frame, text="Column Selection", padding="10")
        col_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        col_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Year column
        ttk.Label(col_frame, text="Year Column:").grid(row=0, column=0, sticky=tk.W, pady=5)
        year_combo = ttk.Combobox(col_frame, textvariable=self.year_col, 
                                 values=self.available_columns, width=25)
        year_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        
        # Value column
        ttk.Label(col_frame, text="Value Column:").grid(row=1, column=0, sticky=tk.W, pady=5)
        value_combo = ttk.Combobox(col_frame, textvariable=self.value_col, 
                                  values=self.available_columns, width=25)
        value_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        
        # Data type frame
        type_frame = ttk.LabelFrame(main_frame, text="Data Interpretation", padding="10")
        type_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        row += 1
        
        # Input type
        ttk.Label(type_frame, text="Input Data Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        input_frame = ttk.Frame(type_frame)
        input_frame.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        ttk.Radiobutton(input_frame, text="Rate (Annual)", variable=self.input_type, value="rate").pack(side=tk.LEFT)
        ttk.Radiobutton(input_frame, text="Cumulative", variable=self.input_type, value="cumulative").pack(side=tk.LEFT, padx=(15, 0))
        
        # Fit space
        ttk.Label(type_frame, text="Fit Space:").grid(row=1, column=0, sticky=tk.W, pady=5)
        fit_frame = ttk.Frame(type_frame)
        fit_frame.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        ttk.Radiobutton(fit_frame, text="Rate", variable=self.fit_on, value="rate").pack(side=tk.LEFT)
        ttk.Radiobutton(fit_frame, text="Cumulative", variable=self.fit_on, value="cumulative").pack(side=tk.LEFT, padx=(15, 0))
        ttk.Radiobutton(fit_frame, text="Auto", variable=self.fit_on, value="auto").pack(side=tk.LEFT, padx=(15, 0))
        
        # Advanced options frame
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding="10")
        advanced_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        advanced_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Max peaks
        ttk.Label(advanced_frame, text="Max Peaks:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(advanced_frame, from_=1, to=5, textvariable=self.max_peaks, width=10).grid(row=0, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        
        # Restarts
        ttk.Label(advanced_frame, text="Random Restarts:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(advanced_frame, from_=1, to=50, textvariable=self.restarts, width=10).grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        
        # Seed
        ttk.Label(advanced_frame, text="Random Seed:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(advanced_frame, textvariable=self.seed, width=10).grid(row=2, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        
        # Title
        ttk.Label(advanced_frame, text="Plot Title:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(advanced_frame, textvariable=self.title, width=40).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # Output directory
        ttk.Label(advanced_frame, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, pady=5)
        output_frame = ttk.Frame(advanced_frame)
        output_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.output_dir).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir).grid(row=0, column=1, padx=(5, 0))
        
        # Custom t0 values section
        ttk.Checkbutton(advanced_frame, text="Use Custom Inflection Years (t0)", 
                       variable=self.use_custom_t0, command=self.toggle_custom_t0).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        ttk.Label(advanced_frame, text="Inflection Years (comma-separated):").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.t0_entry = ttk.Entry(advanced_frame, textvariable=self.custom_t0_values, width=30, state="disabled")
        self.t0_entry.grid(row=6, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # Help text for t0 values
        help_text = ttk.Label(advanced_frame, text="Example: 1950, 1970, 1990 (fixed inflection years, one per peak)", 
                             font=('Arial', 8), foreground='gray')
        help_text.grid(row=7, column=1, sticky=tk.W, pady=(0, 5), padx=(5, 0))
        
        # Action buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        row += 1
        
        ttk.Button(button_frame, text="Run Richards Fit", command=self.run_fit).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Change File", command=self.change_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Exit", command=self.exit_app).pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to process")
        self.status_label.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Results text area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select output directory", parent=self.root)
        if directory:
            self.output_dir.set(directory)
            
    def toggle_custom_t0(self):
        """Enable/disable custom t0 values entry based on checkbox."""
        if self.use_custom_t0.get():
            self.t0_entry.config(state="normal")
        else:
            self.t0_entry.config(state="disabled")
            
    def change_file(self):
        """Allow user to select a different file."""
        file_dialog = FileSelectionDialog(self.root)
        new_file = file_dialog.show()
        if new_file:
            self.csv_file_path = new_file
            self.output_dir.set(os.path.dirname(new_file))
            self.available_columns = self.detect_columns()
            self.root.title(f"Processing Options - {os.path.basename(new_file)}")
            # Update file info label
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label) and "File:" in child.cget("text"):
                            child.config(text=f"File: {os.path.basename(new_file)}")
                            break
            
    def update_progress(self, message):
        """Update progress from worker thread."""
        self.root.after(0, lambda: self.status_label.config(text=message))
        
    def run_fit(self):
        """Run the Richards fit in a separate thread."""
        if not self.value_col.get():
            messagebox.showerror("Error", "Please select a value column.", parent=self.root)
            return
            
        # Start progress bar
        self.progress.start()
        self.results_text.delete(1.0, tk.END)
        
        # Run in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._run_fit_worker)
        thread.daemon = True
        thread.start()
        
    def _run_fit_worker(self):
        """Worker function that runs the actual fitting."""
        try:
            # Parse custom t0 values if enabled
            custom_t0 = None
            if self.use_custom_t0.get():
                try:
                    t0_str = self.custom_t0_values.get().strip()
                    if t0_str:
                        custom_t0 = [float(x.strip()) for x in t0_str.split(',')]
                except ValueError as e:
                    error_msg = f"Invalid t0 values format: {e}. Use comma-separated years like: 1950, 1970, 1990"
                    self.root.after(0, lambda: self._display_error(error_msg))
                    return
            
            result = process_richards_fit(
                csv_path=self.csv_file_path,
                year_col=self.year_col.get(),
                value_col=self.value_col.get(),
                input_type=self.input_type.get(),
                fit_on=self.fit_on.get(),
                max_peaks=self.max_peaks.get(),
                restarts=self.restarts.get(),
                seed=self.seed.get(),
                title=self.title.get(),
                output_dir=self.output_dir.get(),
                progress_callback=self.update_progress,
                custom_t0_values=custom_t0
            )
            
            # Update GUI with results
            self.root.after(0, lambda: self._display_results(result))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._display_error(error_msg))
        finally:
            self.root.after(0, lambda: self.progress.stop())
            
    def _display_results(self, result):
        """Display results in the GUI."""
        best = result["best_result"]
        output_files = result["output_files"]
        
        # Clear and populate results text
        self.results_text.delete(1.0, tk.END)
        
        results_text = "=== BEST MODEL ===\n"
        results_text += f"Fit space: {best.fit_space}   Peaks: {best.n_peaks}\n\n"
        
        results_text += "Parameters:\n"
        for i in range(best.n_peaks):
            Qinf, k, t0, nu = best.params[4*i:4*i+4]
            results_text += f"Peak {i+1}: Qinf={Qinf:.6g}, k={k:.6g}, t0={t0:.3f}, nu={nu:.3f}\n"
        
        results_text += "\nMetrics (fit space):\n"
        for k in ["R2","RMSE","AICc","BIC","RSS","n","kparams","success"]:
            results_text += f"  {k}: {best.metrics_fit_space.get(k)}\n"
        
        results_text += "\nDiagnostics in rate space:\n"
        for k in ["R2","RMSE","AICc","BIC","RSS"]:
            results_text += f"  {k}: {best.metrics_rate.get(k)}\n"
        
        results_text += "\nDiagnostics in cumulative space:\n"
        for k in ["R2","RMSE","AICc","BIC","RSS"]:
            results_text += f"  {k}: {best.metrics_cum.get(k)}\n"
        
        results_text += "\n=== OUTPUT FILES ===\n"
        for name, path in output_files.items():
            results_text += f"{name}: {path}\n"
        
        self.results_text.insert(1.0, results_text)
        self.status_label.config(text="Fit completed successfully!")
        
        messagebox.showinfo("Success", "Richards fit completed successfully!\nCheck the Results panel for details.", parent=self.root)
        
    def _display_error(self, error_msg):
        """Display error in the GUI."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, f"ERROR: {error_msg}")
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Error", f"An error occurred:\n{error_msg}", parent=self.root)
        
    def exit_app(self):
        """Exit the application."""
        self.root.quit()
        self.root.destroy()

class RichardsFitGUI:
    """Legacy GUI class for backward compatibility."""
    def __init__(self, root):
        # This is now just a wrapper that launches the new two-stage system
        root.withdraw()  # Hide the root window
        self.launch_two_stage_gui()
        root.quit()
        
    @staticmethod
    def launch_two_stage_gui():
        """Launch the two-stage GUI system."""
        # Stage 1: File selection
        file_dialog = FileSelectionDialog()
        selected_file = file_dialog.show()
        
        if selected_file:
            # Stage 2: Processing options (persistent)
            options_dialog = ProcessingOptionsDialog(selected_file)
            options_dialog.root.mainloop()

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Fit 1–3 Richards peaks; fit on cumulative or rate; plot both.")
    ap.add_argument("csv", help="CSV with columns for year/value.")
    ap.add_argument("--year-col", default="year", help="Year column name (default: year).")
    ap.add_argument("--value-col", default="discovery", help="Value column (default: discovery).")
    ap.add_argument("--input", choices=["rate","cumulative"], default="rate",
                    help="Interpretation of CSV values (default: rate).")
    ap.add_argument("--fit-on", choices=["rate","cumulative","auto"], default="cumulative",
                    help="Which space to fit in (default: cumulative). 'auto' uses the same as --input.")
    ap.add_argument("--max-peaks", type=int, default=3, help="Max peaks to test (1–3).")
    ap.add_argument("--restarts", type=int, default=14, help="Random restarts per model.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--title", default="Richards Fit to Discovery Data", help="Plot title.")
    ap.add_argument("--out-plot-cum", default="richards_fit_cumulative.png", help="Output cumulative plot.")
    ap.add_argument("--out-plot-rate", default="richards_fit_rate.png", help="Output rate plot.")
    ap.add_argument("--out-params", default="richards_params.csv", help="Output parameters CSV.")
    ap.add_argument("--out-json", default="richards_model.json", help="Output JSON report.")
    args = ap.parse_args()

    # Load data
    t, y_raw = load_series(args.csv, args.year_col, args.value_col)
    y_rate, y_cum = ensure_rate_and_cum(t, y_raw, args.input)

    # pick fit space
    fit_on = args.fit_on
    if fit_on == "auto":
        fit_on = "rate" if args.input == "rate" else "cumulative"

    # Sanity filters
    mask = np.isfinite(t) & np.isfinite(y_rate) & np.isfinite(y_cum)
    t, y_rate, y_cum = t[mask], y_rate[mask], y_cum[mask]
    if len(t) < 6:
        raise SystemExit("Need at least 6 valid points.")

    # Fit and model selection
    best, all_results = fit_best_model(
        t, y_rate, y_cum, fit_on, max_peaks=max(1, min(3, args.max_peaks)),
        restarts=args.restarts, seed=args.seed
    )

    # ---- Console summary ----
    print("\n=== Best model ===")
    print(f"Fit space: {best.fit_space}   Peaks: {best.n_peaks}")
    for i in range(best.n_peaks):
        Qinf, k, t0, nu = best.params[4*i:4*i+4]
        print(f"Peak {i+1}: Qinf={Qinf:.6g}, k={k:.6g}, t0={t0:.3f}, nu={nu:.3f}")
    print("\nMetrics (fit space):")
    for k in ["R2","RMSE","AICc","BIC","RSS","n","kparams","success","message","nfev"]:
        print(f"  {k}: {best.metrics_fit_space.get(k)}")
    print("\nDiagnostics in rate space:")
    for k in ["R2","RMSE","AICc","BIC","RSS","n","kparams"]:
        print(f"  {k}: {best.metrics_rate.get(k)}")
    print("\nDiagnostics in cumulative space:")
    for k in ["R2","RMSE","AICc","BIC","RSS","n","kparams"]:
        print(f"  {k}: {best.metrics_cum.get(k)}")

    # ---- Save parameters table ----
    rows = []
    for i in range(best.n_peaks):
        Qinf, k, t0, nu = best.params[4*i:4*i+4]
        rows.append(dict(peak=i+1, Qinf=Qinf, k=k, t0=t0, nu=nu))
    pd.DataFrame(rows).to_csv(args.out_params, index=False)

    # ---- Save JSON report ----
    report = {
        "fit_space": best.fit_space,
        "n_peaks": best.n_peaks,
        "params": best.params,
        "metrics_fit_space": best.metrics_fit_space,
        "metrics_rate": best.metrics_rate,
        "metrics_cumulative": best.metrics_cum,
        "columns_per_peak": ["Qinf","k","t0","nu"]
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- Plots (always both) ----
    plot_cumulative(t, y_cum, best, args.title, args.out_plot_cum)
    plot_rate(t, y_rate, best, args.title, args.out_plot_rate)
    print(f"\nSaved:\n  {args.out_plot_cum}\n  {args.out_plot_rate}\n  {args.out_params}\n  {args.out_json}")

def main_gui():
    """Launch the two-stage GUI application."""
    RichardsFitGUI.launch_two_stage_gui()

if __name__ == "__main__":
    import sys
    
    # Check if GUI mode is requested or if no arguments provided
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--gui', '-g']):
        main_gui()
    else:
        main()
