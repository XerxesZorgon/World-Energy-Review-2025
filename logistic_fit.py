# file: logistic_fit.py
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# -------- Logistic model utilities --------

def logistic(x, Q, k, m, C):
    """
    Logistic with linear offset C:
        y(x) = Q / (1 + exp(-k*(x - m))) + C
    """
    return Q / (1.0 + np.exp(-k * (x - m))) + C


@dataclass
class LogisticModel:
    Q: float
    k: float
    m: float
    C: float
    cov: Optional[np.ndarray] = None  # covariance from curve_fit
    rss: Optional[float] = None       # residual sum of squares
    n: Optional[int] = None           # number of points used

    def predict(self, years: np.ndarray) -> np.ndarray:
        return logistic(np.asarray(years, dtype=float), self.Q, self.k, self.m, self.C)

    def summary(self) -> str:
        lines = [
            "Logistic fit summary",
            f"  Q (scale):    {self.Q:.6g}",
            f"  k (slope):    {self.k:.6g}",
            f"  m (midpoint): {self.m:.6g}",
            f"  C (offset):   {self.C:.6g}",
            f"  n (points):   {self.n}",
        ]
        if self.rss is not None:
            lines.append(f"  RSS:          {self.rss:.6g}")
        if self.cov is not None:
            try:
                perr = np.sqrt(np.diag(self.cov))
                lines.append("  Std errors:   " + ", ".join(f"{p:.3g}" for p in perr))
            except Exception:
                pass
        return "\n".join(lines)


# -------- Core functions (R -> Python) --------

def fit2Data(df: pd.DataFrame, make_plots: bool = False, value_col: str = "X2PC") -> LogisticModel:
    """
    Translate the R fit2Data. Expects df with columns:
      - 'Year': numeric years
      - value_col (default 'X2PC'): discovery/production cumulative values

    Returns a LogisticModel with fitted params and RSS.
    """
    import warnings

    # Discovery data to fit
    years = np.asarray(df["Year"], dtype=float)
    y = np.asarray(df[value_col], dtype=float)

    # Initial parameters (mirroring the R code)
    C0 = float(np.nanmin(y))
    Q0 = float(np.nanmax(y) - C0)
    if Q0 <= 0:
        # Fallback to range if data is flat or reversed
        Q0 = float((np.nanmax(y) - np.nanmin(y)) or 1.0)

    # Approximate slope near the midpoint
    n = len(y)
    idx_mid = int(round(n / 2))  # R is 1-based; our indexing is 0-based; we’ll guard bounds
    idx_mid = max(1, min(n - 2, idx_mid))  # ensure idx_mid-1 and idx_mid+1 are valid
    dy = y[idx_mid + 1] - y[idx_mid - 1]
    k0 = 2.0 * dy / max(Q0, 1e-12)  # avoid divide-by-zero
    # mildly constrain initial k0
    if not np.isfinite(k0) or abs(k0) < 1e-6:
        k0 = 0.05

    m0 = float(np.nanmean(years))

    p0 = [Q0, k0, m0, C0]

    # Reasonable bounds to stabilize fit (Q>0, small<k<10, m near data, C free)
    lower = [1e-12, 1e-6, float(np.nanmin(years) - 50.0), -np.inf]
    upper = [np.inf, 10.0, float(np.nanmax(years) + 50.0), np.inf]

    # Fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            popt, pcov = curve_fit(
                logistic,
                years,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=200000,
            )
        except Exception as e:
            # If bounded fit fails, try unbounded as a fallback
            popt, pcov = curve_fit(
                logistic,
                years,
                y,
                p0=p0,
                maxfev=200000,
            )

    # Compute RSS
    yhat = logistic(years, *popt)
    rss = float(np.sum((y - yhat) ** 2))

    model = LogisticModel(Q=popt[0], k=popt[1], m=popt[2], C=popt[3], cov=pcov, rss=rss, n=len(y))

    # Optional plot (similar to R)
    if make_plots:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(years, y)
            plt.plot(years, yhat)
            plt.xlabel("Year")
            plt.ylabel("Discoveries (Gb)")
            plt.title("Cumulative Oil Discoveries (Logistic fit)")
            plt.show()
        except Exception:
            pass

    # Print summary like R's summary(logFit)
    print(model.summary())
    return model


def writeFit2xl(LF1: LogisticModel, LF2: LogisticModel, xl_fname: str) -> None:
    """
    Translate writeFit2xl:
      - Predict 2020..2050 for each model
      - Combine as futProd = pred1 + pred2 - pred2[1]  (R is 1-based; that's pred2 at 2020)
      - Write a single column Excel file without headers
    """
    years = np.arange(2020, 2051, dtype=float)
    pred1 = LF1.predict(years)
    pred2 = LF2.predict(years)

    # R's pred2[1] corresponds to Python's pred2[0]
    futProd = pred1 + pred2 - pred2[0]

    # Write to Excel (no col/row names)
    # This will create a single-column sheet named Sheet1 like the R code.
    df_out = pd.DataFrame(futProd)
    with pd.ExcelWriter(xl_fname, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, sheet_name="Sheet1", header=False, index=False)


def fit2dblLogistic(PD: pd.DataFrame, value_col: str = "X2PC",
                    break_range: Tuple[int, int] = (95, 105)) -> Tuple[LogisticModel, LogisticModel, int]:
    """
    Translate fit2dblLogistic:
      - Try breakpoints over indices [95..105]
      - For each breakpoint i, fit head(PD, i) and tail(PD, nYrs - i)
      - Use total error to pick best (here: sum of RSS)
      - Return (L1, L2, i)

    Notes:
      * The R code uses convInfo$finTol from nls as an 'error'. curve_fit doesn't expose that,
        so we use residual sum of squares (RSS) as a comparable scalar error metric.
      * break_range refers to row indices (1-based in R spirit); here we treat them as Python
        indices directly. Adjust if your dataset is shorter.
    """
    nYrs = len(PD)
    minErr = math.inf
    best: Optional[Tuple[LogisticModel, LogisticModel, int]] = None

    lo, hi = break_range
    for i in range(lo, hi + 1):
        if i <= 2 or i >= nYrs - 2:
            continue  # need at least 3 points per segment for a sensible fit

        head_df = PD.iloc[:i].copy()
        tail_df = PD.iloc[i:].copy()

        # Skip if segments are empty or have NaNs only
        if head_df[value_col].notna().sum() < 3 or tail_df[value_col].notna().sum() < 3:
            continue

        try:
            L1 = fit2Data(head_df, make_plots=False, value_col=value_col)
            L2 = fit2Data(tail_df, make_plots=False, value_col=value_col)
            err = (L1.rss or math.inf) + (L2.rss or math.inf)
        except Exception:
            continue

        if np.isfinite(err) and err < minErr:
            minErr = err
            best = (L1, L2, i)

    if best is None:
        raise RuntimeError("Double-logistic fit failed across all breakpoints in range "
                           f"{break_range}. Check data length and values.")

    return best


# -------- Example usage (mirrors the R top-of-file comment) --------
if __name__ == "__main__":
    # Load data (adjust the path/column names as needed)
    # The R code used: read_excel("../data/DiscProd.xlsx")
    prodData = pd.read_excel("../data/DiscProd.xlsx")

    # Example: run a single logistic fit on the first 100 rows
    # logFit = fit2Data(prodData.head(100))

    # Example: run a double-logistic search on the full data
    # L1, L2, idx = fit2dblLogistic(prodData, value_col="X2PC", break_range=(95, 105))
    # print("Best breakpoint index:", idx)
    # print(L1.summary())
    # print(L2.summary())

    # Example: write combined future projection to Excel
    # writeFit2xl(L1, L2, "future_projection.xlsx")
