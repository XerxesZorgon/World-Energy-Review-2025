#!/usr/bin/env python3
"""
statistical_tests.py
--------------------
Statistical checks that observed vs IMPUTED quantities come
from the same distribution.

Usage
-----
    python statistical_tests.py Oil    # operates on Oil_fields
    python statistical_tests.py Gas    # operates on Gas_fields
    python statistical_tests.py Coal   # operates on Coal_open_mines
"""

import argparse, pathlib, sqlite3, sys, warnings, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from scipy import stats
from scipy.stats import cramervonmises_2samp, anderson_ksamp
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Define constants
SCRIPT_DIR = os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'Energy.db')

# optional but nice-to-have libraries -------------------------------
try:
    from hyppo.ksample import Energy
    HAVE_HYPOP = True
except ImportError:
    HAVE_HYPOP = False
    warnings.warn("hyppo not installed – energy distance test skipped.")

# Clean up noisy warnings for a smoother UX
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# -------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(
        description="Compare observed vs imputed field quantities."
    )
    p.add_argument("fuel", choices=["Oil", "Gas", "Coal"], help="Which table to analyse")
    # Default DB under ../data/Energy.db relative to this script
    # statistical_tests.py lives in energy_etl/, and data/ is a sibling directory
    default_db = DB_PATH
    p.add_argument("--db", default=str(default_db), help="Path to SQLite database")
    p.add_argument("--energy-reps", type=int, default=300,
                   help="Repetitions for hyppo energy test (0 to skip)")
    p.add_argument("--use-log-tests", action="store_true",
                   help="Run two-sample tests on log10 quantity instead of raw EJ")
    p.add_argument("--stratified", action="store_true",
                   help="Emit per-stratum KS/AD and aggregate via Fisher")
    p.add_argument("--reweight", action="store_true",
                   help="Approximate reweighting: resample imputed to match observed stratum proportions")
    p.add_argument("--min-stratum-n", type=int, default=10,
                   help="Minimum per-source count in a stratum to run per-stratum tests")
    return p.parse_args()


def load_table(db_path: str, table: str) -> pd.DataFrame:
    # Select only needed columns if present to reduce memory
    with sqlite3.connect(db_path) as con:
        cols_info = pd.read_sql(f"PRAGMA table_info('{table}')", con)
        available = set(cols_info["name"].tolist())
        
        # Coal uses different column names
        if "Coal" in table:
            wanted = [
                "reserves_initial_EJ",
                "reserves_initial_EJ_calibrated",  # if calibrated
                "_imputed_flag_qty",
                "_imputed_flag_year",  # needed for coal source assignment
                "_calibrated_flag_quantity",  # if calibrated
                "opening_year_final",
                "Country / Area",
            ]
        else:
            wanted = [
                "Quantity_initial_EJ",
                "Quantity_initial_EJ_calibrated",
                "_imputed_flag_quantity",
                "_calibrated_flag_quantity",
                "discovery_year_final",
                "Country / Area",
            ]
        select_cols = [c for c in wanted if c in available]
        if not select_cols:
            # Fallback
            df = pd.read_sql(f'SELECT * FROM "{table}"', con)
        else:
            quoted = ", ".join([f'"{c}"' for c in select_cols])
            df = pd.read_sql(f'SELECT {quoted} FROM "{table}"', con)
    
    # Check required columns based on table type
    if "Coal" in table:
        if "_imputed_flag_qty" not in df.columns:
            sys.exit(f"{table} lacks _imputed_flag_qty column.")
        if "reserves_initial_EJ" not in df.columns:
            sys.exit(f"{table} lacks reserves_initial_EJ column.")
    else:
        if "_imputed_flag_quantity" not in df.columns:
            sys.exit(f"{table} lacks _imputed_flag_quantity column.")
        if "Quantity_initial_EJ" not in df.columns:
            sys.exit(f"{table} lacks Quantity_initial_EJ column.")
    return df


def prep_data(df):
    # Determine column names based on table type
    is_coal = "reserves_initial_EJ" in df.columns
    
    if is_coal:
        # Coal uses different column names
        base_qty_col = "reserves_initial_EJ"
        cal_qty_col = "reserves_initial_EJ_calibrated"
        flag_col = "_imputed_flag_qty"
        year_col = "opening_year_final"
    else:
        # Oil/Gas columns
        base_qty_col = "Quantity_initial_EJ"
        cal_qty_col = "Quantity_initial_EJ_calibrated"
        flag_col = "_imputed_flag_quantity"
        year_col = "discovery_year_final"
    
    # Prefer calibrated quantity if available
    q_col = base_qty_col
    if cal_qty_col in df.columns:
        # Use calibrated where available and positive, else fall back
        df["_Q_eff"] = df[base_qty_col]
        mask_cal = df[cal_qty_col].notna() & (df[cal_qty_col] > 0)
        df.loc[mask_cal, "_Q_eff"] = df.loc[mask_cal, cal_qty_col]
        q_col = "_Q_eff"
    
    # keep only strictly positive quantities for log
    df = df.loc[df[q_col].notna() & (df[q_col] > 0)].copy()
    
    # For coal, use year flag since all quantities are imputed
    # For oil/gas, use quantity flag as usual
    if is_coal:
        df["Source"] = np.where(df["_imputed_flag_year"] == 0, "Observed", "Imputed")
    else:
        df["Source"] = np.where(df[flag_col] == 0, "Observed", "Imputed")
    
    df["logQ"] = np.log10(df[q_col])
    
    # Derive decade for optional stratification
    if year_col in df.columns:
        yrs = pd.to_numeric(df[year_col], errors="coerce")
        df["decade"] = (np.floor(yrs/10.0)*10).astype("Int64")
    else:
        df["decade"] = pd.Series([pd.NA]*len(df), dtype="Int64")
    return df


# ───────────────────────── plots ────────────────────────────────────
def make_plots(df, fuel):
    plot_dir = pathlib.Path("plots")
    plot_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")

    # KDE
    plt.figure()
    sns.kdeplot(
        data=df, x="logQ", hue="Source",
        common_norm=False, fill=True, alpha=.4
    )
    # Dynamic xlabel based on fuel type
    qty_label = "reserves_initial_EJ" if fuel == "Coal" else "Quantity_initial_EJ"
    plt.xlabel(f"log₁₀ {qty_label}")
    plt.title(f"{fuel}: Kernel Density")
    plt.tight_layout(); plt.savefig(plot_dir/f"{fuel}_kde.png", dpi=300); plt.close()

    # Box-plot + Violin
    plt.figure()
    sns.violinplot(data=df, x="Source", y="logQ", inner="quartile")
    plt.title(f"{fuel}: Violin/Box")
    plt.tight_layout(); plt.savefig(plot_dir/f"{fuel}_violin.png", dpi=300); plt.close()

    # Q-Q
    for src, g in df.groupby("Source"):
        plt.figure()
        stats.probplot(g["logQ"], dist="norm", plot=plt)
        plt.title(f"{fuel} – {src}: Q-Q")
        plt.tight_layout(); plt.savefig(plot_dir/f"{fuel}_qq_{src}.png", dpi=300)
        plt.close()

    # ECDF overlay
    plt.figure()
    for src, g in df.groupby("Source"):
        x = np.sort(g["logQ"])
        y = np.arange(1, len(x)+1)/len(x)
        plt.step(x, y, where="post", label=src)
    plt.legend(); plt.xlabel("logQ"); plt.ylabel("ECDF")
    plt.title(f"{fuel}: Empirical CDF")
    plt.tight_layout(); plt.savefig(plot_dir/f"{fuel}_ecdf.png", dpi=300); plt.close()


# ─────────────────────── statistical tests ─────────────────────────
def _choose_series(df: pd.DataFrame, use_log: bool) -> tuple[pd.Series, pd.Series]:
    if use_log:
        col = "logQ"
    elif "_Q_eff" in df.columns:
        col = "_Q_eff"
    elif "reserves_initial_EJ" in df.columns:
        col = "reserves_initial_EJ"  # Coal
    else:
        col = "Quantity_initial_EJ"  # Oil/Gas fallback
    obs = df.loc[df["Source"] == "Observed", col]
    imp = df.loc[df["Source"] == "Imputed", col]
    return obs, imp


def run_tests(df, energy_reps: int = 300, use_log: bool = False):
    obs, imp = _choose_series(df, use_log)

    results = []

    # Sample-size guard
    if len(obs) < 5 or len(imp) < 5:
        results.append(("Insufficient samples for two-sample tests", float("nan"), float("nan")))
        return results

    # Kolmogorov–Smirnov
    ks = stats.ks_2samp(obs, imp, alternative="two-sided")
    results.append(("Kolmogorov–Smirnov D", ks.statistic, ks.pvalue))

    # Mann-Whitney U
    mw = stats.mannwhitneyu(obs, imp, alternative="two-sided", method="asymptotic")
    results.append(("Mann–Whitney U", mw.statistic, mw.pvalue))

    # Anderson–Darling
    try:
        ad = anderson_ksamp([obs.values, imp.values])
        # significance_level is percent in some SciPy versions
        p_ad = getattr(ad, "significance_level", np.nan)
        if not np.isnan(p_ad):
            p_ad = p_ad/100.0
        results.append(("Anderson–Darling A²", ad.statistic, p_ad))
    except Exception:
        results.append(("Anderson–Darling A²", np.nan, np.nan))

    # Cramér–von Mises
    cvm = cramervonmises_2samp(obs, imp)
    results.append(("Cramér–von Mises T", cvm.statistic, cvm.pvalue))

    # Wasserstein
    w = stats.wasserstein_distance(obs, imp)
    results.append(("Wasserstein EMD", w, np.nan))

    # Energy distance
    if HAVE_HYPOP and energy_reps > 0:
        try:
            res = Energy().test(
                obs.values.reshape(-1, 1), imp.values.reshape(-1, 1), reps=energy_reps
            )
            en_stat = en_p = np.nan
            if isinstance(res, tuple):
                if len(res) >= 2:
                    en_stat, en_p = res[0], res[1]
                elif len(res) == 1:
                    en_stat, en_p = res[0], np.nan
            else:
                en_stat = getattr(res, "stat", np.nan)
                en_p = getattr(res, "pvalue", np.nan)
            results.append(("Energy distance", en_stat, en_p))
        except Exception as e:
            warnings.warn(f"Energy test failed: {e}")
            results.append(("Energy distance", np.nan, np.nan))
    else:
        results.append(("Energy distance", np.nan, np.nan))

    # Logistic separability
    base_qty = "reserves_initial_EJ" if "reserves_initial_EJ" in df.columns else "Quantity_initial_EJ"
    feats = ["logQ" if use_log else ("_Q_eff" if "_Q_eff" in df.columns else base_qty)]
    if "discovery_year_final" in df.columns:
        feats.append("discovery_year_final")
    X = df[feats].copy()
    if "discovery_year_final" in X.columns:
        X["discovery_year_final"] = X["discovery_year_final"].fillna(X["discovery_year_final"].median())
    y = (df["Source"] == "Imputed").astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    # Holdout AUC for realism
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:,1])
    results.append(("AUC separability (≃0.5 good)", auc, np.nan))

    return results


# ───────────────── stratified diagnostics and reweighting ─────────────────
def reweight_imputed(df: pd.DataFrame, min_stratum_n: int = 10) -> pd.DataFrame:
    # Define strata as Country/Area × decade when available
    strata_cols = []
    if "Country / Area" in df.columns:
        strata_cols.append("Country / Area")
    if "decade" in df.columns:
        strata_cols.append("decade")
    if not strata_cols:
        return df.copy()
    df2 = df.copy()
    # Observed and imputed distributions
    obs_mask = df2["Source"] == "Observed"
    imp_mask = ~obs_mask
    n_imp = imp_mask.sum()
    # proportions by stratum in observed
    obs_props = (
        df2.loc[obs_mask].groupby(strata_cols).size().rename("n").pipe(lambda s: s/s.sum())
    )
    imp_counts = df2.loc[imp_mask].groupby(strata_cols).size().rename("n").to_frame()
    # target counts for imputed matching observed mixture
    target = (obs_props * n_imp).round().astype(int)
    # resample within each stratum to target count
    sampled_frames = []
    for key, tgt in target.items():
        try:
            grp = df2.loc[imp_mask]
            for c, v in zip(strata_cols, key if isinstance(key, tuple) else (key,)):
                grp = grp[grp[c] == v]
            if tgt <= 0 or grp.empty:
                continue
            take = grp.sample(n=int(tgt), replace=len(grp) < tgt, random_state=42)
            sampled_frames.append(take)
        except Exception:
            continue
    if sampled_frames:
        imp_rew = pd.concat(sampled_frames, axis=0)
        df_out = pd.concat([df2.loc[obs_mask], imp_rew.assign(Source="Imputed")], axis=0, ignore_index=True)
        return df_out
    return df2


def stratified_tests(df: pd.DataFrame, energy_reps: int = 0, use_log: bool = False, min_stratum_n: int = 10):
    results = []
    pvals = []
    strata_cols = []
    if "Country / Area" in df.columns:
        strata_cols.append("Country / Area")
    if "decade" in df.columns:
        strata_cols.append("decade")
    if not strata_cols:
        return results, np.nan
    for key, g in df.groupby(strata_cols):
        obs, imp = _choose_series(g, use_log)
        if (obs.notna().sum() < min_stratum_n) or (imp.notna().sum() < min_stratum_n):
            continue
        try:
            ks = stats.ks_2samp(obs, imp)
            results.append((f"KS {key}", ks.statistic, ks.pvalue))
            if not np.isnan(ks.pvalue):
                pvals.append(max(ks.pvalue, 1e-12))
        except Exception:
            continue
    # Fisher aggregation of p-values
    if pvals:
        chi2 = -2.0 * np.sum(np.log(pvals))
        df_deg = 2 * len(pvals)
        agg_p = 1 - stats.chi2.cdf(chi2, df_deg)
    else:
        agg_p = np.nan
    return results, agg_p

    # Kolmogorov–Smirnov
    ks = stats.ks_2samp(obs, imp, alternative="two-sided")
    results.append(("Kolmogorov–Smirnov D", ks.statistic, ks.pvalue))

    # Mann-Whitney U
    mw = stats.mannwhitneyu(obs, imp, alternative="two-sided", method="asymptotic")
    results.append(("Mann–Whitney U", mw.statistic, mw.pvalue))

    # Anderson–Darling
    try:
        ad = anderson_ksamp([obs.values, imp.values])
        # significance_level is percent in some SciPy versions
        p_ad = getattr(ad, "significance_level", np.nan)
        if not np.isnan(p_ad):
            p_ad = p_ad/100.0
        results.append(("Anderson–Darling A²", ad.statistic, p_ad))
    except Exception:
        results.append(("Anderson–Darling A²", np.nan, np.nan))

    # Cramér–von Mises
    cvm = cramervonmises_2samp(obs, imp)
    results.append(("Cramér–von Mises T", cvm.statistic, cvm.pvalue))

    # Wasserstein
    w = stats.wasserstein_distance(obs, imp)
    results.append(("Wasserstein EMD", w, np.nan))

    # Energy distance
    if HAVE_HYPOP and energy_reps > 0:
        try:
            res = Energy().test(
                obs.values.reshape(-1, 1), imp.values.reshape(-1, 1), reps=energy_reps
            )
            en_stat = en_p = np.nan
            if isinstance(res, tuple):
                if len(res) >= 2:
                    en_stat, en_p = res[0], res[1]
                elif len(res) == 1:
                    en_stat, en_p = res[0], np.nan
            else:
                en_stat = getattr(res, "stat", np.nan)
                en_p = getattr(res, "pvalue", np.nan)
            results.append(("Energy distance", en_stat, en_p))
        except Exception as e:
            warnings.warn(f"Energy test failed: {e}")
            results.append(("Energy distance", np.nan, np.nan))
    else:
        results.append(("Energy distance", np.nan, np.nan))

    # Logistic separability
    base_qty = "reserves_initial_EJ" if "reserves_initial_EJ" in df.columns else "Quantity_initial_EJ"
    feats = ["logQ" if use_log else ("_Q_eff" if "_Q_eff" in df.columns else base_qty)]
    if "discovery_year_final" in df.columns:
        feats.append("discovery_year_final")
    X = df[feats].copy()
    if "discovery_year_final" in X.columns:
        X["discovery_year_final"] = X["discovery_year_final"].fillna(X["discovery_year_final"].median())
    y = (df["Source"] == "Imputed").astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    # Holdout AUC for realism
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:,1])
    results.append(("AUC separability (≃0.5 good)", auc, np.nan))

    return results


def save_report(results, fuel):
    out_txt = f"results_{fuel}.txt"
    out_tsv = f"results_{fuel}.tsv"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Statistical comparison: {fuel} ===\n")
        f.write("Metric\tStatistic\tp-value\n")
        for name, stat, p in results:
            f.write(f"{name}\t{stat:.5g}\t{'' if np.isnan(p) else f'{p:.4g}'}\n")
    # TSV for downstream ingestion
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("Metric\tStatistic\tp_value\n")
        for name, stat, p in results:
            p_str = "" if np.isnan(p) else f"{p:.6g}"
            f.write(f"{name}\t{stat:.6g}\t{p_str}\n")
    print(f"Report saved → {out_txt}, {out_tsv}")


# -------------------------------------------------------------------
def main():
    args = parse_cli()
    # Map fuel to table name
    if args.fuel == "Coal":
        table = "Coal_open_mines"
    else:
        table = f"{args.fuel}_fields"      # Oil_fields or Gas_fields
    df_raw = load_table(args.db, table)
    df = prep_data(df_raw)

    if df["Source"].nunique() < 2:
        sys.exit("Table contains only observed or only imputed rows – nothing to compare.")

    print(f"{args.fuel}: {len(df)} rows "
          f"({(df.Source=='Observed').sum()} observed, {(df.Source=='Imputed').sum()} imputed)")

    make_plots(df, args.fuel)
    print("Plots saved to plots/ folder.")

    # Save reproducible analysis frame
    try:
        df.to_parquet(pathlib.Path("plots")/f"{args.fuel}_analysis_frame.parquet", index=False)  # type: ignore[arg-type]
    except Exception:
        df.to_csv(pathlib.Path("plots")/f"{args.fuel}_analysis_frame.csv", index=False)

    # Optional reweighting (approximate via stratified resampling of imputed)
    df_test = reweight_imputed(df, min_stratum_n=args.min_stratum_n) if args.reweight else df

    res = run_tests(df_test, energy_reps=args.energy_reps, use_log=args.use_log_tests)
    for name, stat, p in res:
        print(f"{name:<28}  stat={stat:.5g}" + (f", p={p:.4g}" if not np.isnan(p) else ""))

    save_report(res, args.fuel)

    # Optional stratified diagnostics
    if args.stratified:
        strat_res, agg_p = stratified_tests(df, energy_reps=0, use_log=args.use_log_tests, min_stratum_n=args.min_stratum_n)
        if strat_res:
            with open(f"results_{args.fuel}_stratified.tsv", "w", encoding="utf-8") as f:
                f.write("Stratum\tKS_stat\tp_value\n")
                for name, stat, p in strat_res:
                    f.write(f"{name}\t{stat:.6g}\t{p:.6g}\n")
            print(f"Stratified KS saved → results_{args.fuel}_stratified.tsv; Fisher aggregate p={agg_p:.4g}.")


if __name__ == "__main__":
    # create plot directory if absent
    os.makedirs("plots", exist_ok=True)
    main()
