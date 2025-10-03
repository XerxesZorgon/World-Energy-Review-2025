#!/usr/bin/env python
"""
impute_oil_gas_db.py – v2
Populate discovery_year_final and Quantity_initial_EJ in Oil_Gas_fields
using ONLY Oil_Gas_Production_Reserves + Oil_Gas_fields.

Run:  python impute_oil_gas_db.py --db Energy.db
"""

import argparse, logging, sqlite3, re, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

from convert_gas_oil_to_EJ import convert_to_ej  # local helper

LOG = logging.getLogger(__name__)
INIT_PAT = re.compile(r"(initial|original|ultimate recovery|EUR)", re.I)
RATE_PAT = re.compile(r"(/y|/d|\bbpd\b)", re.I)
VOL_PAT = re.compile(r"(/y|/d|\bbpd\b)", re.I)  # to strip rate suffix

# ───────────────────────── helpers ───────────────────────────────────────────
def to_EJ(volume: float, unit: str, fuel: str) -> float | None:
    """Convert any *volume* unit to EJ using supplied helper."""
    try:
        # Strip possible "/y", "/d", "bpd" to get the pure volume unit
        unit_clean = VOL_PAT.sub("", unit).strip()
        result = convert_to_ej(volume, unit_clean, fuel_type=fuel)
        if isinstance(result, dict) and 'total_ej' in result:
            return float(result['total_ej'])
        return None
    except Exception:
        return None


def rate_to_EJ_per_year(rate: float, unit: str, fuel: str) -> float | None:
    """
    Convert a production *rate* (per year or per day) to EJ per year.
    """
    try:
        unit = str(unit).lower()
        if "/d" in unit or "bpd" in unit:
            ej_per_day = to_EJ(rate, unit.replace("/d", "").replace("bpd", ""), fuel)
            return ej_per_day * 365 if ej_per_day is not None else None
        elif "/y" in unit:
            return to_EJ(rate, unit.replace("/y", ""), fuel)
        else:  # assume it's already a volume unit
            return to_EJ(rate, unit, fuel)
    except Exception:
        return None


def add_missing_cols(conn: sqlite3.Connection):
    cols = {
        "discovery_year_final": "INTEGER",
        "_imputed_flag_year": "INTEGER",
        "_imputed_by_year": "TEXT",
        "_imputation_conf_year": "REAL",
        "Quantity_initial_EJ": "REAL",
        "_imputed_flag_quantity": "INTEGER",
        "_imputed_by_quantity": "TEXT",
        "_imputation_conf_quantity": "REAL",
    }
    cur = conn.execute("PRAGMA table_info('Oil_Gas_fields')")
    existing = {r[1] for r in cur.fetchall()}
    for c, t in cols.items():
        if c not in existing:
            conn.execute(f"ALTER TABLE Oil_Gas_fields ADD COLUMN {c} {t};")
    conn.commit()


# ───────────────────────── YEAR SECTION ──────────────────────────────────────
def copy_and_impute_year(conn: sqlite3.Connection):
    # direct copy
    conn.execute("""
        UPDATE Oil_Gas_fields
        SET discovery_year_final = "Discovery year",
            _imputed_flag_year   = 0,
            _imputed_by_year     = 'observed',
            _imputation_conf_year= 1.0
        WHERE "Discovery year" IS NOT NULL
          AND discovery_year_final IS NULL
    """)
    conn.commit()

    df = pd.read_sql("SELECT * FROM Oil_Gas_fields", conn)
    mask = df["discovery_year_final"].isna()

    if not mask.any():
        LOG.info("All discovery years already filled.")
        return

    # Check available columns and preprocess
    feats_num = ["FID Year", "Production start year", "Status year"]
    feats_cat = ["Country/Area", "Fuel type", "Status", "Basin"]
    
    # Filter to only existing columns
    feats_num = [c for c in feats_num if c in df.columns]
    feats_cat = [c for c in feats_cat if c in df.columns]
    
    if not feats_num and not feats_cat:
        LOG.warning("No feature columns available for year imputation")
        return
    
    # Preprocess numeric features: coerce to numeric, handle missing
    for col in feats_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # Preprocess categorical features: fill missing
    for col in feats_cat:
        df[col] = df[col].fillna('Unknown')
    
    X = df[feats_num + feats_cat]
    y = df["discovery_year_final"]

    ct = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), feats_cat),
        remainder="passthrough",
    )
    pipe = Pipeline([
        ("prep", ct),
        ("imp", IterativeImputer(
            estimator=HistGradientBoostingRegressor(max_depth=6, random_state=0),
            random_state=0, max_iter=20, sample_posterior=True)),
    ])
    pipe.fit(X, y)

    imputed = np.round(pipe.predict(X.loc[mask])).astype(int)
    df.loc[mask, "discovery_year_final"] = imputed
    df.loc[mask, "_imputed_flag_year"] = 1
    df.loc[mask, "_imputed_by_year"] = "IterativeImputer+HGB"
    df.loc[mask, "_imputation_conf_year"] = 0.5

    df[["Unit ID", "discovery_year_final", "_imputed_flag_year",
        "_imputed_by_year", "_imputation_conf_year"]].to_sql(
        "tmp_year", conn, if_exists="replace", index=False)
    conn.executescript("""
        UPDATE Oil_Gas_fields AS f
        SET discovery_year_final   = (SELECT discovery_year_final FROM tmp_year WHERE tmp_year."Unit ID"=f."Unit ID"),
            _imputed_flag_year     = (SELECT _imputed_flag_year     FROM tmp_year WHERE tmp_year."Unit ID"=f."Unit ID"),
            _imputed_by_year       = (SELECT _imputed_by_year       FROM tmp_year WHERE tmp_year."Unit ID"=f."Unit ID"),
            _imputation_conf_year  = (SELECT _imputation_conf_year  FROM tmp_year WHERE tmp_year."Unit ID"=f."Unit ID")
        WHERE f."Unit ID" IN (SELECT "Unit ID" FROM tmp_year);
        DROP TABLE tmp_year;
    """)
    conn.commit()
    LOG.info("Imputed discovery year for %d fields.", mask.sum())


# ───────────────────────── QUANTITY SECTION ──────────────────────────────────
def load_pr_table(conn: sqlite3.Connection) -> pd.DataFrame:
    pr = pd.read_sql("SELECT * FROM Oil_Gas_Production_Reserves", conn)
    # normalise fuel text
    pr["fuel_norm"] = pr["Fuel description"].str.lower().str.extract("(oil|gas)", expand=False)
    return pr


def first_initial_rows(pr: pd.DataFrame) -> pd.DataFrame:
    mask = pr["Reserves classification (original)"].str.contains(INIT_PAT, na=False, regex=True)
    init = pr[mask & pr["Quantity (converted)"].notna()].copy()
    # earliest data year per field
    init.sort_values("Data year", inplace=True)
    return init.drop_duplicates("Unit ID")


def latest_current_reserves(pr: pd.DataFrame) -> pd.DataFrame:
    """Aggregate current remaining reserves by Unit ID across all fuel types."""
    mask_rate = pr["Units (converted)"].str.contains(RATE_PAT, na=False, regex=True)
    mask_init = pr["Reserves classification (original)"].str.contains(INIT_PAT, na=False, regex=True)
    cur = pr[(~mask_rate) & (~mask_init) & pr["Quantity (converted)"].notna()].copy()
    
    # Group by Unit ID and sum all fuel types
    unit_reserves = {}
    for _, r in cur.iterrows():
        try:
            q_ej = to_EJ(r["Quantity (converted)"], r["Units (converted)"], r["fuel_norm"])
            if q_ej is not None and q_ej > 0:
                uid = r["Unit ID"]
                if uid not in unit_reserves:
                    unit_reserves[uid] = {
                        "Unit ID": uid,
                        "Country/Area": r["Country/Area"],
                        "fuel_norm": "aggregated",
                        "Quantity (converted)": 0.0,
                        "Units (converted)": "EJ",
                        "Data year": r["Data year"]
                    }
                unit_reserves[uid]["Quantity (converted)"] += q_ej
                # Keep latest data year
                if r["Data year"] > unit_reserves[uid]["Data year"]:
                    unit_reserves[uid]["Data year"] = r["Data year"]
        except Exception as e:
            LOG.warning(f"Failed to convert reserves for {r['Unit ID']}: {e}")
    
    result = pd.DataFrame(list(unit_reserves.values()))
    LOG.info(f"Aggregated reserves for {len(result)} fields from {len(cur)} records")
    return result


def latest_prod_rates(pr: pd.DataFrame) -> pd.DataFrame:
    """Aggregate production rates by Unit ID across all fuel types."""
    mask_rate = pr["Units (converted)"].str.contains(RATE_PAT, na=False, regex=True)
    prod = pr[mask_rate & pr["Quantity (converted)"].notna()].copy()
    
    # Group by Unit ID and sum all fuel types
    unit_prod = {}
    for _, r in prod.iterrows():
        try:
            p_ej = rate_to_EJ_per_year(r["Quantity (converted)"], r["Units (converted)"], r["fuel_norm"])
            if p_ej is not None and p_ej > 0:
                uid = r["Unit ID"]
                if uid not in unit_prod:
                    unit_prod[uid] = {
                        "Unit ID": uid,
                        "Country/Area": r["Country/Area"],
                        "fuel_norm": "aggregated",
                        "Quantity (converted)": 0.0,
                        "Units (converted)": "EJ/y",
                        "Data year": r["Data year"]
                    }
                unit_prod[uid]["Quantity (converted)"] += p_ej
                # Keep latest data year
                if r["Data year"] > unit_prod[uid]["Data year"]:
                    unit_prod[uid]["Data year"] = r["Data year"]
        except Exception as e:
            LOG.warning(f"Failed to convert production for {r['Unit ID']}: {e}")
    
    result = pd.DataFrame(list(unit_prod.values()))
    LOG.info(f"Aggregated production for {len(result)} fields from {len(prod)} records")
    return result


def convert_initials(pr_init: pd.DataFrame) -> pd.DataFrame:
    """Convert and aggregate all initial quantities by Unit ID across all fuel types."""
    unit_totals = {}
    
    for _, r in pr_init.iterrows():
        try:
            q_ej = to_EJ(r["Quantity (converted)"], r["Units (converted)"], r["fuel_norm"])
            if q_ej is not None and q_ej > 0:
                uid = r["Unit ID"]
                if uid not in unit_totals:
                    unit_totals[uid] = 0.0
                unit_totals[uid] += q_ej
                LOG.debug(f"Added {q_ej:.6f} EJ from {r['fuel_norm']} for {uid}")
        except Exception as e:
            LOG.warning(f"Failed to convert initial quantity for {r['Unit ID']}: {e}")
    
    # Convert to DataFrame
    out = [{"Unit ID": uid, "Quantity_initial_EJ": total} 
           for uid, total in unit_totals.items()]
    
    LOG.info(f"Aggregated initial quantities for {len(out)} fields from {len(pr_init)} records")
    return pd.DataFrame(out)


def populate_initial_observed(conn, pr_init_df: pd.DataFrame):
    df = convert_initials(pr_init_df)
    if df.empty:
        return
    df["_imputed_flag_quantity"] = 0
    df["_imputed_by_quantity"] = "observed_initial_PR"
    df["_imputation_conf_quantity"] = 1.0
    df.to_sql("tmp_init", conn, if_exists="replace", index=False)
    conn.executescript("""
        UPDATE Oil_Gas_fields AS f
        SET Quantity_initial_EJ     = (SELECT Quantity_initial_EJ     FROM tmp_init WHERE tmp_init."Unit ID"=f."Unit ID"),
            _imputed_flag_quantity  = (SELECT _imputed_flag_quantity  FROM tmp_init WHERE tmp_init."Unit ID"=f."Unit ID"),
            _imputed_by_quantity    = (SELECT _imputed_by_quantity    FROM tmp_init WHERE tmp_init."Unit ID"=f."Unit ID"),
            _imputation_conf_quantity = (SELECT _imputation_conf_quantity FROM tmp_init WHERE tmp_init."Unit ID"=f."Unit ID")
        WHERE f."Unit ID" IN (SELECT "Unit ID" FROM tmp_init)
          AND f.Quantity_initial_EJ IS NULL;
        DROP TABLE tmp_init;
    """)
    conn.commit()
    LOG.info("Copied %d observed initial quantities.", len(df))


def logistic_backcast(conn: sqlite3.Connection, pr_df: pd.DataFrame):
    # build lookup tables -----------------------------------------------------
    cur_r = latest_current_reserves(pr_df)
    prod_r = latest_prod_rates(pr_df)

    cur_map = {r["Unit ID"]: r for _, r in cur_r.iterrows()}
    prod_map = {r["Unit ID"]: r for _, r in prod_r.iterrows()}

    # compute global r --------------------------------------------------------
    Rs, Ps = [], []
    for uid in cur_map.keys() & prod_map.keys():
        try:
            # Data is already in EJ from aggregation
            R = cur_map[uid]["Quantity (converted)"]
            P = prod_map[uid]["Quantity (converted)"]
            if R and P and R > 0 and P > 0:
                Rs.append(R)
                Ps.append(P)
        except Exception as e:
            LOG.warning(f"Failed to get R/P for {uid}: {e}")
    global_r = float(np.median(np.array(Ps) / np.array(Rs))) if Rs else 0.05
    LOG.info("Global logistic r = %.4f  (from %d field pairs)", global_r, len(Rs))

    # country-level R/P -------------------------------------------------------
    rp_by_country = {}
    for uid in cur_map.keys() & prod_map.keys():
        try:
            country = cur_map[uid]["Country/Area"]
            # Data is already in EJ from aggregation
            R = cur_map[uid]["Quantity (converted)"]
            P = prod_map[uid]["Quantity (converted)"]
            if R and P and R > 0 and P > 0:
                rp_by_country.setdefault(country, []).append(P / R)
        except Exception as e:
            LOG.warning(f"Failed to get country R/P for {uid}: {e}")
    rp_by_country = {k: float(np.median(v)) for k, v in rp_by_country.items()}

    # fields still missing ----------------------------------------------------
    fields = pd.read_sql("SELECT * FROM Oil_Gas_fields", conn)
    mask = fields["Quantity_initial_EJ"].isna()
    pending = fields[mask]

    updates = []
    for _, row in pending.iterrows():
        uid = row["Unit ID"]
        Rrow = cur_map.get(uid)
        if not Rrow:
            continue
        # Data is already in EJ from aggregation
        R = Rrow["Quantity (converted)"]
        if not R or R <= 0:
            continue

        Prow = prod_map.get(uid)
        if Prow:
            # Data is already in EJ/y from aggregation
            P = Prow["Quantity (converted)"]
        else:
            rp = rp_by_country.get(row["Country/Area"])
            P = R * rp if rp else None

        if not P or P <= 0:
            K = R / 0.95
            method, conf = "R_only_fallback", 0.15
        elif global_r * R <= P:
            K = R / 0.95
            method, conf = "rate_gt_rR_fallback", 0.15
        else:
            K = global_r * R * R / (global_r * R - P)
            method, conf = "logistic_backcast", 0.35

        updates.append(
            dict(uid=uid, K=K, flag=1, by=method, conf=conf)
        )

    if not updates:
        LOG.info("No quantities imputed via logistic path.")
        return

    upd = pd.DataFrame(updates)
    upd.to_sql("tmp_qty", conn, if_exists="replace", index=False)
    conn.executescript("""
        UPDATE Oil_Gas_fields AS f
        SET Quantity_initial_EJ       = (SELECT K    FROM tmp_qty WHERE tmp_qty.uid=f."Unit ID"),
            _imputed_flag_quantity    = (SELECT flag FROM tmp_qty WHERE tmp_qty.uid=f."Unit ID"),
            _imputed_by_quantity      = (SELECT by   FROM tmp_qty WHERE tmp_qty.uid=f."Unit ID"),
            _imputation_conf_quantity = (SELECT conf FROM tmp_qty WHERE tmp_qty.uid=f."Unit ID")
        WHERE f."Unit ID" IN (SELECT uid FROM tmp_qty)
          AND f.Quantity_initial_EJ IS NULL;
        DROP TABLE tmp_qty;
    """)
    conn.commit()
    LOG.info("Imputed initial quantities for %d fields.", len(updates))


# ────────────────────────── main ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="Energy.db", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        filename="impute_oil_gas_db.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    LOG.info("=== impute_oil_gas_db.py (v2) ===")

    with sqlite3.connect(str(args.db)) as conn:
        conn.row_factory = sqlite3.Row
        add_missing_cols(conn)
        copy_and_impute_year(conn)

        pr_df = load_pr_table(conn)
        populate_initial_observed(conn, first_initial_rows(pr_df))
        logistic_backcast(conn, pr_df)

        LOG.info("Finished without errors.")


if __name__ == "__main__":
    main()
