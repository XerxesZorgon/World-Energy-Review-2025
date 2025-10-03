#!/usr/bin/env python
"""
impute_oil_gas_db.py
--------------------
Populate discovery year and initial-quantity columns in Oil_Gas_fields, using
Oil_Gas_Production_Reserves plus world & country production histories.

Add / update these columns (all REAL except *_flag which are INT):

    discovery_year_final
    _imputed_flag_year
    _imputed_by_year             (TEXT)
    _imputation_conf_year        (REAL)

    Quantity_initial_EJ
    _imputed_flag_quantity
    _imputed_by_quantity         (TEXT)
    _imputation_conf_quantity    (REAL)

Exit status 0 on success; non-zero on failure.

Author: John Peach 
eurAIka sciences
Date: August 7, 2025
Version: 0.1
License: MIT
"""

import argparse, logging, sqlite3, re, math, statistics, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

# local dependency ────────────────────────────────────────────────────────────
try:
    from convert_gas_oil_to_EJ import convert_to_ej
except ImportError as e:
    print(f"[WARN] convert_gas_oil_to_EJ not found - EJ conversion will be skipped: {e}")
    convert_to_ej = lambda qty, unit, fuel, ratio=None: {"total_ej": "0.0"}  # fallback

# ── PATHS (mirrors other scripts) ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "Energy.db"

# ─────────────────────────────────────────────────────────────────────────────
INIT_KEYWORDS = re.compile(r"(initial|original|ultimate recovery|EUR)", re.I)
PRODUCTION_UNITS_PATTERN = re.compile(r"/\s*y", re.I)  # “/y”, “/yr”, “/ year”…


def get_connection(db_path: Path = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def add_missing_columns(conn: sqlite3.Connection):
    """Add result + audit columns if absent."""
    needed = {
        "discovery_year_final": "INTEGER",
        "_imputed_flag_year": "INTEGER",
        "_imputed_by_year": "TEXT",
        "_imputation_conf_year": "REAL",
        "Quantity_initial_EJ": "REAL",
        "_imputed_flag_quantity": "INTEGER",
        "_imputed_by_quantity": "TEXT",
        "_imputation_conf_quantity": "REAL",
    }
    cur = conn.cursor()
    cur.execute("PRAGMA table_info('Oil_Gas_fields');")
    existing = {row["name"] for row in cur.fetchall()}
    for col, col_type in needed.items():
        if col not in existing:
            logging.info("Adding column %s", col)
            conn.execute(f"ALTER TABLE Oil_Gas_fields ADD COLUMN {col} {col_type};")
    conn.commit()


# ───────────────────────── YEAR SECTION ──────────────────────────────────────
def copy_present_discovery_years(conn: sqlite3.Connection):
    """Copy already-present discovery years straight across."""
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE Oil_Gas_fields
        SET discovery_year_final = "Discovery year",
            _imputed_flag_year   = 0,
            _imputed_by_year     = 'observed',
            _imputation_conf_year= 1.0
        WHERE "Discovery year" IS NOT NULL
          AND discovery_year_final IS NULL
        """
    )
    logging.info("Copied %d observed discovery years", cur.rowcount)
    conn.commit()


def impute_discovery_years(conn: sqlite3.Connection):
    """Impute discovery years for the remaining records."""
    df = pd.read_sql(
        'SELECT * FROM Oil_Gas_fields',
        conn,
        parse_dates=False,
    )
    # Use correct column name 'discovery_year' (from schema) for original, and 'discovery_year_final' for imputed
    mask_missing = df["discovery_year_final"].isna()
    if not mask_missing.any():
        logging.info("No discovery years need imputation")
        return

    # --- quick feature selection and preprocessing
    # Use correct column names from Oil_Gas_fields
    features_num = ["Production start year", "Status year"]  # present in table
    features_cat = ["Country/Area", "Fuel type"]
    
    # Preprocess numeric features: coerce to numeric, handle missing
    for col in features_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only use features that actually exist and have some non-null values
    available_features = []
    for col in features_num + features_cat:
        if col in df.columns and df[col].notna().sum() > 10:  # at least 10 non-null values
            available_features.append(col)
    
    if len(available_features) == 0:
        logging.warning("No usable features found for discovery year imputation")
        return
    
    # Separate numeric and categorical features
    available_num = [col for col in available_features if col in features_num]
    available_cat = [col for col in available_features if col in features_cat]
    
    X = df.loc[:, available_features].copy()
    y = df["discovery_year_final"]
    
    # Fill missing values in features for imputation
    for col in available_num:
        X[col] = X[col].fillna(X[col].median())
    for col in available_cat:
        X[col] = X[col].fillna('Unknown')
    
    # encode categoricals
    if available_cat:
        col_trans = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", sparse_output=False), available_cat),
            remainder="passthrough",
        )
    else:
        col_trans = "passthrough"  # no categorical features to encode
    
    estimator = HistGradientBoostingRegressor(random_state=0, max_depth=5)
    pipe = Pipeline(
        steps=[
            ("preprocess", col_trans),
            ("regressor", estimator),
        ]
    )

    # Create mask for rows with known discovery years (for training)
    mask_known = df["Discovery year"].notna()
    
    if mask_known.sum() < 10:  # need at least 10 samples to train
        logging.warning("Insufficient training data for discovery year imputation (<%d samples)", mask_known.sum())
        return
    
    # fit on rows where discovery year is known
    X_train = X.loc[mask_known]
    y_train = pd.to_numeric(df.loc[mask_known, "Discovery year"], errors='coerce')
    
    # Remove any remaining NaN values from training data
    train_mask = y_train.notna()
    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    
    if len(X_train) < 5:  # final check
        logging.warning("Too few valid training samples for discovery year imputation")
        return
    
    pipe.fit(X_train, y_train)

    # predict missing discovery years
    X_missing = X.loc[mask_missing]
    if len(X_missing) == 0:
        logging.info("No missing discovery years to impute")
        return
        
    imputed_vals = pipe.predict(X_missing)

    df.loc[mask_missing, "discovery_year_final"] = np.round(imputed_vals).astype(int)
    df.loc[mask_missing, "_imputed_flag_year"] = 1
    df.loc[mask_missing, "_imputed_by_year"] = "HGBRegressor"
    df.loc[mask_missing, "_imputation_conf_year"] = 0.5   # heuristic

    # write back
    df[["Unit ID", "discovery_year_final", "_imputed_flag_year",
        "_imputed_by_year", "_imputation_conf_year"]].to_sql(
        "temp_update", conn, if_exists="replace", index=False
    )
    conn.executescript(
        """
        UPDATE Oil_Gas_fields
        SET discovery_year_final = (
                SELECT discovery_year_final FROM temp_update
                WHERE temp_update."Unit ID" = Oil_Gas_fields."Unit ID"
            ),
            _imputed_flag_year = (
                SELECT _imputed_flag_year FROM temp_update
                WHERE temp_update."Unit ID" = Oil_Gas_fields."Unit ID"
            ),
            _imputed_by_year = (
                SELECT _imputed_by_year FROM temp_update
                WHERE temp_update."Unit ID" = Oil_Gas_fields."Unit ID"
            ),
            _imputation_conf_year = (
                SELECT _imputation_conf_year FROM temp_update
                WHERE temp_update."Unit ID" = Oil_Gas_fields."Unit ID"
            )
        WHERE "Unit ID" IN (SELECT "Unit ID" FROM temp_update);
        DROP TABLE temp_update;
        """
    )
    conn.commit()
    logging.info("Imputed discovery years for %d rows", mask_missing.sum())


# ───────────────────────── QUANTITY SECTION ──────────────────────────────────
def select_initial_rows(pr_df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with initial/original classifications."""
    return pr_df[pr_df["Reserves classification (original)"].str.contains(
        INIT_KEYWORDS, na=False
    ) & pr_df["Quantity (converted)"].notna()]


def to_EJ(row) -> float | None:
    try:
        result = convert_to_ej(
            quantity=row["Quantity (converted)"],
            unit=row["Units (converted)"],
            fuel_type=row["Fuel description"],
        )
        # Handle dictionary return values
        if isinstance(result, dict):
            return float(result.get('total_ej', 0))
        return float(result) if result is not None else None
    except Exception as exc:  # unsupported unit etc.
        warnings.warn(f"Conversion failed for Unit ID={row['Unit ID']}: {exc}")
        return None


def populate_initial_from_pr(conn: sqlite3.Connection):
    pr_df = pd.read_sql(
        'SELECT * FROM Oil_Gas_Production_Reserves',
        conn,
        parse_dates=False,
    )  # columns: unit_id, unit_name, country/area, production/reserves, wiki_url, fuel_description, reserves_classification_(original), quantity_(original), units_(original), data_year, quantity_(converted), units_(converted)
    init_df = select_initial_rows(pr_df).copy()
    if init_df.empty:
        logging.info("No rows classify as initial/original in Oil_Gas_Production_Reserves")
        return

    # Map to correct columns for join/update
    init_df["Quantity_initial_EJ"] = init_df.apply(to_EJ, axis=1)
    keep_cols = ["Unit ID", "Quantity_initial_EJ"]
    init_df = init_df[keep_cols].dropna(subset=["Quantity_initial_EJ"]) # 'Unit ID' is the join key

    # keep earliest Data year per Unit ID
    init_df = init_df.groupby("Unit ID", as_index=False)["Quantity_initial_EJ"].first()

    init_df["_imputed_flag_quantity"] = 0
    init_df["_imputed_by_quantity"] = "observed_initial_PR_table"
    init_df["_imputation_conf_quantity"] = 1.0

    init_df.to_sql("temp_q_init", conn, if_exists="replace", index=False)
    conn.executescript(
        """
        UPDATE Oil_Gas_fields
        SET Quantity_initial_EJ = (
                SELECT Quantity_initial_EJ FROM temp_q_init
                WHERE temp_q_init."Unit ID" = Oil_Gas_fields."Unit ID"
            ),
            _imputed_flag_quantity = (
                SELECT _imputed_flag_quantity FROM temp_q_init
                WHERE temp_q_init."Unit ID" = Oil_Gas_fields."Unit ID"
            ),
            _imputed_by_quantity = (
                SELECT _imputed_by_quantity FROM temp_q_init
                WHERE temp_q_init."Unit ID" = Oil_Gas_fields."Unit ID"
            ),
            _imputation_conf_quantity = (
                SELECT _imputation_conf_quantity FROM temp_q_init
                WHERE temp_q_init."Unit ID" = Oil_Gas_fields."Unit ID"
            )
        WHERE "Unit ID" IN (SELECT "Unit ID" FROM temp_q_init)
          AND Quantity_initial_EJ IS NULL;  -- keep any earlier values
        DROP TABLE temp_q_init;
        """
    )
    conn.commit()
    logging.info("Copied %d initial quantities from PR table", len(init_df))


# ───────────── logistic-based back-cast for remaining fields ────────────────
def world_r_by_fuel(conn: sqlite3.Connection, fuel: str) -> float:
    """
    Estimate logistic parameter r for a given fuel (oil|gas) from world
    production & reserves tables.  r ≈ median(P / R)
    """
    if fuel == "oil":
        prod_table = "EI_oil_production"
        res_table = "EI_oil_proved_reserves"
    else:
        prod_table = "EI_gas_production_EJ"  # already converted to EJ
        res_table = "EI_gas_proved_reserves"

    # EI tables have wide format: Country, 1965, 1966, ..., 2024
    # Get "Total World" row and calculate P/R ratios across years
    try:
        prod = pd.read_sql(f'SELECT * FROM {prod_table} WHERE Country = "Total World"', conn)
        resv = pd.read_sql(f'SELECT * FROM {res_table} WHERE Country = "Total World"', conn)
        
        if prod.empty or resv.empty:
            logging.warning(f"No Total World data found in {prod_table} or {res_table}")
            return 0.1  # fallback default
        
        # Get year columns (exclude 'Country')
        year_cols = [col for col in prod.columns if col != 'Country' and col.isdigit()]
        
        ratios = []
        for year in year_cols:
            if year in prod.columns and year in resv.columns:
                p_val = pd.to_numeric(prod[year].iloc[0], errors='coerce')
                r_val = pd.to_numeric(resv[year].iloc[0], errors='coerce')
                if pd.notna(p_val) and pd.notna(r_val) and r_val > 0:
                    ratios.append(p_val / r_val)
        
        if ratios:
            r = float(pd.Series(ratios).median())
            return r
        else:
            logging.warning(f"No valid P/R ratios found for {fuel}")
            return 0.1  # fallback default
            
    except Exception as e:
        logging.warning(f"Error calculating world R by fuel for {fuel}: {e}")
        return 0.1  # fallback default


def country_RP_ratio(conn: sqlite3.Connection, fuel: str) -> pd.DataFrame:
    """Return dataframe with Country/Area and typical R/P ratio in 1/years."""
    if fuel == "oil":
        prod_table = "EI_oil_production"
        res_table = "EI_oil_proved_reserves"
    else:
        prod_table = "EI_gas_production_EJ"
        res_table = "EI_gas_proved_reserves"

    try:
        prod = pd.read_sql(f'SELECT * FROM {prod_table}', conn)
        resv = pd.read_sql(f'SELECT * FROM {res_table}', conn)
        
        # Get year columns (exclude 'Country')
        year_cols = [col for col in prod.columns if col != 'Country' and col.isdigit()]
        
        ratios = {}
        for _, prod_row in prod.iterrows():
            country = prod_row['Country']
            if country in resv['Country'].values:
                resv_row = resv[resv['Country'] == country].iloc[0]
                
                # Calculate P/R ratios across years for this country
                country_ratios = []
                for year in year_cols:
                    if year in prod.columns and year in resv.columns:
                        p_val = pd.to_numeric(prod_row[year], errors='coerce')
                        r_val = pd.to_numeric(resv_row[year], errors='coerce')
                        if pd.notna(p_val) and pd.notna(r_val) and r_val > 0:
                            country_ratios.append(p_val / r_val)
                
                if country_ratios:
                    ratios[country] = pd.Series(country_ratios).median()
        
        rp_df = pd.DataFrame({"Country/Area": ratios.keys(), "RP_ratio": ratios.values()})
        return rp_df
        
    except Exception as e:
        logging.warning(f"Error calculating country R/P ratios for {fuel}: {e}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=["Country/Area", "RP_ratio"])


def impute_remaining_quantities(conn: sqlite3.Connection):
    """Logistic back-cast for fields still missing Quantity_initial_EJ."""
    fields_df = pd.read_sql('SELECT * FROM Oil_Gas_fields', conn)
    mask_missing = fields_df["Quantity_initial_EJ"].isna()
    if not mask_missing.any():
        logging.info("No initial quantities need imputation")
        return

    logging.info(f"Attempting to impute quantities for {mask_missing.sum()} fields")

    # Get reserves data from PR table
    reserves_df = pd.read_sql(
        'SELECT * FROM Oil_Gas_Production_Reserves WHERE "Production/reserves" = "reserves"', 
        conn
    )
    reserves_df = reserves_df[reserves_df["Quantity (converted)"].notna()]
    
    # Convert reserves to EJ and create lookup
    reserves_lookup = {}  # Unit ID -> reserves_EJ
    for _, row in reserves_df.iterrows():
        uid = row["Unit ID"]
        try:
            result = convert_to_ej(
                quantity=row["Quantity (converted)"],
                unit=row["Units (converted)"],
                fuel_type=row["Fuel description"]
            )
            if isinstance(result, dict):
                reserves_ej = float(result.get('total_ej', 0))
            else:
                reserves_ej = float(result) if result is not None else 0
            
            if reserves_ej > 0:
                # Keep the largest reserves value per unit
                if uid not in reserves_lookup or reserves_ej > reserves_lookup[uid]:
                    reserves_lookup[uid] = reserves_ej
        except Exception as e:
            continue
    
    logging.info(f"Found reserves data for {len(reserves_lookup)} units")

    # Get production rate data from PR table
    prod_df = pd.read_sql('SELECT * FROM Oil_Gas_Production_Reserves', conn)
    prod_df = prod_df[prod_df["Quantity (converted)"].notna()]
    # Filter for production rates (units containing /y)
    prod_df = prod_df[prod_df["Units (converted)"].str.contains(PRODUCTION_UNITS_PATTERN, na=False)]

    # pre-compute global r and country R/P
    r_oil = world_r_by_fuel(conn, "oil")
    r_gas = world_r_by_fuel(conn, "gas")
    rp_oil = country_RP_ratio(conn, "oil")
    rp_gas = country_RP_ratio(conn, "gas")

    # build production lookup
    field_to_prod = {}  # Unit ID -> (P_EJ_per_y, fuel, data_year)
    for _, row in prod_df.iterrows():
        uid = row["Unit ID"]
        try:
            result = convert_to_ej(
                quantity=row["Quantity (converted)"],
                unit=row["Units (converted)"],
                fuel_type=row["Fuel description"]
            )
            if isinstance(result, dict):
                P_ej = float(result.get('total_ej', 0))
            else:
                P_ej = float(result) if result is not None else 0
        except Exception:
            continue
        # keep the most recent production rate
        if uid not in field_to_prod or row["Data year"] > field_to_prod[uid][2]:
            field_to_prod[uid] = (P_ej, row["Fuel description"], row["Data year"])

    logging.info(f"Found production data for {len(field_to_prod)} units")

    updates = []
    for idx, row in fields_df.loc[mask_missing].iterrows():
        uid = row["Unit ID"]
        fuel = "gas" if "gas" in row["Fuel type"].lower() else "oil"

        # Get reserves from our lookup
        R = reserves_lookup.get(uid)
        if not R or R <= 0:
            continue

        # choose r and get production rate
        r = r_gas if fuel == "gas" else r_oil
        P = None
        if uid in field_to_prod:
            P = field_to_prod[uid][0]
        
        if not P or math.isnan(P) or P <= 0:
            # try country-based R/P ratio
            country_ratio = (rp_gas if fuel == "gas" else rp_oil)
            if not country_ratio.empty:
                rp_row = country_ratio[country_ratio["Country/Area"] == row["Country/Area"]]
                if not rp_row.empty and not pd.isna(rp_row.iloc[0]["RP_ratio"]):
                    P = R * rp_row.iloc[0]["RP_ratio"]

        if not P or P <= 0 or r * R <= P:
            continue  # cannot compute

        # logistic back-cast: K = r*R^2 / (r*R - P)
        K = r * R * R / (r * R - P)
        updates.append(
            dict(
                uid=uid,
                q_init=K,
                imp_flag=1,
                imp_by="logistic_backcast",
                conf=0.3,
            )
        )
    
    logging.info(f"Generated {len(updates)} quantity imputations")

    if not updates:
        logging.info("No quantities could be imputed via logistic back-cast")
        return

    upd_df = pd.DataFrame(updates)
    upd_df.to_sql("temp_q_imp", conn, if_exists="replace", index=False)

    conn.executescript(
        """
        UPDATE Oil_Gas_fields
        SET Quantity_initial_EJ = (
                SELECT q_init FROM temp_q_imp
                WHERE temp_q_imp.uid = Oil_Gas_fields."Unit ID"
            ),
            _imputed_flag_quantity = (
                SELECT imp_flag FROM temp_q_imp
                WHERE temp_q_imp.uid = Oil_Gas_fields."Unit ID"
            ),
            _imputed_by_quantity = (
                SELECT imp_by FROM temp_q_imp
                WHERE temp_q_imp.uid = Oil_Gas_fields."Unit ID"
            ),
            _imputation_conf_quantity = (
                SELECT conf FROM temp_q_imp
                WHERE temp_q_imp.uid = Oil_Gas_fields."Unit ID"
            )
        WHERE "Unit ID" IN (SELECT uid FROM temp_q_imp)
          AND Quantity_initial_EJ IS NULL;
        DROP TABLE temp_q_imp;
        """
    )
    conn.commit()
    logging.info("Imputed initial quantities for %d fields", len(updates))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Impute discovery year and initial quantities for Oil/Gas fields."
    )
    parser.add_argument("--db", default=DB_PATH, type=Path, help="Path to SQLite db")
    args = parser.parse_args()
    
    if not args.db.exists():
        print(f"[ERROR] Database not found at {args.db} - run import_batch.py first.")
        raise SystemExit(1)

    logging.basicConfig(
        filename="impute_oil_gas_db.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logging.info("=== impute_oil_gas_db.py started ===")

    try:
        conn = get_connection(args.db)
        add_missing_columns(conn)
        copy_present_discovery_years(conn)
        impute_discovery_years(conn)
        populate_initial_from_pr(conn)
        impute_remaining_quantities(conn)
        logging.info("All tasks completed successfully")
    except Exception as exc:
        logging.exception("Failed: %s", exc)
        raise SystemExit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
