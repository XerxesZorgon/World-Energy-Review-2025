#!/usr/bin/env python
"""
impute_oil_gas_100_percent.py
Comprehensive multi-tier strategy to achieve 100% quantity coverage
"""

import argparse, logging, sqlite3, re, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

from convert_gas_oil_to_EJ import convert_to_ej

LOG = logging.getLogger(__name__)

def to_EJ(volume: float, unit: str, fuel: str) -> float | None:
    """Convert any volume unit to EJ using supplied helper."""
    try:
        unit_clean = re.sub(r"(/y|/d|\bbpd\b)", "", str(unit)).strip()
        result = convert_to_ej(volume, unit_clean, fuel_type=fuel)
        if isinstance(result, dict) and 'total_ej' in result:
            return float(result['total_ej'])
        return None
    except Exception:
        return None

def add_missing_cols(conn: sqlite3.Connection):
    """Add imputation metadata columns if they don't exist."""
    cols = {
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

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: DIRECT COPY FROM INITIAL/ULTIMATE RESERVES
# ═══════════════════════════════════════════════════════════════════════════════

def tier1_direct_initial(conn: sqlite3.Connection):
    """Copy direct initial/ultimate reserves from PR table."""
    LOG.info("=== TIER 1: Direct Initial Reserves ===")
    
    # Get initial/ultimate reserves and aggregate by Unit ID
    initial_reserves = pd.read_sql("""
        SELECT pr.[Unit ID], pr.[Fuel description], pr.[Quantity (converted)], pr.[Units (converted)]
        FROM Oil_Gas_Production_Reserves pr
        WHERE pr.[Production/reserves] = 'reserves'
        AND pr.[Reserves classification (original)] LIKE '%initial%'
        OR pr.[Reserves classification (original)] LIKE '%ultimate%'
        OR pr.[Reserves classification (original)] LIKE '%EUR%'
        AND pr.[Quantity (converted)] IS NOT NULL
    """, conn)
    
    # Aggregate by Unit ID
    unit_totals = {}
    for _, row in initial_reserves.iterrows():
        try:
            ej = to_EJ(row['Quantity (converted)'], row['Units (converted)'], row['Fuel description'])
            if ej and ej > 0:
                uid = row['Unit ID']
                unit_totals[uid] = unit_totals.get(uid, 0) + ej
        except Exception as e:
            LOG.warning(f"Failed to convert initial for {row['Unit ID']}: {e}")
    
    if unit_totals:
        updates = pd.DataFrame([
            {'uid': uid, 'qty': qty, 'flag': 0, 'by': 'direct_initial', 'conf': 1.0}
            for uid, qty in unit_totals.items()
        ])
        
        updates.to_sql('tmp_tier1', conn, if_exists='replace', index=False)
        conn.executescript("""
            UPDATE Oil_Gas_fields AS f
            SET Quantity_initial_EJ = (SELECT qty FROM tmp_tier1 WHERE tmp_tier1.uid = f.[Unit ID]),
                _imputed_flag_quantity = (SELECT flag FROM tmp_tier1 WHERE tmp_tier1.uid = f.[Unit ID]),
                _imputed_by_quantity = (SELECT by FROM tmp_tier1 WHERE tmp_tier1.uid = f.[Unit ID]),
                _imputation_conf_quantity = (SELECT conf FROM tmp_tier1 WHERE tmp_tier1.uid = f.[Unit ID])
            WHERE f.[Unit ID] IN (SELECT uid FROM tmp_tier1)
            AND f.Quantity_initial_EJ IS NULL;
            DROP TABLE tmp_tier1;
        """)
        conn.commit()
        LOG.info(f"TIER 1: Populated {len(unit_totals)} fields with direct initial reserves")

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: CURRENT RESERVES + CUMULATIVE PRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def tier2_reserves_plus_production(conn: sqlite3.Connection):
    """Initial = Current Reserves + Cumulative Production."""
    LOG.info("=== TIER 2: Reserves + Cumulative Production ===")
    
    # Get current reserves (aggregated)
    reserves = pd.read_sql("""
        SELECT pr.[Unit ID], pr.[Fuel description], pr.[Quantity (converted)], pr.[Units (converted)]
        FROM Oil_Gas_Production_Reserves pr
        WHERE pr.[Production/reserves] = 'reserves'
        AND pr.[Reserves classification (original)] NOT LIKE '%initial%'
        AND pr.[Reserves classification (original)] NOT LIKE '%ultimate%'
        AND pr.[Quantity (converted)] IS NOT NULL
    """, conn)
    
    # Get cumulative production (aggregated)
    production = pd.read_sql("""
        SELECT pr.[Unit ID], pr.[Fuel description], pr.[Quantity (converted)], pr.[Units (converted)]
        FROM Oil_Gas_Production_Reserves pr
        WHERE pr.[Production/reserves] = 'production'
        AND pr.[Quantity (converted)] IS NOT NULL
    """, conn)
    
    # Aggregate reserves by Unit ID
    reserves_totals = {}
    for _, row in reserves.iterrows():
        try:
            ej = to_EJ(row['Quantity (converted)'], row['Units (converted)'], row['Fuel description'])
            if ej and ej > 0:
                uid = row['Unit ID']
                reserves_totals[uid] = reserves_totals.get(uid, 0) + ej
        except Exception:
            pass
    
    # Aggregate production by Unit ID
    production_totals = {}
    for _, row in production.iterrows():
        try:
            ej = to_EJ(row['Quantity (converted)'], row['Units (converted)'], row['Fuel description'])
            if ej and ej > 0:
                uid = row['Unit ID']
                production_totals[uid] = production_totals.get(uid, 0) + ej
        except Exception:
            pass
    
    # Calculate Initial = Reserves + Production
    updates = []
    for uid in reserves_totals.keys():
        R = reserves_totals[uid]
        P = production_totals.get(uid, 0)  # Default to 0 if no production
        initial = R + P
        if initial > 0:
            updates.append({
                'uid': uid, 'qty': initial, 'flag': 1, 
                'by': f'reserves_plus_prod(R={R:.3f},P={P:.3f})', 'conf': 0.8
            })
    
    if updates:
        upd_df = pd.DataFrame(updates)
        upd_df.to_sql('tmp_tier2', conn, if_exists='replace', index=False)
        conn.executescript("""
            UPDATE Oil_Gas_fields AS f
            SET Quantity_initial_EJ = (SELECT qty FROM tmp_tier2 WHERE tmp_tier2.uid = f.[Unit ID]),
                _imputed_flag_quantity = (SELECT flag FROM tmp_tier2 WHERE tmp_tier2.uid = f.[Unit ID]),
                _imputed_by_quantity = (SELECT by FROM tmp_tier2 WHERE tmp_tier2.uid = f.[Unit ID]),
                _imputation_conf_quantity = (SELECT conf FROM tmp_tier2 WHERE tmp_tier2.uid = f.[Unit ID])
            WHERE f.[Unit ID] IN (SELECT uid FROM tmp_tier2)
            AND f.Quantity_initial_EJ IS NULL;
            DROP TABLE tmp_tier2;
        """)
        conn.commit()
        LOG.info(f"TIER 2: Populated {len(updates)} fields with reserves + production")

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: STATISTICAL IMPUTATION BY COUNTRY/FUEL TYPE
# ═══════════════════════════════════════════════════════════════════════════════

def tier3_statistical_imputation(conn: sqlite3.Connection):
    """Use country/fuel type medians for statistical imputation."""
    LOG.info("=== TIER 3: Statistical Imputation ===")
    
    # Get all populated quantities for statistical analysis
    populated = pd.read_sql("""
        SELECT [Country/Area], [Fuel type], [Status], Quantity_initial_EJ
        FROM Oil_Gas_fields
        WHERE Quantity_initial_EJ IS NOT NULL
    """, conn)
    
    # Get missing fields
    missing = pd.read_sql("""
        SELECT [Unit ID], [Country/Area], [Fuel type], [Status]
        FROM Oil_Gas_fields
        WHERE Quantity_initial_EJ IS NULL
    """, conn)
    
    updates = []
    
    for _, row in missing.iterrows():
        uid = row['Unit ID']
        country = row['Country/Area']
        fuel = row['Fuel type']
        status = row['Status']
        
        # Try country + fuel type median
        subset = populated[(populated['Country/Area'] == country) & 
                          (populated['Fuel type'] == fuel)]
        
        if len(subset) >= 3:  # Need at least 3 samples
            median_qty = subset['Quantity_initial_EJ'].median()
            updates.append({
                'uid': uid, 'qty': median_qty, 'flag': 1,
                'by': f'country_fuel_median({country},{fuel})', 'conf': 0.4
            })
            continue
        
        # Fallback: country median (any fuel type)
        subset = populated[populated['Country/Area'] == country]
        if len(subset) >= 3:
            median_qty = subset['Quantity_initial_EJ'].median()
            updates.append({
                'uid': uid, 'qty': median_qty, 'flag': 1,
                'by': f'country_median({country})', 'conf': 0.3
            })
            continue
        
        # Fallback: fuel type median (any country)
        subset = populated[populated['Fuel type'] == fuel]
        if len(subset) >= 3:
            median_qty = subset['Quantity_initial_EJ'].median()
            updates.append({
                'uid': uid, 'qty': median_qty, 'flag': 1,
                'by': f'fuel_median({fuel})', 'conf': 0.25
            })
            continue
        
        # Final fallback: global median
        global_median = populated['Quantity_initial_EJ'].median()
        updates.append({
            'uid': uid, 'qty': global_median, 'flag': 1,
            'by': 'global_median', 'conf': 0.1
        })
    
    if updates:
        upd_df = pd.DataFrame(updates)
        upd_df.to_sql('tmp_tier3', conn, if_exists='replace', index=False)
        conn.executescript("""
            UPDATE Oil_Gas_fields AS f
            SET Quantity_initial_EJ = (SELECT qty FROM tmp_tier3 WHERE tmp_tier3.uid = f.[Unit ID]),
                _imputed_flag_quantity = (SELECT flag FROM tmp_tier3 WHERE tmp_tier3.uid = f.[Unit ID]),
                _imputed_by_quantity = (SELECT by FROM tmp_tier3 WHERE tmp_tier3.uid = f.[Unit ID]),
                _imputation_conf_quantity = (SELECT conf FROM tmp_tier3 WHERE tmp_tier3.uid = f.[Unit ID])
            WHERE f.[Unit ID] IN (SELECT uid FROM tmp_tier3)
            AND f.Quantity_initial_EJ IS NULL;
            DROP TABLE tmp_tier3;
        """)
        conn.commit()
        LOG.info(f"TIER 3: Populated {len(updates)} fields with statistical imputation")

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4: MACHINE LEARNING IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def tier4_ml_imputation(conn: sqlite3.Connection):
    """Use machine learning to predict missing quantities."""
    LOG.info("=== TIER 4: Machine Learning Imputation ===")
    
    # Get all fields with features
    df = pd.read_sql("""
        SELECT [Unit ID], [Country/Area], [Fuel type], [Status], [Basin],
               [Production start year], [FID Year], [Latitude], [Longitude],
               Quantity_initial_EJ
        FROM Oil_Gas_fields
    """, conn)
    
    # Prepare features
    feature_cols = ['Country/Area', 'Fuel type', 'Status', 'Basin']
    numeric_cols = ['Production start year', 'FID Year', 'Latitude', 'Longitude']
    
    # Clean numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns
    for col in feature_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Split into training and prediction sets
    train_mask = df['Quantity_initial_EJ'].notna()
    train_df = df[train_mask]
    predict_df = df[~train_mask]
    
    if len(train_df) < 100 or len(predict_df) == 0:
        LOG.warning("Insufficient data for ML imputation")
        return
    
    # Prepare features and target
    X_train = train_df[feature_cols + numeric_cols]
    y_train = train_df['Quantity_initial_EJ']
    X_predict = predict_df[feature_cols + numeric_cols]
    
    # Create preprocessing pipeline
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), feature_cols),
        remainder='passthrough'
    )
    
    # Create ML pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', HistGradientBoostingRegressor(random_state=42, max_depth=8))
    ])
    
    try:
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predict missing values
        predictions = pipeline.predict(X_predict)
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.001)  # Minimum 0.001 EJ
        
        # Create updates
        updates = []
        for i, (_, row) in enumerate(predict_df.iterrows()):
            updates.append({
                'uid': row['Unit ID'],
                'qty': predictions[i],
                'flag': 1,
                'by': 'ml_gradient_boosting',
                'conf': 0.2
            })
        
        if updates:
            upd_df = pd.DataFrame(updates)
            upd_df.to_sql('tmp_tier4', conn, if_exists='replace', index=False)
            conn.executescript("""
                UPDATE Oil_Gas_fields AS f
                SET Quantity_initial_EJ = (SELECT qty FROM tmp_tier4 WHERE tmp_tier4.uid = f.[Unit ID]),
                    _imputed_flag_quantity = (SELECT flag FROM tmp_tier4 WHERE tmp_tier4.uid = f.[Unit ID]),
                    _imputed_by_quantity = (SELECT by FROM tmp_tier4 WHERE tmp_tier4.uid = f.[Unit ID]),
                    _imputation_conf_quantity = (SELECT conf FROM tmp_tier4 WHERE tmp_tier4.uid = f.[Unit ID])
                WHERE f.[Unit ID] IN (SELECT uid FROM tmp_tier4)
                AND f.Quantity_initial_EJ IS NULL;
                DROP TABLE tmp_tier4;
            """)
            conn.commit()
            LOG.info(f"TIER 4: Populated {len(updates)} fields with ML imputation")
    
    except Exception as e:
        LOG.error(f"ML imputation failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/Energy.db", type=Path)
    args = parser.parse_args()
    
    logging.basicConfig(
        filename="impute_oil_gas_100_percent.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    LOG.info("=== 100% Coverage Imputation Strategy ===")
    
    with sqlite3.connect(str(args.db)) as conn:
        add_missing_cols(conn)
        
        # Execute all tiers
        tier1_direct_initial(conn)
        tier2_reserves_plus_production(conn)
        tier3_statistical_imputation(conn)
        tier4_ml_imputation(conn)
        
        # Final coverage report
        stats = pd.read_sql("""
            SELECT 
                COUNT(*) as total,
                COUNT(Quantity_initial_EJ) as populated,
                SUM(_imputed_flag_quantity) as imputed
            FROM Oil_Gas_fields
        """, conn)
        
        total = stats.iloc[0]['total']
        populated = stats.iloc[0]['populated']
        imputed = stats.iloc[0]['imputed']
        
        LOG.info(f"FINAL RESULTS: {populated}/{total} fields populated ({populated/total*100:.1f}%)")
        LOG.info(f"Total imputed: {imputed}")
        print(f"FINAL COVERAGE: {populated}/{total} fields ({populated/total*100:.1f}%)")

if __name__ == "__main__":
    main()
