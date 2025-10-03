#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
impute_oil.py
Impute missing initial field sizes (Quantity_initial_EJ) for OIL only.

Tiers:
  1) Direct "initial/ultimate/EUR" reserves from Oil_Gas_Production_Reserves
  2) Initial = (current reserves) + (cumulative production), by Unit ID
  3) Statistical medians (Country×Status → Country → Status → Global)
  4) ML fallback (HistGradientBoostingRegressor)

Writes into table: Oil_fields

Usage:
  python impute_oil.py --db path/to/Energy.db
"""

import argparse, logging, sqlite3, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

# Helper used in your combined script
from convert_gas_oil_to_EJ import convert_to_ej

LOG = logging.getLogger("impute_oil")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def to_EJ(volume, unit, fuel="oil"):
    """Convert any unit to EJ using your helper. Returns float or None."""
    try:
        unit_clean = re.sub(r"(/y|/d|\bbpd\b)", "", str(unit)).strip()
        out = convert_to_ej(volume, unit_clean, fuel_type=fuel)
        if isinstance(out, dict) and "total_ej" in out:
            return float(out["total_ej"])
        if isinstance(out, (int, float)) and np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return None


def add_missing_cols(conn: sqlite3.Connection, table: str):
    """Add imputation metadata columns if they don't exist."""
    want = {
        "Quantity_initial_EJ": "REAL",
        "_imputed_flag_quantity": "INTEGER",
        "_imputed_by_quantity": "TEXT",
        "_imputation_conf_quantity": "REAL",
    }
    have = {r[1] for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
    for col, typ in want.items():
        if col not in have:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {typ}')
    conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: Direct initial/ultimate/EUR from P/R table
# ──────────────────────────────────────────────────────────────────────────────

def tier1_direct_initial(conn: sqlite3.Connection, table: str):
    LOG.info("TIER 1 (oil): direct initial/ultimate/EUR")

    sql = """
    SELECT pr.unit_id, pr.fuel_description, pr.[quantity_(converted)] AS qty, pr.[units_(converted)] AS units
    FROM Oil_Gas_Production_Reserves pr
    WHERE pr.[production/reserves] = 'reserves'
      AND LOWER(pr.fuel_description) LIKE '%oil%'
      AND (
            LOWER(pr.[reserves_classification_(original)]) LIKE '%initial%'
         OR LOWER(pr.[reserves_classification_(original)]) LIKE '%ultimate%'
         OR LOWER(pr.[reserves_classification_(original)]) LIKE '%eur%'
      )
      AND pr.[quantity_(converted)] IS NOT NULL
    """
    df = pd.read_sql(sql, conn)
    if df.empty:
        return 0

    df["ej"] = df.apply(lambda r: to_EJ(r["qty"], r["units"], "oil"), axis=1)
    df = df[pd.to_numeric(df["ej"], errors="coerce").notna() & (df["ej"] > 0)]
    agg = df.groupby("unit_id", as_index=False)["ej"].sum()

    if agg.empty: 
        return 0

    agg.rename(columns={"ej": "qty"}, inplace=True)
    agg["flag"] = 0
    agg["by"] = "direct_initial"
    agg["conf"] = 1.0
    agg.to_sql("tmp_tier1_oil", conn, if_exists="replace", index=False)

    conn.executescript(f"""
        UPDATE "{table}" AS f
        SET Quantity_initial_EJ        = (SELECT qty  FROM tmp_tier1_oil t WHERE t.unit_id = f.unit_id),
            _imputed_flag_quantity     = (SELECT flag FROM tmp_tier1_oil t WHERE t.unit_id = f.unit_id),
            _imputed_by_quantity       = (SELECT by   FROM tmp_tier1_oil t WHERE t.unit_id = f.unit_id),
            _imputation_conf_quantity  = (SELECT conf FROM tmp_tier1_oil t WHERE t.unit_id = f.unit_id)
        WHERE f.unit_id IN (SELECT unit_id FROM tmp_tier1_oil)
          AND (f.Quantity_initial_EJ IS NULL OR f.Quantity_initial_EJ <= 0);
        DROP TABLE tmp_tier1_oil;
    """)
    conn.commit()
    return len(agg)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2: Initial = current reserves + cumulative production
# ──────────────────────────────────────────────────────────────────────────────

def tier2_reserves_plus_prod(conn: sqlite3.Connection, table: str):
    LOG.info("TIER 2 (oil): reserves + cumulative production")

    # Current reserves
    reserves = pd.read_sql("""
        SELECT unit_id, [quantity_(converted)] AS qty, [units_(converted)] AS units
        FROM Oil_Gas_Production_Reserves
        WHERE [production/reserves] = 'reserves'
          AND LOWER(fuel_description) LIKE '%oil%'
          AND [quantity_(converted)] IS NOT NULL
    """, conn)

    # Cumulative production
    production = pd.read_sql("""
        SELECT unit_id, [quantity_(converted)] AS qty, [units_(converted)] AS units
        FROM Oil_Gas_Production_Reserves
        WHERE [production/reserves] = 'production'
          AND LOWER(fuel_description) LIKE '%oil%'
          AND [quantity_(converted)] IS NOT NULL
    """, conn)

    if reserves.empty and production.empty:
        return 0

    def agg_to_ej(df):
        if df.empty: 
            return {}
        df["ej"] = df.apply(lambda r: to_EJ(r["qty"], r["units"], "oil"), axis=1)
        df = df[pd.to_numeric(df["ej"], errors="coerce").notna() & (df["ej"] > 0)]
        return df.groupby("unit_id")["ej"].sum().to_dict()

    R = agg_to_ej(reserves)
    P = agg_to_ej(production)

    rows = []
    for uid, r_ej in R.items():
        p_ej = float(P.get(uid, 0.0))
        initial = r_ej + p_ej
        if initial > 0:
            rows.append({
                "unit_id": uid,
                "qty": initial,
                "flag": 1,
                "by": f"reserves_plus_prod(R={r_ej:.3f},P={p_ej:.3f})",
                "conf": 0.8
            })
    if not rows:
        return 0

    upd = pd.DataFrame(rows)
    upd.to_sql("tmp_tier2_oil", conn, if_exists="replace", index=False)
    conn.executescript(f"""
        UPDATE "{table}" AS f
        SET Quantity_initial_EJ        = (SELECT qty  FROM tmp_tier2_oil t WHERE t.unit_id = f.unit_id),
            _imputed_flag_quantity     = (SELECT flag FROM tmp_tier2_oil t WHERE t.unit_id = f.unit_id),
            _imputed_by_quantity       = (SELECT by   FROM tmp_tier2_oil t WHERE t.unit_id = f.unit_id),
            _imputation_conf_quantity  = (SELECT conf FROM tmp_tier2_oil t WHERE t.unit_id = f.unit_id)
        WHERE f.unit_id IN (SELECT unit_id FROM tmp_tier2_oil)
          AND (f.Quantity_initial_EJ IS NULL OR f.Quantity_initial_EJ <= 0);
        DROP TABLE tmp_tier2_oil;
    """)
    conn.commit()
    return len(upd)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 3: Statistical medians (Country×Status → Country → Status → Global)
# ──────────────────────────────────────────────────────────────────────────────

def tier3_statistical(conn: sqlite3.Connection, table: str):
    LOG.info("TIER 3 (oil): statistical medians")

    populated = pd.read_sql(f"""
        SELECT [country/area] AS Country, fuel_type, Quantity_initial_EJ
        FROM "{table}"
        WHERE Quantity_initial_EJ IS NOT NULL AND Quantity_initial_EJ > 0
    """, conn)

    missing = pd.read_sql(f"""
        SELECT unit_id, [country/area] AS Country, fuel_type
        FROM "{table}"
        WHERE Quantity_initial_EJ IS NULL OR Quantity_initial_EJ <= 0
    """, conn)

    if populated.empty or missing.empty:
        return 0

    updates = []

    # Standard medians
    med_country_fuel = populated.groupby(["Country", "fuel_type"], dropna=False)["Quantity_initial_EJ"].median()
    med_country      = populated.groupby(["Country"], dropna=False)["Quantity_initial_EJ"].median()
    med_fuel         = populated.groupby(["fuel_type"], dropna=False)["Quantity_initial_EJ"].median()
    med_global       = float(populated["Quantity_initial_EJ"].median())

    # fuel_type is already in missing DataFrame, no need to merge again
    missing["fuel_type"] = missing["fuel_type"].fillna("Unknown")
    
    for _, r in missing.iterrows():
        uid, ctry, fuel = r["unit_id"], r["Country"], r["fuel_type"]
        
        # Try country + fuel type median first
        val = med_country_fuel.get((ctry, fuel), np.nan)
        if pd.notna(val):
            by, conf = f"country_fuel_median({ctry},{fuel})", 0.45
        else:
            # Fallback to country only
            val = med_country.get(ctry, np.nan)
            if pd.notna(val):
                by, conf = f"country_median({ctry})", 0.30
            else:
                # Fallback to fuel type only
                val = med_fuel.get(fuel, np.nan)
                if pd.notna(val):
                    by, conf = f"fuel_median({fuel})", 0.25
                else:
                    # Final fallback to global median
                    val, by, conf = med_global, "global_median", 0.10
        
        updates.append({"unit_id": uid, "qty": float(val), "flag": 1, "by": by, "conf": conf})

    upd = pd.DataFrame(updates)
    upd.to_sql("tmp_tier3_oil", conn, if_exists="replace", index=False)
    conn.executescript(f"""
        UPDATE "{table}" AS f
        SET Quantity_initial_EJ        = (SELECT qty  FROM tmp_tier3_oil t WHERE t.unit_id = f.unit_id),
            _imputed_flag_quantity     = (SELECT flag FROM tmp_tier3_oil t WHERE t.unit_id = f.unit_id),
            _imputed_by_quantity       = (SELECT by   FROM tmp_tier3_oil t WHERE t.unit_id = f.unit_id),
            _imputation_conf_quantity  = (SELECT conf FROM tmp_tier3_oil t WHERE t.unit_id = f.unit_id)
        WHERE f.unit_id IN (SELECT unit_id FROM tmp_tier3_oil)
          AND (f.Quantity_initial_EJ IS NULL OR f.Quantity_initial_EJ <= 0);
        DROP TABLE tmp_tier3_oil;
    """)
    conn.commit()
    return len(upd)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 4: ML fallback
# ──────────────────────────────────────────────────────────────────────────────

def tier4_ml(conn: sqlite3.Connection, table: str):
    LOG.info("TIER 4 (oil): ML fallback")

    df = pd.read_sql(f"""
        SELECT unit_id, [country/area] AS Country, fuel_type, unit_type,
               latitude, longitude, discovery_year, Quantity_initial_EJ
        FROM "{table}"
    """, conn)

    # Features - fix column names to match actual schema
    cat_cols = ["Country", "fuel_type", "unit_type"]
    num_cols = ["latitude", "longitude", "discovery_year"]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    # Map actual columns to expected names
    df["Country"] = df["Country"].fillna("Unknown")
    df["fuel_type"] = df["fuel_type"].fillna("Unknown") 
    df["unit_type"] = df["unit_type"].fillna("Unknown")

    train = df[df["Quantity_initial_EJ"].notna() & (df["Quantity_initial_EJ"] > 0)]
    pred  = df[df["Quantity_initial_EJ"].isna() | (df["Quantity_initial_EJ"] <= 0)]

    if len(train) < 100 or pred.empty:
        LOG.warning(f"ML skipped: need ≥100 training samples, have {len(train)}; predict targets: {len(pred)}")
        return 0

    Xtr = train[cat_cols + num_cols]
    ytr = train["Quantity_initial_EJ"]
    Xpd = pred[cat_cols + num_cols]

    pre = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), cat_cols),
        remainder="passthrough"
    )

    pipe = Pipeline([
        ("prep", pre),
        ("hgb", HistGradientBoostingRegressor(random_state=42, max_depth=8))
    ])

    pipe.fit(Xtr, ytr)
    yhat = np.maximum(pipe.predict(Xpd), 0.001)

    upd = pd.DataFrame({
        "unit_id": pred["unit_id"].values,
        "qty": yhat,
        "flag": 1,
        "by": "ml_gradient_boosting",
        "conf": 0.20
    })
    upd.to_sql("tmp_tier4_oil", conn, if_exists="replace", index=False)
    conn.executescript(f"""
        UPDATE "{table}" AS f
        SET Quantity_initial_EJ        = (SELECT qty  FROM tmp_tier4_oil t WHERE t.unit_id = f.unit_id),
            _imputed_flag_quantity     = (SELECT flag FROM tmp_tier4_oil t WHERE t.unit_id = f.unit_id),
            _imputed_by_quantity       = (SELECT by   FROM tmp_tier4_oil t WHERE t.unit_id = f.unit_id),
            _imputation_conf_quantity  = (SELECT conf FROM tmp_tier4_oil t WHERE t.unit_id = f.unit_id)
        WHERE f.unit_id IN (SELECT unit_id FROM tmp_tier4_oil)
          AND (f.Quantity_initial_EJ IS NULL OR f.Quantity_initial_EJ <= 0);
        DROP TABLE tmp_tier4_oil;
    """)
    conn.commit()
    return len(upd)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("data") / "Energy.db")
    args = ap.parse_args()

    logging.basicConfig(
        filename="impute_oil.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    with sqlite3.connect(str(args.db)) as conn:
        table = "Oil_fields"
        add_missing_cols(conn, table)
        n1 = tier1_direct_initial(conn, table)
        n2 = tier2_reserves_plus_prod(conn, table)
        n3 = tier3_statistical(conn, table)
        n4 = tier4_ml(conn, table)

        stats = pd.read_sql(f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN Quantity_initial_EJ IS NOT NULL AND Quantity_initial_EJ > 0 THEN 1 ELSE 0 END) AS populated,
                   SUM(CASE WHEN _imputed_flag_quantity=1 THEN 1 ELSE 0 END) AS imputed
            FROM "{table}"
        """, conn).iloc[0]

        print(f"[oil] tier1={n1} tier2={n2} tier3={n3} tier4={n4}")
        print(f"[oil] FINAL COVERAGE: {int(stats['populated'])}/{int(stats['total'])} "
              f"({100*stats['populated']/max(1,stats['total']):.1f}%)")

if __name__ == "__main__":
    main()
