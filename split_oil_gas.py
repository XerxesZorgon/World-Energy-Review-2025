import sqlite3
import pandas as pd
from pathlib import Path
import os

# Define constants
SCRIPT_DIR = os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'Energy.db')

MENA_COUNTRIES = {
    # Based on World Bank/IMF MENA definition (Middle East & North Africa):
    "Algeria", "Bahrain", "Djibouti", "Egypt, Arab Rep.", "Iran, Islamic Rep.",
    "Iraq", "Jordan", "Kuwait", "Lebanon", "Libya", "Morocco", "Oman", "Qatar",
    "Saudi Arabia", "Sudan", "Syrian Arab Republic", "Tunisia",
    "United Arab Emirates", "Yemen"
}

NORTH_SEA_COUNTRIES = {
    "United Kingdom", "Norway", "Netherlands", "Denmark", "Germany"
}

# North Sea oil/gas basin -- bounding box
NSEA_BOUND = {
    "min_lat": 51.00,   # 50°59′43″ N (≈51.00°N)
    "max_lat": 61.02,   # 61° 1′ 1″ N
    "min_lon": -4.45,   # 4°26′ W (≈ –4.45°)
    "max_lon": 12.01    # 12°0′ E (≈12.01°E)
}

# Approximate U.S. bounding box for onshore Permian Basin (West Texas & SE New Mexico)
Permian = {
    "min_lat": 28.0,
    "max_lat": 34.5,
    "min_lon": -108.0,
    "max_lon": -101.0
}

# Gas fraction constants
REGION_RATIO = {
    "MENA": 0.15,           # 10–20 % → mid‑span
    "NorthSea": 0.35,       # 30–40 %
    "Permian": 0.37,        # 34–40 % (Permian)
    "Fallback": 0.20        # global fallback
}

def is_north_sea(lat, lon):
    return (NSEA_BOUND["min_lat"] <= lat <= NSEA_BOUND["max_lat"]
            and NSEA_BOUND["min_lon"] <= lon <= NSEA_BOUND["max_lon"])

def is_in_permian(lat, lon):
    return (Permian["min_lat"] <= lat <= Permian["max_lat"] and
            Permian["min_lon"] <= lon <= Permian["max_lon"])

def assign_gas_fraction(country, lat=None, lon=None):
    # MENA countries get fixed low GOR
    if country in MENA_COUNTRIES:
        return REGION_RATIO["MENA"]
    
    # North Sea countries must also fall within North Sea bounds
    if country in NORTH_SEA_COUNTRIES and lat is not None and lon is not None:
        if is_north_sea(lat, lon):
            return REGION_RATIO["NorthSea"]
    
    # U.S. wells: if in Permian Basin, assign Permian ratio
    if country == "United States" and lat is not None and lon is not None:
        return REGION_RATIO["Permian"] if is_in_permian(lat, lon) else REGION_RATIO["Fallback"]
    
    # All others use fallback
    return REGION_RATIO["Fallback"]

def _resolve_column(df: pd.DataFrame, candidates, required=True, label=""):
    """Return the first column from candidates that exists in df.columns.
    Candidates may include exact strings or callables for custom matching.
    If required and not found, raise KeyError with available columns preview.
    """
    cols = list(df.columns)
    # Try exact matches first (case-sensitive), then case-insensitive
    for c in candidates:
        if isinstance(c, str) and c in cols:
            return c
        if callable(c):
            for col in cols:
                try:
                    if c(col):
                        return col
                except Exception:
                    continue
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if isinstance(c, str) and c.lower() in lower_map:
            return lower_map[c.lower()]
    # Allow simple slash/space normalization
    norm = {str(c).lower().replace(" ", "").replace("_", "").replace("/", "/"): c for c in cols}
    for c in candidates:
        if isinstance(c, str):
            key = c.lower().replace(" ", "").replace("_", "").replace("/", "/")
            if key in norm:
                return norm[key]
    if required:
        preview = ", ".join(cols[:30]) + (" ..." if len(cols) > 30 else "")
        raise KeyError(f"Missing required column {label or candidates} in Oil_Gas_fields. Available: {preview}")
    return None

def split_oil_gas_fields(db_path=DB_PATH):
    """
    Split the Oil_Gas_fields table into separate Oil_fields and Gas_fields tables.
    For fields with 'oil and gas' fuel type, splits the quantity based on a typical ratio.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Read the original table
    query = "SELECT * FROM Oil_Gas_fields"
    df = pd.read_sql(query, conn)

    # Resolve schema variants for key columns
    fuel_col = _resolve_column(
        df,
        [
            'Fuel type', 'fuel_type', 'Fuel', 'fuel', 'Fueltype', 'fueltype'
        ],
        required=True,
        label='fuel'
    )
    country_col = _resolve_column(
        df,
        [
            'Country / Area', 'Country/Area', 'country / area', 'country/area',
            'Country', 'country'
        ],
        required=False,
        label='country'
    )
    lat_col = _resolve_column(df, ['Latitude', 'latitude', 'Lat', 'lat'], required=False, label='latitude')
    lon_col = _resolve_column(df, ['Longitude', 'longitude', 'Lon', 'lon'], required=False, label='longitude')
    # Quantity in EJ: prefer columns containing both quantity and EJ and ideally initial/init
    qty_col = _resolve_column(
        df,
        [
            'Quantity_initial_EJ', 'quantity_initial_ej', 'Quantity_initial', 'quantity_initial',
            # Heuristic predicate
            lambda c: ('quant' in str(c).lower()) and ('ej' in str(c).lower()) and (('initial' in str(c).lower()) or ('init' in str(c).lower())),
            # Fallback: any quantity with EJ
            lambda c: ('quant' in str(c).lower()) and ('ej' in str(c).lower())
        ],
        required=False,
        label='quantity_initial_ej'
    )
    uid_col = _resolve_column(df, ['Unit ID', 'unit_id', 'Unit_ID'], required=False, label='unit_id')

    # Normalize and coerce dtypes we rely on
    df[fuel_col] = df[fuel_col].astype(str).str.strip().str.lower()
    if country_col and country_col not in df.columns:
        df[country_col] = None

    # Coerce coordinates to numeric if present
    for c in (lat_col, lon_col):
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure numeric quantity
    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')

    # Create oil and gas dataframes (oil/gas only entries pass-through)
    oil_df = df[df[fuel_col] == 'oil'].copy()
    gas_df = df[df[fuel_col] == 'gas'].copy()

    # Handle 'oil and gas' fields
    combined_df = df[df[fuel_col].isin(['oil and gas', 'oil & gas', 'oil&gas'])].copy()

    if not combined_df.empty:
        # Compute per-row gas fraction using geography-aware heuristic
        ratios = combined_df.apply(
            lambda r: assign_gas_fraction(
                r.get(country_col), r.get(lat_col), r.get(lon_col)
            ), axis=1
        ).fillna(REGION_RATIO['Fallback'])

        # Create oil entries
        oil_entries = combined_df.copy()
        oil_entries[fuel_col] = 'oil'
        oil_entries['_split_ratio_gas'] = ratios
        oil_entries['_split_ratio_oil'] = 1 - ratios
        if qty_col and qty_col in oil_entries.columns:
            oil_entries[qty_col] = (oil_entries[qty_col].fillna(0) * oil_entries['_split_ratio_oil'])
        oil_entries['_split_from_combined'] = 1
        oil_entries['_original_unit_id'] = oil_entries[uid_col] if uid_col and uid_col in oil_entries.columns else None

        # Create gas entries
        gas_entries = combined_df.copy()
        gas_entries[fuel_col] = 'gas'
        gas_entries['_split_ratio_gas'] = ratios
        gas_entries['_split_ratio_oil'] = 1 - ratios
        if qty_col and qty_col in gas_entries.columns:
            gas_entries[qty_col] = (gas_entries[qty_col].fillna(0) * gas_entries['_split_ratio_gas'])
        gas_entries['_split_from_combined'] = 1
        gas_entries['_original_unit_id'] = gas_entries[uid_col] if uid_col and uid_col in gas_entries.columns else None

        # Append to the respective dataframes
        oil_df = pd.concat([oil_df, oil_entries], ignore_index=True)
        gas_df = pd.concat([gas_df, gas_entries], ignore_index=True)

    # Add source tracking to all entries
    oil_df['_source_table'] = 'Oil_Gas_fields'
    gas_df['_source_table'] = 'Oil_Gas_fields'

    # Default flags for pass-through rows
    for df_out in (oil_df, gas_df):
        if '_split_from_combined' not in df_out.columns:
            df_out['_split_from_combined'] = 0

    # Create new tables
    oil_df.to_sql('Oil_fields', conn, if_exists='replace', index=False)
    gas_df.to_sql('Gas_fields', conn, if_exists='replace', index=False)

    # Add indexes for better query performance
    with conn:
        conn.execute('CREATE INDEX IF NOT EXISTS idx_oil_fields_unit_id ON Oil_fields("Unit ID")')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_gas_fields_unit_id ON Gas_fields("Unit ID")')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_oil_fields_country ON Oil_fields("Country / Area")')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_gas_fields_country ON Gas_fields("Country / Area")')

    # Print summary
    print(f"Original records: {len(df)}")
    oil_split_ct = oil_df['_split_from_combined'].sum() if '_split_from_combined' in oil_df.columns else 0
    gas_split_ct = gas_df['_split_from_combined'].sum() if '_split_from_combined' in gas_df.columns else 0
    print(f"Oil fields: {len(oil_df)} (including {int(oil_split_ct)} split from combined)")
    print(f"Gas fields: {len(gas_df)} (including {int(gas_split_ct)} split from combined)")
    if not qty_col:
        print("[warn] No quantity (EJ) column found in Oil_Gas_fields; performed split without quantity scaling.")

    conn.close()
    return oil_df, gas_df

if __name__ == "__main__":
    oil_df, gas_df = split_oil_gas_fields(DB_PATH)