"""
import_batch.py
─────────────────────────   

Batch import all sheets from the Excel files to the database.

Usage:
    python import_batch.py

Author: John Peach
eurAIka sciences
Date: 2025-08-02
Version: 0.1
License: MIT

"""

# Import libraries
import pandas as pd
import numpy as np
import sqlite3
import os
import difflib

# Define manifest reader
def read_manifest(path):
    """
    Read the manifest file (Excel or CSV) and return as a DataFrame.
    """
    if path.endswith(".xlsx"):
        return pd.read_excel(path, header=None)
    elif path.endswith(".csv"):
        return pd.read_csv(path, header=None)
    else:
        raise ValueError("Manifest must be .xlsx or .csv")

# Define constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'Energy.db')
MANIFEST_PATH = os.path.join(SCRIPT_DIR, 'Spreadsheet data', 'Energy manifest.xlsx')
TEXT_PATH = os.path.join(SCRIPT_DIR, 'Text files', 'column_types.txt')

MISSING = {'*','-','n/a','N/A','TBD','', None}

# Helper: Convert percent string to float
def percent_to_float(val):
    """
    Convert a percent string to a float.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.strip()
        if val.endswith('%'):
            try:
                return float(val.rstrip('%')) / 100.0
            except Exception:
                return np.nan
        if val in MISSING:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def is_year_column(colname):
    """
    Check if a column name is a year.
    """ 
    try:
        year = int(str(colname).strip())
        return 1800 <= year <= 2100
    except:
        return False

def average_range(val):
    """
    Convert a range string to the average of the range.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.strip()
        if '-' in val:
            try:
                start, end = map(float, val.split('-'))
                return (start + end) / 2.0
            except Exception:
                return np.nan
        if val in MISSING:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def parse_column_types(path=TEXT_PATH):
    """Parse column_types.txt into a mapping: {table: {col: type}}"""
    colmap = {}
    table = None
    type_map = {"REAL": float, "INTEGER": int}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.endswith(":"):
                table = line.rstrip(":").lower() if line else table
                continue
            if "\t" in line:
                col, typ = line.split("\t")
            elif " " in line:
                col, typ = line.rsplit(" ", 1)
            else:
                continue
            col = col.strip().lower()
            typ = typ.strip().upper()
            if table:
                colmap.setdefault(table, {})[col] = type_map.get(typ, str)
    return colmap

def coerce_types(df, table_name, coltype_map):
    """Coerce DataFrame columns to types as per coltype_map for table_name."""
    table = table_name.lower()
    if table not in coltype_map:
        return df
    for col, typ in coltype_map[table].items():
        matches = [c for c in df.columns if c.strip().lower() == col]
        if matches:
            c = matches[0]
            if typ is float:
                df[c] = df[c].apply(average_range)
            try:
                df[c] = df[c].astype(typ)
            except Exception:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_dataframe(df, table_name=None, coltype_map=None):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
    missing_set = {'*', 'n/a', 'N/A', '-', 'TBD', '', None}
    for col in df.columns:
        df[col] = df[col].replace(list(missing_set), np.nan)
    if table_name and coltype_map:
        df = coerce_types(df, table_name, coltype_map)
    print("[DIAG] Preview after cleaning:")
    print(df.head(2), df.dtypes)
    return df

def import_sheet(file_path, sheet_name, table_name, conn, coltype_map=None):
    """
    Import a sheet from an Excel file.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError as e:
        xl = pd.ExcelFile(file_path)
        candidates = xl.sheet_names
        closest = difflib.get_close_matches(sheet_name, candidates, n=1, cutoff=0.7)
        if closest:
            print(f"⚠ Sheet '{sheet_name}' not found. Using closest match: '{closest[0]}'")
            df = xl.parse(closest[0])
        else:
            print(f"❌ Sheet '{sheet_name}' not found in '{os.path.basename(file_path)}'. Available sheets: {candidates}")
            raise e
    df = clean_dataframe(df, table_name, coltype_map)
    if df.empty:
        print(f"[WARN] DataFrame for table '{table_name}' is empty after cleaning. Skipping import.")
        return
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"✔ Imported '{sheet_name}' → '{table_name}' ({df.shape[0]} rows)")

# --- Main Execution ---
def batch_import(manifest_path):
    """
    Import all sheets from the manifest file.
    """
    manifest_df = read_manifest(manifest_path)
    conn = sqlite3.connect(DB_PATH)

    current_file = None
    for i, row in manifest_df.iterrows():
        cells = row.dropna().astype(str).tolist()
        if len(cells) == 1:
            current_file = os.path.join(os.path.dirname(manifest_path), cells[0])
            if not os.path.exists(current_file):
                print(f"⚠ File not found: {current_file}")
                current_file = None
            else:
                print(f"\n📂 Now processing: {os.path.basename(current_file)}")
        elif len(cells) == 2 and current_file:
            sheet_name, table_name = cells
            try:
                import_sheet(current_file, sheet_name.strip(), table_name.strip(), conn)
            except Exception as e:
                print(f"❌ Failed: {sheet_name} → {table_name} | {e}")
        else:
            print(f"⚠ Skipping malformed row {i+1}: {cells}")

    conn.close()
    print("\n✅ All manifest entries processed.")


def batch_import_from_manifest(manifest_path: str | None = None) -> None:
    """Public wrapper used by `run_pipeline.py` to rebuild the database."""
    path = manifest_path or MANIFEST_PATH
    batch_import(path)


def main() -> None:
    batch_import_from_manifest()


if __name__ == "__main__":
    main()