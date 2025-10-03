# energy_etl/utils.py
from pathlib import Path
import logging
import sqlite3
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data"
DB_PATH  = DATA_DIR / "Energy.db"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def get_logger(name: str = "energy_etl", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)

        # Optional file handler
        fh = logging.FileHandler(ROOT_DIR / "etl.log")
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger

log = get_logger()

# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
def get_conn(path: Path | str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(path)

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
_MISSING = {'*', '-', 'n/a', 'N/A', 'TBD', '', None}

def percent_to_float(val) -> float | None:
    """Convert '23%' or '30–40%' → 0.23 / 0.35, else NaN."""
    import re
    if pd.isna(val) or val in _MISSING: 
        return np.nan
    s = str(val).strip().replace(' ', '')  # remove narrow NBSP
    if '–' in s:
        lo, hi = (p.replace('%', '') for p in s.split('–'))
        try:
            return (float(lo) + float(hi)) / 200.0
        except ValueError:
            return np.nan
    if s.endswith('%'):
        try: 
            return float(s.rstrip('%')) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast='signed')
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df
