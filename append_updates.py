"""Utilities for appending annual update workbooks into the Energy ETL database.

The legacy outline referenced an `append_updates.py` module that would ingest new
GOGET/EI update workbooks. This implementation provides a pragmatic version that
reuses the cleaning helpers from `import_batch.py` and appends sheets to the
existing SQLite database with light schema alignment.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

try:  # Allow running as a package (`python -m energy_etl.append_updates`) or script
    from . import import_batch
except ImportError:  # pragma: no cover - direct script execution
    import import_batch  # type: ignore

LOGGER = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "Energy.db"
COLUMN_TYPES_FILE = PACKAGE_ROOT / "Text files" / "column_types.txt"


@dataclass(slots=True)
class AppendResult:
    """Summary of a single sheet append operation."""

    workbook: Path
    sheet: str
    table: str
    rows_written: int
    mode: str


def _slugify_table_name(name: str) -> str:
    cleaned = name.strip().replace("/", "_").replace("\\", "_")
    cleaned = cleaned.replace(" ", "_")
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in cleaned)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    return sanitized or "Sheet"


def _load_column_type_map() -> MutableMapping[str, MutableMapping[str, type]]:
    if COLUMN_TYPES_FILE.exists():
        return import_batch.parse_column_types(str(COLUMN_TYPES_FILE))  # type: ignore[arg-type]
    return {}


def _get_existing_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info('{table}')")
    except sqlite3.DatabaseError:
        return []
    return [row[1] for row in cur.fetchall()]


def _align_to_existing(
    df: pd.DataFrame,
    existing_cols: Sequence[str],
    table: str,
    strict_columns: bool,
) -> pd.DataFrame:
    if not existing_cols:
        return df

    df = df.copy()
    extra_cols = [c for c in df.columns if c not in existing_cols]
    if extra_cols:
        if strict_columns:
            raise ValueError(
                f"Sheet columns {extra_cols!r} do not exist in table '{table}'"
            )
        LOGGER.warning(
            "Dropping %d unexpected column(s) for table '%s': %s",
            len(extra_cols),
            table,
            ", ".join(extra_cols),
        )
        df = df.drop(columns=extra_cols)

    for missing in (c for c in existing_cols if c not in df.columns):
        LOGGER.debug("Adding missing column '%s' with nulls for table '%s'", missing, table)
        df[missing] = pd.NA

    return df.loc[:, list(existing_cols)]


def append_workbook(
    workbook_path: Path | str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    sheet_map: Mapping[str, str] | None = None,
    include_sheets: Iterable[str] | None = None,
    exclude_sheets: Iterable[str] | None = None,
    if_exists: str = "append",
    strict_columns: bool = False,
    drop_empty: bool = True,
) -> list[AppendResult]:
    """Append an update workbook into the SQLite database.

    Parameters
    ----------
    workbook_path:
        Path to the Excel workbook containing GOGET/EI updates.
    db_path:
        SQLite database path. Defaults to `data/Energy.db`.
    sheet_map:
        Optional explicit mapping ``{"Sheet Name": "TableName"}``. If omitted,
        sheet names are slugified to derive table names.
    include_sheets / exclude_sheets:
        Optional iterables to whitelist/skip specific sheet names (case-insensitive).
    if_exists:
        How to handle existing tables: ``append`` (default), ``replace``, or ``skip``.
    strict_columns:
        If ``True``, raise when sheet columns deviate from the existing table schema.
        Otherwise, unexpected columns are dropped and missing ones filled with NULLs.
    drop_empty:
        If ``True`` (default) skip sheets that become empty after cleaning.
    """

    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {db_path}. Run import_batch.py first."
        )

    include_set = {s.lower() for s in include_sheets} if include_sheets else None
    exclude_set = {s.lower() for s in exclude_sheets} if exclude_sheets else set()
    mapping = {k.lower(): v for k, v in (sheet_map or {}).items()}

    column_map = _load_column_type_map()
    xls = pd.ExcelFile(workbook_path)
    results: list[AppendResult] = []

    with sqlite3.connect(str(db_path)) as conn:
        for sheet_name in xls.sheet_names:
            key = sheet_name.lower()
            if include_set is not None and key not in include_set:
                LOGGER.debug("Skipping sheet '%s' (not in include filter)", sheet_name)
                continue
            if key in exclude_set:
                LOGGER.debug("Skipping sheet '%s' (exclude filter)", sheet_name)
                continue

            table_name = mapping.get(key) or _slugify_table_name(sheet_name)
            LOGGER.info("Processing sheet '%s' → table '%s'", sheet_name, table_name)

            df = xls.parse(sheet_name=sheet_name)
            df = import_batch.clean_dataframe(df, table_name, column_map)
            if drop_empty and df.empty:
                LOGGER.warning("Sheet '%s' produced an empty frame after cleaning; skipped", sheet_name)
                continue

            existing_cols = _get_existing_columns(conn, table_name)

            if existing_cols and if_exists == "skip":
                LOGGER.info("Table '%s' already exists; skipping due to if_exists=skip", table_name)
                continue

            mode = if_exists
            if existing_cols and if_exists == "replace":
                LOGGER.info("Replacing existing table '%s'", table_name)
                conn.execute(f"DROP TABLE IF EXISTS '{table_name}'")
                existing_cols = []

            if existing_cols:
                df = _align_to_existing(df, existing_cols, table_name, strict_columns)
                write_mode = "append"
            else:
                write_mode = "replace"

            df.to_sql(table_name, conn, if_exists=write_mode, index=False)
            rows = int(df.shape[0])
            LOGGER.info("Wrote %d row(s) to table '%s'", rows, table_name)
            results.append(AppendResult(workbook=workbook_path, sheet=sheet_name, table=table_name, rows_written=rows, mode=mode))

    return results


def _parse_sheet_map_arg(pairs: Sequence[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"Mapping '{pair}' must use the form Sheet=Table")
        sheet, table = pair.split("=", 1)
        mapping[sheet.strip()] = table.strip()
    return mapping


def _parse_sheet_map_file(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Mapping JSON must be an object mapping sheets to tables")
    return {str(k): str(v) for k, v in data.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one or more update workbooks to the Energy ETL SQLite database."
    )
    parser.add_argument("workbooks", nargs="+", type=Path, help="Update workbook(s) to ingest")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to Energy.db")
    parser.add_argument(
        "--map",
        action="append",
        dest="map_pairs",
        metavar="SHEET=TABLE",
        help="Explicit mapping from sheet name to target table (repeatable)",
    )
    parser.add_argument(
        "--map-json",
        type=Path,
        help="JSON file containing {\"Sheet\": \"Table\"} overrides",
    )
    parser.add_argument(
        "--include-sheet",
        action="append",
        dest="include_sheets",
        help="Process only the specified sheet (can repeat)",
    )
    parser.add_argument(
        "--exclude-sheet",
        action="append",
        dest="exclude_sheets",
        help="Skip the specified sheet (can repeat)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["append", "replace", "skip"],
        default="append",
        help="Behaviour when the destination table already exists",
    )
    parser.add_argument(
        "--strict-columns",
        action="store_true",
        help="Abort if sheet columns differ from the current table schema",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_false",
        dest="drop_empty",
        help="Write empty sheets even after cleaning",
    )
    parser.set_defaults(drop_empty=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    sheet_map = _parse_sheet_map_file(args.map_json)
    sheet_map.update(_parse_sheet_map_arg(args.map_pairs))

    for workbook in args.workbooks:
        results = append_workbook(
            workbook,
            db_path=args.db,
            sheet_map=sheet_map,
            include_sheets=args.include_sheets,
            exclude_sheets=args.exclude_sheets,
            if_exists=args.if_exists,
            strict_columns=args.strict_columns,
            drop_empty=args.drop_empty,
        )
        total_rows = sum(r.rows_written for r in results)
        LOGGER.info(
            "Workbook '%s': appended %d sheet(s), %d total row(s)",
            workbook,
            len(results),
            total_rows,
        )


if __name__ == "__main__":
    main()
