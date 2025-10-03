#!/usr/bin/env python
"""
run_pipeline.py ─ master driver for the Energy ETL
=================================================

A single command‑line entry‑point that lets you:
  • **--rebuild**  → rebuild the SQLite database from the manifest (rare)
  • **--append**   → append one‑or‑many annual update workbooks
  • **--summary**  → refresh the high‑level summary tables
  • **--impute**   → launch the heavy imputation stage

Typical runs
------------
```bash
# once per year after you drop the 2025 workbook in
python run_pipeline.py --append "updates_2025.xlsx" --summary

# cold rebuild (only if the manifest layout changes)
python run_pipeline.py --rebuild --summary --impute
```

Notes
-----
*   All heavyweight logic remains in the specialised modules you already wrote
    (`import_batch`, `append_updates`, `build_summary`, `impute_energy_db`).
*   Works on Windows 11 (no POSIX‑specific calls).
*   Uses `logging` rather than `print` so you can redirect to a file if desired:
    ```bash
    python run_pipeline.py --append 2025.xlsx -v > run_2025.log 2>&1
    ```

Author: John Peach
eurAIka sciences
Date: 2025-08-02
Version: 0.1
License: MIT
    
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
#  internal imports – keep them late‑bound so the CLI works even if a stage is
#  missing in early development
# ──────────────────────────────────────────────────────────────────────────────
from energy_etl import import_batch, append_updates, build_summary

# Heavy imputation only imported if requested (it pulls in sklearn, etc.)
IMPUTE_MODULE = "energy_etl.impute.impute_energy_db"

#  helpers
# ──────────────────────────────────────────────────────────────────────────────

def _config_logging(verbose: bool = False) -> None:
    fmt = "%(asctime)s :: %(levelname)s :: %(message)s"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _import_optional(name: str):
    """Import *name* only if the user asked for its stage."""
    import importlib
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        logging.error("Requested stage '%s' but module is missing (\n  %s\n)", name, exc)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
#  command‑line interface
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Energy ETL pipeline")

    # mutually exclusive? → no, you might rebuild *and* append in one go
    p.add_argument("--rebuild", action="store_true", help="rebuild DB from manifest")
    p.add_argument("--append", nargs="+", metavar="XLSX", help="one or more update workbooks to append")
    p.add_argument("--summary", action="store_true", help="refresh summary tables")
    p.add_argument("--impute", action="store_true", help="run heavy imputation stage")
    p.add_argument("-v", "--verbose", action="store_true", help="chatty logging (DEBUG level)")

    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
#  main orchestration
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _config_logging(args.verbose)

    t0 = time.time()

    # 1) bootstrap / rebuild
    if args.rebuild:
        logging.info("[1/4] Rebuilding SQLite database from manifest …")
        import_batch.batch_import_from_manifest()

    # 2) append yearly updates
    if args.append:
        logging.info("[2/4] Appending %d workbook(s) …", len(args.append))
        for wb in args.append:
            path = Path(wb).expanduser()
            if not path.is_file():
                logging.error("Workbook not found: %s", path)
                sys.exit(1)
            append_updates.append_workbook(path)

    # 3) summary tables
    if args.summary:
        logging.info("[3/4] Generating summary tables …")
        build_summary.generate_summaries()
        build_summary.apply_reserve_fixes()

    # 4) imputation (late import to keep startup light)
    if args.impute:
        logging.info("[4/4] Running imputation stage … this can take a while …")
        imputer = _import_optional(IMPUTE_MODULE)
        imputer.main()

    d = time.time() - t0
    logging.info("Pipeline complete in %.1f seconds", d)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
