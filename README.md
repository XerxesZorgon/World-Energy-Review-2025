# Energy ETL

## Overview
The Energy ETL toolkit ingests Energy Institute (EI) workbooks and GOGET field data into a consolidated SQLite database, builds fossil fuel discovery and production summaries, and applies repair/imputation utilities for reserves analysis. Scripts live in the root of `energy_etl/` and share a common data store at `data/Energy.db`.

## Key capabilities
- **Database ingest** `import_batch.py` rebuilds `Energy.db` from the manifest and raw spreadsheets.
- **Summary generation** `summarize_fossil_fuels.py` creates long and wide CSV exports of discoveries, production, and reserves.
- **Reserves repair** `fix_reserves.py`/`fix_reserves_v2.py` benchmark against EI proved reserves and smooth negative values; `build_modified_discoveries.py` prepares backdated discoveries.
- **Conversion utilities** `convert_gas_oil_to_EJ.py` and `convert_coal_to_EJ.py` normalize disparate units to exajoules.
- **Imputation workflows** `impute_oil_gas_db.py`, `impute_oil.py`, `impute_gas.py`, and `impute_coal.py` fill gaps with stratified statistics and calibration from `calibrate_imputation.py`.
- **Diagnostics & checks** scripts such as `check_tables.py`, `diagnose_missing_quantities.py`, `coverage_analysis.py`, `statistical_tests.py`, and `verify_imputation.py` validate schema, coverage, and model quality.
- **Pipeline driver** `run_pipeline.py` ties the stages together (rebuild, append, summary, impute). It currently expects `append_updates.py` and `build_summary.py`, which are not yet present in this snapshot.

## Repository structure (selected)
- **data/** `Energy.db` plus `Discoveries_Production_backdated.csv` seed inputs.
- **Spreadsheet data/** EI manifest and update workbooks (not tracked here).
- **Text files/** documentation aids including this outline and column mappings.
- **archive/** historical prototypes (`impute_energy_db_v0.py`, `impute_energy_db_v1.py`).
- **Conversion scripts** `convert_coal_to_EJ.py`, `convert_gas_oil_to_EJ.py`.
- **Reserves & summaries** `summarize_fossil_fuels.py`, `build_modified_discoveries.py`, `fix_reserves*.py`.
- **Imputation suite** `impute_*` scripts, fit helpers (`fit_Richards.py`, `fit_Richards_2Pt.py`, `logistic_fit.py`).
- **Utilities** `utils.py`, `_peek_schema.py`, `_peek_cols_runtime.py`, `_peek_schema_cols.py`.
- **Verification artifacts** `verify_imputation.py`, `verify_100_percent.py`, `results_*.txt`, `results_*.tsv`, and recent `impute_*.log` files.

A fuller inventory lives in `Text files/Outline.txt` and includes generated CSV outputs (e.g., `fossil_fuels_summary*.csv`, `fuels_summary_reduced*.csv`).

## Requirements
- Python 3.10+
- Python packages: `pandas`, `numpy`, `sqlite3` (standard library), `argparse` (standard), and `scikit-learn` for advanced imputation modules. Additional scripts may rely on `scipy` and `matplotlib` for fitting/plotting.
- EI/GOGET workbooks referenced by the manifest in `Spreadsheet data/`.

To install recommended packages inside a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate        # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn scipy matplotlib
echo "Manual: add any other libs used by your scripts" 
```

## Usage
- **Rebuild database**:
  ```bash
  python run_pipeline.py --rebuild --summary
  ```
  This calls `import_batch.py` to ingest the manifest and (once implemented) would call `build_summary.py` to refresh aggregates.

- **Append yearly updates** *(pending)*: `run_pipeline.py` anticipates `append_updates.py`, which should ingest new GOGET/EI workbooks. Implement or restore this module before using `--append`.

- **Generate summaries only**:
  ```bash
  python summarize_fossil_fuels.py --db data/Energy.db --out fossil_fuels_summary.csv --format long --year-min 1850 --year-max 2100
  ```

- **Run imputation**:
  ```bash
  python impute_oil_gas_db.py --tables Oil Gas --db data/Energy.db
  python calibrate_imputation.py --tables Oil Gas Coal
  ```
  Review `verify_imputation.py` and `statistical_tests.py` for post-run validation.

- **Unit conversion**:
  ```bash
  python convert_gas_oil_to_EJ.py --quantity 20 --unit "bcm" --fuel_type "dry gas"
  python convert_coal_to_EJ.py --quantity 5 --unit "Mt"
  ```

## Data inputs & outputs
- **Inputs**: EI reserves/production tables stored in `data/Energy.db`, GOGET manifest and supplementary CSVs under `Spreadsheet data/`.
- **Outputs**: Summary CSVs (`fossil_fuels_summary*.csv`, `fuels_summary_reduced*.csv`), reconciliation reports, modified discovery series, and diagnostic logs in the project root.

## Notes & future work
- Implement `append_updates.py` and `build_summary.py` (referenced in `run_pipeline.py`).
- Consider restructuring into a package (`energy_etl/__init__.py`, configuration modules) if distribution is planned.
- Maintain versioned copies of large CSV outputs outside version control or rotate into a `results/` directory.

## Support
For questions or enhancements, contact John Peach (Wild Peaches) or update the issue tracker accompanying this repository.
