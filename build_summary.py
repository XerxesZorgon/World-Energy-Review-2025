"""High-level summary orchestration for the Energy ETL.

`run_pipeline.py` expects a `build_summary.py` module that can be invoked as part
of the pipeline to regenerate downstream summary artifacts. This implementation
leverages the existing `summarize_fossil_fuels.py` and `fix_reserves.py` scripts
and coordinates additional reconciliation outputs.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Iterable, Sequence
import contextlib

try:  # Allow execution either inside the package or as scripts
    from . import summarize_fossil_fuels
    from . import fix_reserves
except ImportError:  # pragma: no cover
    import summarize_fossil_fuels  # type: ignore
    import fix_reserves  # type: ignore

LOGGER = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data") / "Energy.db"
DEFAULT_SUMMARY_PATH = Path("fossil_fuels_summary.csv")
DEFAULT_RECON_PATH = Path("fossil_fuels_reconciliation.csv")
DEFAULT_FIXED_PATH = Path("fossil_fuels_summary_fixed_v2.csv")


@contextlib.contextmanager
def _temporary_argv(argv: Iterable[str]) -> Iterable[str]:
    original = sys.argv
    sys.argv = list(argv)
    try:
        yield sys.argv
    finally:
        sys.argv = original


def generate_summaries(
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
    recon_path: Path | str = DEFAULT_RECON_PATH,
    year_min: int | None = None,
    year_max: int | None = None,
    prefer_calibrated: bool = True,
    apply_reserve_growth: bool = False,
    growth_rate: float = 0.3,
    growth_horizon: int = 30,
    growth_tau: float | None = None,
) -> None:
    """Generate the canonical fossil fuel summary CSVs and reconciliation report."""

    argv = [
        "summarize_fossil_fuels.py",
        "--db",
        str(Path(db_path)),
        "--out",
        str(Path(summary_path)),
        "--format",
        "wide",
        "--recon-out",
        str(Path(recon_path)),
    ]
    if year_min is not None:
        argv.extend(["--year-min", str(year_min)])
    if year_max is not None:
        argv.extend(["--year-max", str(year_max)])
    if not prefer_calibrated:
        LOGGER.warning("summarize_fossil_fuels CLI lacks a flag to disable calibrated preference; proceeding with defaults")
    if apply_reserve_growth:
        argv.append("--apply-reserve-growth")
        argv.extend(["--growth-rate", str(growth_rate)])
        argv.extend(["--growth-horizon", str(growth_horizon)])
        if growth_tau is not None:
            argv.extend(["--growth-tau", str(growth_tau)])

    LOGGER.info("Running summarize_fossil_fuels via argv: %s", " ".join(argv[1:]))
    with _temporary_argv(argv):
        summarize_fossil_fuels.main()


def apply_reserve_fixes(
    *,
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
    db_path: Path | str = DEFAULT_DB_PATH,
    output_path: Path | str = DEFAULT_FIXED_PATH,
) -> None:
    LOGGER.info("Applying reserve fixes using %s", getattr(fix_reserves, "__file__", "fix_reserves"))
    fix_reserves.main(
        summary_path=Path(summary_path),
        db_path=Path(db_path),
        output_path=Path(output_path),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Energy ETL summary tables")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to Energy.db")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output fossil fuels summary CSV",
    )
    parser.add_argument(
        "--recon",
        type=Path,
        default=DEFAULT_RECON_PATH,
        help="Reconciliation report output path",
    )
    parser.add_argument(
        "--fixed",
        type=Path,
        default=DEFAULT_FIXED_PATH,
        help="Reserve-fixed summary output path",
    )
    parser.add_argument("--year-min", type=int, help="Minimum year to include", default=None)
    parser.add_argument("--year-max", type=int, help="Maximum year to include", default=None)
    parser.add_argument(
        "--no-calibrated",
        dest="prefer_calibrated",
        action="store_false",
        help="Prefer raw Quantity_initial_EJ columns instead of calibrated ones",
    )
    parser.add_argument(
        "--apply-reserve-growth",
        action="store_true",
        help="Apply reserve-growth/backdating when summarising discoveries",
    )
    parser.add_argument("--growth-rate", type=float, default=0.3)
    parser.add_argument("--growth-horizon", type=int, default=30)
    parser.add_argument("--growth-tau", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | argparse.Namespace | None = None) -> None:
    if isinstance(argv, argparse.Namespace):
        args = argv
    else:
        args = parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    generate_summaries(
        db_path=args.db,
        summary_path=args.summary,
        recon_path=args.recon,
        year_min=args.year_min,
        year_max=args.year_max,
        prefer_calibrated=args.prefer_calibrated,
        apply_reserve_growth=args.apply_reserve_growth,
        growth_rate=args.growth_rate,
        growth_horizon=args.growth_horizon,
        growth_tau=args.growth_tau,
    )

    apply_reserve_fixes(
        summary_path=args.summary,
        db_path=args.db,
        output_path=args.fixed,
    )

    LOGGER.info("Summary build complete → %s", Path(args.summary).resolve())
    LOGGER.info("Reserve fixes saved → %s", Path(args.fixed).resolve())


if __name__ == "__main__":
    main()
