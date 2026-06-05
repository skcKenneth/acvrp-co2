"""
aggregate_results.py
====================

Aggregate randomized ACVRP-CO2 raw results into summary tables.

Default input:
    results/randomized/raw_results.csv

Outputs:
    results/randomized/penalty_summary.csv
    results/randomized/city_overview.csv
    results/randomized/statistical_tests.csv  (if SciPy is available)
    results/randomized/randomized_results_table.tex

Raw input schema:
city,instance_id,solver,variant,reference_variant,num_customers,total_demand,
distance_m,fuel_l,co2_kg,distance_penalty_pct,fuel_penalty_pct,
co2_penalty_pct,feasible,infeasible_reason,runtime_seconds,seed

CLI:
    python -m src.aggregate_results --input results/randomized/raw_results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "city",
    "instance_id",
    "solver",
    "variant",
    "reference_variant",
    "num_customers",
    "total_demand",
    "distance_m",
    "fuel_l",
    "co2_kg",
    "distance_penalty_pct",
    "fuel_penalty_pct",
    "co2_penalty_pct",
    "feasible",
    "infeasible_reason",
    "runtime_seconds",
    "seed",
]

VARIANTS = ["AR", "SE", "SM", "SR"]
NON_AR_VARIANTS = ["SE", "SM", "SR"]
METRICS = ["distance_m", "fuel_l", "co2_kg"]
PENALTY_COLUMNS = [
    "distance_penalty_pct",
    "fuel_penalty_pct",
    "co2_penalty_pct",
]


def project_path(raw_path: str | Path, project_root: Path) -> Path:
    """Resolve relative paths against the repository root."""
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def load_raw_results(input_path: Path) -> pd.DataFrame:
    """Read raw randomized results and validate required columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Raw randomized results not found: {input_path}")
    df = pd.read_csv(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Raw results missing required columns: {missing}")
    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric and boolean columns used by validation/aggregation."""
    out = df.copy()
    out["city"] = out["city"].astype(str)
    out["variant"] = out["variant"].astype(str)
    out["reference_variant"] = out["reference_variant"].astype(str)
    out["instance_id"] = pd.to_numeric(out["instance_id"], errors="raise").astype(int)
    for column in [
        "num_customers",
        "total_demand",
        "distance_m",
        "fuel_l",
        "co2_kg",
        "distance_penalty_pct",
        "fuel_penalty_pct",
        "co2_penalty_pct",
        "runtime_seconds",
        "seed",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out["feasible"].dtype == bool:
        return out
    out["feasible"] = (
        out["feasible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "false": False, "0": False})
    )
    if out["feasible"].isna().any():
        raise ValueError("Column feasible contains values other than true/false.")
    return out


def validate_experiment(df: pd.DataFrame) -> list[str]:
    """
    Validate completeness and return non-fatal warnings.

    Raises ValueError for schema/completeness failures that make summaries
    unreliable.
    """
    warnings: list[str] = []
    seen_variants = set(df["variant"].unique())
    missing_global = set(VARIANTS) - seen_variants
    if missing_global:
        raise ValueError(f"Missing variants in raw results: {sorted(missing_global)}")

    grouped = df.groupby(["city", "instance_id"])
    for (city, instance_id), group in grouped:
        variants = set(group["variant"])
        missing = set(VARIANTS) - variants
        extra = variants - set(VARIANTS)
        if missing or extra:
            raise ValueError(
                f"{city} instance {instance_id} variants invalid: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        counts = group["variant"].value_counts()
        duplicates = counts[counts != 1]
        if not duplicates.empty:
            raise ValueError(
                f"{city} instance {instance_id} has duplicate variant rows: "
                f"{duplicates.to_dict()}"
            )

    city_variant_counts = df.groupby(["city", "variant"])["instance_id"].nunique()
    for city, counts in city_variant_counts.groupby(level=0):
        values = counts.droplevel(0)
        if values.nunique() != 1:
            raise ValueError(
                f"City {city} has unbalanced instance counts by variant: "
                f"{values.to_dict()}"
            )

    ar = df[df["variant"] == "AR"]
    tolerance = 1e-7
    for column in PENALTY_COLUMNS:
        bad = ar[column].abs() > tolerance
        if bad.any():
            max_abs = float(ar.loc[bad, column].abs().max())
            raise ValueError(
                f"AR rows have non-zero {column}; max abs={max_abs:.6g}"
            )

    infeasible = df[~df["feasible"]]
    if not infeasible.empty:
        warnings.append(
            f"{len(infeasible)} raw result rows have feasible=false."
        )

    non_ar = df[df["variant"] != "AR"]
    for column in PENALTY_COLUMNS:
        negative = non_ar[non_ar[column] < 0]
        if not negative.empty:
            warnings.append(
                f"{len(negative)} non-AR rows have negative {column}."
            )

    return warnings


def iqr(series: pd.Series) -> float:
    """Return interquartile range using pandas quantiles."""
    return float(series.quantile(0.75) - series.quantile(0.25))


def summarize_penalties(df: pd.DataFrame) -> pd.DataFrame:
    """Compute city/variant penalty summary statistics excluding AR rows."""
    non_ar = df[df["variant"] != "AR"].copy()
    rows: list[dict[str, float | int | str]] = []
    for (city, variant), group in non_ar.groupby(["city", "variant"], sort=True):
        rows.append(
            {
                "city": city,
                "variant": variant,
                "n": int(group["instance_id"].nunique()),
                "mean_distance_penalty": group["distance_penalty_pct"].mean(),
                "sd_distance_penalty": group["distance_penalty_pct"].std(ddof=1),
                "median_distance_penalty": group["distance_penalty_pct"].median(),
                "iqr_distance_penalty": iqr(group["distance_penalty_pct"]),
                "mean_fuel_penalty": group["fuel_penalty_pct"].mean(),
                "sd_fuel_penalty": group["fuel_penalty_pct"].std(ddof=1),
                "median_fuel_penalty": group["fuel_penalty_pct"].median(),
                "iqr_fuel_penalty": iqr(group["fuel_penalty_pct"]),
                "mean_co2_penalty": group["co2_penalty_pct"].mean(),
                "sd_co2_penalty": group["co2_penalty_pct"].std(ddof=1),
                "median_co2_penalty": group["co2_penalty_pct"].median(),
                "iqr_co2_penalty": iqr(group["co2_penalty_pct"]),
                "min_co2_penalty": group["co2_penalty_pct"].min(),
                "max_co2_penalty": group["co2_penalty_pct"].max(),
            }
        )
    return pd.DataFrame(rows)


def city_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Compute city-level overview metrics from AR and non-AR rows."""
    rows: list[dict[str, float | int | str]] = []
    for city, group in df.groupby("city", sort=True):
        ar = group[group["variant"] == "AR"]
        non_ar = group[group["variant"] != "AR"]
        per_instance = non_ar.groupby("instance_id")["co2_penalty_pct"]
        best = per_instance.min()
        worst = per_instance.max()
        rows.append(
            {
                "city": city,
                "n_instances": int(ar["instance_id"].nunique()),
                "mean_ar_distance_m": ar["distance_m"].mean(),
                "mean_ar_fuel_l": ar["fuel_l"].mean(),
                "mean_ar_co2_kg": ar["co2_kg"].mean(),
                "mean_best_non_ar_co2_penalty": best.mean(),
                "mean_worst_non_ar_co2_penalty": worst.mean(),
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_tests(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Run paired Wilcoxon tests if SciPy is available."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None, "SciPy is not available; skipping Wilcoxon signed-rank tests."

    rows: list[dict[str, float | int | str]] = []
    for city, city_df in df.groupby("city", sort=True):
        for variant in NON_AR_VARIANTS:
            merged = city_df[city_df["variant"] == variant].merge(
                city_df[city_df["variant"] == "AR"],
                on=["city", "instance_id"],
                suffixes=("_variant", "_ar"),
                how="inner",
            )
            for metric in METRICS:
                variant_values = merged[f"{metric}_variant"].to_numpy(dtype=float)
                ar_values = merged[f"{metric}_ar"].to_numpy(dtype=float)
                finite = np.isfinite(variant_values) & np.isfinite(ar_values)
                variant_values = variant_values[finite]
                ar_values = ar_values[finite]
                if len(variant_values) == 0:
                    statistic = np.nan
                    p_value = np.nan
                else:
                    differences = variant_values - ar_values
                    if np.allclose(differences, 0.0):
                        statistic = 0.0
                        p_value = 1.0
                    else:
                        result = wilcoxon(variant_values, ar_values)
                        statistic = float(result.statistic)
                        p_value = float(result.pvalue)
                rows.append(
                    {
                        "city": city,
                        "variant": variant,
                        "metric": metric,
                        "test": "wilcoxon_signed_rank",
                        "n": int(len(variant_values)),
                        "statistic": statistic,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(rows), None


def latex_escape(value: str) -> str:
    """Escape a small subset of LaTeX-sensitive characters."""
    return value.replace("_", r"\_").replace("&", r"\&")


def mean_sd(mean: float, sd: float) -> str:
    """Format mean +/- SD for LaTeX table cells."""
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def make_latex_table(summary: pd.DataFrame) -> str:
    """Build a LaTeX tabular summarizing randomized penalties."""
    lines = [
        r"\begin{tabular}{llrrr r}",
        r"\hline",
        r"City & Variant & Distance penalty (\%) & Fuel penalty (\%) & CO2 penalty (\%) & N \\",
        r"\hline",
    ]
    for row in summary.sort_values(["city", "variant"]).itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row.city)),
                    latex_escape(str(row.variant)),
                    mean_sd(row.mean_distance_penalty, row.sd_distance_penalty),
                    mean_sd(row.mean_fuel_penalty, row.sd_fuel_penalty),
                    mean_sd(row.mean_co2_penalty, row.sd_co2_penalty),
                    str(int(row.n)),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    latex_table_path: Path,
    summary: pd.DataFrame,
    overview: pd.DataFrame,
    tests: pd.DataFrame | None,
) -> tuple[Path, Path, Path | None, Path]:
    """Write CSV and LaTeX outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "penalty_summary.csv"
    overview_path = output_dir / "city_overview.csv"
    tests_path = output_dir / "statistical_tests.csv"
    summary.to_csv(summary_path, index=False)
    overview.to_csv(overview_path, index=False)
    written_tests_path: Path | None = None
    if tests is not None:
        tests.to_csv(tests_path, index=False)
        written_tests_path = tests_path
    latex_table_path.parent.mkdir(parents=True, exist_ok=True)
    latex_table_path.write_text(make_latex_table(summary), encoding="utf-8")
    return summary_path, overview_path, written_tests_path, latex_table_path


def print_console_summary(summary: pd.DataFrame, overview: pd.DataFrame) -> None:
    """Print compact summaries for validation feedback."""
    display_cols = [
        "city",
        "variant",
        "n",
        "mean_distance_penalty",
        "sd_distance_penalty",
        "mean_fuel_penalty",
        "sd_fuel_penalty",
        "mean_co2_penalty",
        "sd_co2_penalty",
    ]
    print("\nPenalty summary:")
    print(summary[display_cols].round(3).to_string(index=False))
    print("\nCity overview:")
    print(overview.round(3).to_string(index=False))


def aggregate(
    *,
    input_path: Path,
    output_dir: Path,
    latex_table_path: Path,
) -> tuple[list[str], Path, Path, Path | None, Path]:
    """Run validation, aggregation, statistical tests, and output writing."""
    df = normalize_types(load_raw_results(input_path))
    warnings = validate_experiment(df)
    summary = summarize_penalties(df)
    overview = city_overview(df)
    tests, scipy_warning = wilcoxon_tests(df)
    if scipy_warning is not None:
        warnings.append(scipy_warning)
    paths = write_outputs(
        output_dir=output_dir,
        latex_table_path=latex_table_path,
        summary=summary,
        overview=overview,
        tests=tests,
    )
    print_console_summary(summary, overview)
    return warnings, *paths


def default_output_dir(input_path: Path) -> Path:
    """Use the input file's parent directory as the default output dir."""
    return input_path.parent


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate randomized ACVRP-CO2 raw results.",
    )
    parser.add_argument(
        "--input",
        default="results/randomized/raw_results.csv",
        help="Path to raw randomized result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for penalty_summary.csv, city_overview.csv, and tests.",
    )
    parser.add_argument(
        "--latex-table",
        default=None,
        help="Path for the LaTeX summary table.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    input_path = project_path(args.input, project_root)
    output_dir = (
        project_path(args.output_dir, project_root)
        if args.output_dir is not None
        else default_output_dir(input_path)
    )
    latex_table_path = (
        project_path(args.latex_table, project_root)
        if args.latex_table is not None
        else output_dir / "randomized_results_table.tex"
    )

    warnings, summary_path, overview_path, tests_path, table_path = aggregate(
        input_path=input_path,
        output_dir=output_dir,
        latex_table_path=latex_table_path,
    )

    print("\nOutputs written:")
    print(f"  penalty summary: {summary_path}")
    print(f"  city overview:   {overview_path}")
    if tests_path is not None:
        print(f"  tests:           {tests_path}")
    print(f"  LaTeX table:     {table_path}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nWarnings: none")


if __name__ == "__main__":
    main()
