"""
make_figures.py
===============

Generate publication-quality figures for the ACVRP-CO2 paper from saved
result files. This script does not rerun experiments.

Outputs, by default:
    figures/fig1_pipeline.pdf
    figures/fig1_pipeline.png
    figures/fig2_fixed_penalty.pdf
    figures/fig2_fixed_penalty.png
    figures/fig3_randomized_boxplot.pdf
    figures/fig3_randomized_boxplot.png
    figures/fig4_solver_heatmap.pdf
    figures/fig4_solver_heatmap.png

Usage:
    python -m src.make_figures
    python -m src.make_figures --output-dir figures
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd


VARIANTS = ["SE", "SM", "SR"]
CITY_LABELS = {"macau": "Macau", "hongkong": "Hong Kong"}
CITY_COLORS = {"macau": "#2F6B8F", "hongkong": "#B55A30"}
VARIANT_COLORS = {"SE": "#D98C00", "SM": "#4C97C9", "SR": "#2C8C63"}
SOLVERS = ["OR-Tools", "GA", "MatNet-CVRP", "Vanilla-AM"]


@dataclass
class FigureRunReport:
    """Track generated figures, skipped figures, and warnings."""

    generated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_generated_pair(self, base_path: Path) -> None:
        """Record PDF/PNG paths for a generated figure base path."""
        self.generated.append(str(base_path.with_suffix(".pdf")))
        self.generated.append(str(base_path.with_suffix(".png")))


def project_path(raw_path: str | Path, project_root: Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def configure_matplotlib() -> None:
    """Set paper-friendly matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_if_exists(
    path: Path,
    figure_name: str,
    report: FigureRunReport,
    required_columns: Iterable[str],
) -> pd.DataFrame | None:
    """Read a CSV or record a skip/warning if missing or incompatible."""
    if not path.exists():
        report.skipped.append(f"{figure_name}: missing input {path}")
        return None
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        report.skipped.append(
            f"{figure_name}: {path} missing columns {sorted(missing)}"
        )
        return None
    return df


def save_figure(fig: plt.Figure, base_path: Path, report: FigureRunReport) -> None:
    """Save a figure as PDF and PNG."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    report.add_generated_pair(base_path)


def make_pipeline_figure(output_dir: Path, report: FigureRunReport) -> None:
    """Create Figure 1: experiment pipeline diagram."""
    nodes = [
        ("OpenStreetMap\nroad graph", 0.17, 0.68),
        ("Depot + customer\ninstances", 0.50, 0.68),
        ("SE / SM / SR / AR\nmatrices", 0.83, 0.68),
        ("Routing\nsolvers", 0.83, 0.28),
        ("AR\nre-evaluation", 0.50, 0.28),
        ("Distance / fuel /\nCO$_2$ penalties", 0.17, 0.28),
    ]
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_axis_off()

    box_w = 0.25
    box_h = 0.20
    colors = ["#E8EEF2", "#F3E6D8", "#E9F0DF", "#ECE7F2", "#E3EEF0", "#F2E5E5"]

    for idx, (label, x, y) in enumerate(nodes):
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=0.9,
            edgecolor="#333333",
            facecolor=colors[idx],
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            linespacing=1.2,
            fontsize=8.5,
        )

    for start_idx, end_idx in arrows:
        _, x0, y0 = nodes[start_idx]
        _, x1, y1 = nodes[end_idx]
        if y0 == y1 and x1 > x0:
            start = (x0 + box_w / 2 + 0.014, y0)
            end = (x1 - box_w / 2 - 0.014, y1)
        elif y0 == y1 and x1 < x0:
            start = (x0 - box_w / 2 - 0.014, y0)
            end = (x1 + box_w / 2 + 0.014, y1)
        else:
            start = (x0, y0 - box_h / 2 - 0.014)
            end = (x1, y1 + box_h / 2 + 0.014)
        arrow = FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#333333",
        )
        ax.add_patch(arrow)

    save_figure(fig, output_dir / "fig1_pipeline", report)


def fixed_co2_penalties(summary_path: Path, city: str) -> pd.DataFrame:
    """Compute fixed-case CO2 penalties relative to AR for one city."""
    df = pd.read_csv(summary_path)
    required = {"variant", "co2_kg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{summary_path} missing columns {sorted(missing)}")
    ar_rows = df[df["variant"] == "AR"]
    if len(ar_rows) != 1:
        raise ValueError(f"{summary_path} must contain exactly one AR row.")
    ar_co2 = float(ar_rows.iloc[0]["co2_kg"])
    out = df[df["variant"].isin(VARIANTS)].copy()
    out["city"] = city
    out["co2_penalty_pct"] = 100.0 * (out["co2_kg"] - ar_co2) / ar_co2
    return out[["city", "variant", "co2_penalty_pct"]]


def make_fixed_penalty_figure(
    macau_summary: Path,
    hongkong_summary: Path,
    output_dir: Path,
    report: FigureRunReport,
) -> None:
    """Create Figure 2: fixed-case OR-Tools CO2 penalties."""
    figure_name = "fig2_fixed_penalty"
    if not macau_summary.exists() or not hongkong_summary.exists():
        report.skipped.append(
            f"{figure_name}: missing fixed summary input(s): "
            f"{macau_summary}, {hongkong_summary}"
        )
        return
    try:
        data = pd.concat(
            [
                fixed_co2_penalties(macau_summary, "macau"),
                fixed_co2_penalties(hongkong_summary, "hongkong"),
            ],
            ignore_index=True,
        )
    except Exception as err:
        report.skipped.append(f"{figure_name}: incompatible input ({err})")
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(VARIANTS))
    width = 0.34
    for offset, city in [(-width / 2, "macau"), (width / 2, "hongkong")]:
        vals = [
            float(
                data[(data["city"] == city) & (data["variant"] == variant)][
                    "co2_penalty_pct"
                ].iloc[0]
            )
            for variant in VARIANTS
        ]
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=CITY_COLORS[city],
            edgecolor="#333333",
            linewidth=0.5,
            label=CITY_LABELS[city],
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in vals],
            padding=2,
            fontsize=7.5,
        )

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("AR-evaluated CO$_2$ penalty (%)")
    ax.set_xlabel("Planning matrix")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / figure_name, report)


def make_randomized_boxplot(
    randomized_raw: Path,
    output_dir: Path,
    report: FigureRunReport,
) -> None:
    """Create Figure 3: randomized CO2 penalty distributions."""
    figure_name = "fig3_randomized_boxplot"
    df = read_csv_if_exists(
        randomized_raw,
        figure_name,
        report,
        required_columns=["city", "variant", "co2_penalty_pct"],
    )
    if df is None:
        return
    data = df[df["variant"].isin(VARIANTS)].copy()
    if data.empty:
        report.skipped.append(f"{figure_name}: no SE/SM/SR rows in {randomized_raw}")
        return

    fig, ax = plt.subplots(figsize=(5.8, 3.35))
    base_positions = np.arange(len(VARIANTS))
    offsets = {"macau": -0.18, "hongkong": 0.18}
    width = 0.28

    for city in ["macau", "hongkong"]:
        box_data = [
            data[(data["city"] == city) & (data["variant"] == variant)][
                "co2_penalty_pct"
            ].dropna()
            for variant in VARIANTS
        ]
        if any(series.empty for series in box_data):
            report.skipped.append(
                f"{figure_name}: missing city/variant data for {CITY_LABELS[city]}"
            )
            plt.close(fig)
            return
        positions = base_positions + offsets[city]
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#111111", "linewidth": 1.1},
            boxprops={"linewidth": 0.8, "edgecolor": "#333333"},
            whiskerprops={"linewidth": 0.8, "color": "#333333"},
            capprops={"linewidth": 0.8, "color": "#333333"},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "markerfacecolor": CITY_COLORS[city],
                "markeredgecolor": "none",
                "alpha": 0.55,
            },
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(CITY_COLORS[city])
            patch.set_alpha(0.72)

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(VARIANTS)
    ax.set_xlabel("Planning matrix")
    ax.set_ylabel("AR-evaluated CO$_2$ penalty (%)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(
        handles=[
            Patch(facecolor=CITY_COLORS["macau"], label="Macau", alpha=0.72),
            Patch(facecolor=CITY_COLORS["hongkong"], label="Hong Kong", alpha=0.72),
        ],
        frameon=False,
        loc="upper left",
    )
    save_figure(fig, output_dir / figure_name, report)


def solver_distance_penalty_matrix(summary_full: Path) -> np.ndarray:
    """Compute solver-by-matrix fixed-case distance penalties."""
    df = pd.read_csv(summary_full)
    required = {"solver", "variant", "distance_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{summary_full} missing columns {sorted(missing)}")

    matrix = np.full((len(SOLVERS), len(VARIANTS)), np.nan, dtype=float)
    for row_idx, solver in enumerate(SOLVERS):
        solver_rows = df[df["solver"] == solver]
        if solver_rows.empty:
            raise ValueError(f"{summary_full} missing solver {solver!r}")
        ar_rows = solver_rows[solver_rows["variant"] == "AR"]
        if len(ar_rows) != 1:
            raise ValueError(
                f"{summary_full} must contain exactly one AR row for {solver!r}"
            )
        ar_distance = float(ar_rows.iloc[0]["distance_m"])
        for col_idx, variant in enumerate(VARIANTS):
            variant_rows = solver_rows[solver_rows["variant"] == variant]
            if len(variant_rows) != 1:
                raise ValueError(
                    f"{summary_full} must contain exactly one {variant} row "
                    f"for {solver!r}"
                )
            value = float(variant_rows.iloc[0]["distance_m"])
            matrix[row_idx, col_idx] = 100.0 * (value - ar_distance) / ar_distance
    return matrix


def annotate_heatmap(ax: plt.Axes, values: np.ndarray) -> None:
    """Annotate heatmap cells with one-decimal values."""
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                label = ""
            else:
                label = f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#111111")


def make_solver_heatmap(
    macau_full: Path,
    hongkong_full: Path,
    output_dir: Path,
    report: FigureRunReport,
) -> None:
    """Create Figure 4: solver-by-matrix fixed-case distance heatmap."""
    figure_name = "fig4_solver_heatmap"
    if not macau_full.exists() or not hongkong_full.exists():
        report.skipped.append(
            f"{figure_name}: missing full summary input(s): {macau_full}, {hongkong_full}"
        )
        return
    try:
        matrices = {
            "Macau": solver_distance_penalty_matrix(macau_full),
            "Hong Kong": solver_distance_penalty_matrix(hongkong_full),
        }
    except Exception as err:
        report.skipped.append(f"{figure_name}: incompatible input ({err})")
        return

    vmin = -5.0
    vmax = 45.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    image = None
    for ax, (city_label, values) in zip(axes, matrices.items()):
        image = ax.imshow(values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        annotate_heatmap(ax, values)
        ax.text(
            0.02,
            0.98,
            city_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax.set_xticks(np.arange(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS)
        ax.set_yticks(np.arange(len(SOLVERS)))
        ax.set_yticklabels(SOLVERS)
        ax.set_xlabel("Planning matrix")
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(VARIANTS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SOLVERS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Solver")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.03)
        cbar.set_label("AR-evaluated distance penalty (%)")
    save_figure(fig, output_dir / figure_name, report)


def run(args: argparse.Namespace) -> FigureRunReport:
    """Generate all requested figures from existing result files."""
    project_root = Path.cwd()
    output_dir = project_path(args.output_dir, project_root)
    randomized_raw = project_path(args.randomized_raw, project_root)
    _ = project_path(args.randomized_summary, project_root)
    macau_summary = project_path(args.macau_summary, project_root)
    hongkong_summary = project_path(args.hongkong_summary, project_root)
    macau_full = project_path(args.macau_full, project_root)
    hongkong_full = project_path(args.hongkong_full, project_root)

    configure_matplotlib()
    report = FigureRunReport()
    output_dir.mkdir(parents=True, exist_ok=True)

    make_pipeline_figure(output_dir, report)
    make_fixed_penalty_figure(macau_summary, hongkong_summary, output_dir, report)
    make_randomized_boxplot(randomized_raw, output_dir, report)
    make_solver_heatmap(macau_full, hongkong_full, output_dir, report)
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ACVRP-CO2 publication figures from saved results.",
    )
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--randomized-raw", default="results/randomized/raw_results.csv")
    parser.add_argument(
        "--randomized-summary",
        default="results/randomized/penalty_summary.csv",
    )
    parser.add_argument("--macau-summary", default="results_macau/summary.csv")
    parser.add_argument("--hongkong-summary", default="results_hongkong/summary.csv")
    parser.add_argument("--macau-full", default="results_macau/summary_full.csv")
    parser.add_argument("--hongkong-full", default="results_hongkong/summary_full.csv")
    args = parser.parse_args()

    report = run(args)
    print("Generated figures:")
    if report.generated:
        for path in report.generated:
            print(f"  {path}")
    else:
        print("  none")

    print("\nSkipped figures:")
    if report.skipped:
        for item in report.skipped:
            print(f"  {item}")
    else:
        print("  none")

    print("\nWarnings:")
    if report.warnings:
        for warning in report.warnings:
            print(f"  {warning}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
