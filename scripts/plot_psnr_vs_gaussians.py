#!/usr/bin/env python3
"""Create publication-ready PSNR vs Number of Gaussians plots."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

MODEL_NAMES = {
    "net": "Instant-GI",
    "net1": "Instant-GI",
    "net2": "SoftKNN-Instant-GI",
}

K_PATTERN = re.compile(r"_k(\d+)")


def discover_csvs(patterns: Iterable[str]) -> List[Path]:
    """Return all CSV files matching the provided glob patterns."""
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(path for path in Path.cwd().glob(pattern) if path.is_file())
    return sorted(paths)


def model_from_source(source: str) -> str:
    """Infer the model name (Instant-GI vs SoftKNN) from the file path."""
    for part in Path(source).parts[::-1]:
        normalized = part.lower()
        if normalized in MODEL_NAMES:
            return MODEL_NAMES[normalized]
    raise ValueError(f"Could not determine model from source: {source}")


def extract_k(csv_path: Path) -> int:
    match = K_PATTERN.search(csv_path.stem)
    if not match:
        raise ValueError(f"Filename does not contain k value: {csv_path}")
    return int(match.group(1))


def load_data(csv_files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["Model"] = df["source"].apply(model_from_source)
        df["Number of Gaussians"] = df["num_points"]
        df["PSNR (dB)"] = df["test_psnr"]
        df["k"] = extract_k(csv_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_by_k(df: pd.DataFrame) -> pd.DataFrame:
    """Average PSNR and Gaussians count for each (k, Model) pair."""
    grouped = (
        df.groupby(["k", "Model"], as_index=False)[["Number of Gaussians", "PSNR (dB)"]]
        .mean()
        .sort_values(["k", "Model"])
    )
    return grouped


def aggregate_by_gaussians(df: pd.DataFrame) -> pd.DataFrame:
    """Average PSNR for identical Gaussian counts per model."""
    return (
        df.groupby(["Model", "Number of Gaussians"], as_index=False)["PSNR (dB)"]
        .mean()
        .sort_values(["Model", "Number of Gaussians"])
    )


def make_plot(df: pd.DataFrame, output: Path, title: str, log_psnr: bool) -> None:
    plt.style.use("seaborn-v0_8-colorblind")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    hue_col = "Model"
    x_col = "Number of Gaussians"
    y_col = "PSNR (dB)"

    for model, subset in df.groupby(hue_col):
        subset = subset.sort_values(x_col)
        ax.plot(
            subset[x_col],
            subset[y_col],
            marker="o",
            linewidth=2,
            markersize=5,
            label=model,
        )

    ax.set_xlabel("Number of Gaussians")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    if log_psnr:
        ax.set_yscale("log", base=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PSNR vs Number of Gaussians for Instant-GI experiments. "
            "Use --aggregate-k to average measurements per k setting."
        )
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["imagenet1k_512_*k*.csv"],
        help="Glob patterns used to locate CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/imagenet_psnr_vs_gaussians.png"),
        help="Output image path (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        default="Instant-GI vs SoftKNN-Instant-GI (ImageNet1K, 512×512)",
        help="Figure title (default: %(default)s)",
    )
    parser.add_argument(
        "--aggregate-k",
        action="store_true",
        help="Average PSNR and Gaussians count for each k to create a minimal plot.",
    )
    parser.add_argument(
        "--log-psnr",
        action="store_true",
        help="Display PSNR axis on a base-10 log scale.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_files = discover_csvs(args.patterns)
    if not csv_files:
        parser.error("No CSV files matched the provided patterns.")

    data = load_data(csv_files)
    if args.aggregate_k:
        plot_data = aggregate_by_k(data)
    else:
        plot_data = aggregate_by_gaussians(data)

    make_plot(plot_data, args.output, args.title, args.log_psnr)
    print(f"Wrote plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
