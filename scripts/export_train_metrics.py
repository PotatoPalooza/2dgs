#!/usr/bin/env python3
"""Aggregate Instant-GI training summaries into a CSV file."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

# Pre-compiled regex matchers for the non-trivial lines.
SIZE_RE = re.compile(r"(?P<width>\d+)\s*x\s*(?P<height>\d+)")
PSNR_RE = re.compile(
    r"Test\s+PSNR:(?P<psnr>[-+]?\d*\.?\d+),\s*MS_SSIM:(?P<ms_ssim>[-+]?\d*\.?\d+)"
)
TRAIN_RE = re.compile(
    r"Training\s+Complete\s+in\s+(?P<seconds>[-+]?\d*\.?\d+)s,\s*FPS:(?P<fps>[-+]?\d*\.?\d+)"
)


def parse_train_file(train_file: Path) -> Dict[str, object]:
    """Parse a single train.txt file into a dict of metrics."""
    metrics: Dict[str, object] = {
        "source": str(train_file),
        "source_model": train_file.parent.name,
    }
    lines = [line.strip() for line in train_file.read_text().splitlines() if line.strip()]

    for line in lines:
        if line.startswith("Image name:"):
            metrics["image_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Image size:"):
            match = SIZE_RE.search(line.split(":", 1)[1])
            if not match:
                raise ValueError(f"Could not parse image size in {train_file}")
            metrics["width"] = int(match.group("width"))
            metrics["height"] = int(match.group("height"))
        elif line.startswith("Number of points:"):
            metrics["num_points"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Test PSNR:"):
            match = PSNR_RE.search(line)
            if not match:
                raise ValueError(f"Could not parse PSNR/MS-SSIM in {train_file}")
            metrics["test_psnr"] = float(match.group("psnr"))
            metrics["ms_ssim"] = float(match.group("ms_ssim"))
        elif line.startswith("Training Complete"):
            match = TRAIN_RE.search(line)
            if not match:
                raise ValueError(f"Could not parse training speed in {train_file}")
            metrics["train_seconds"] = float(match.group("seconds"))
            metrics["fps"] = float(match.group("fps"))

    required_fields = [
        "image_name",
        "width",
        "height",
        "num_points",
        "test_psnr",
        "ms_ssim",
        "train_seconds",
        "fps",
    ]
    missing = [field for field in required_fields if field not in metrics]
    if missing:
        raise ValueError(f"{train_file} is missing fields: {', '.join(missing)}")
    return metrics


def iter_train_files(root: Path) -> Iterable[Path]:
    """Yield every train.txt file below the provided root."""
    yield from root.rglob("train.txt")


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write the aggregated rows to disk."""
    if not rows:
        raise SystemExit("No train.txt files were found; aborting.")

    fieldnames = [
        "image_name",
        "width",
        "height",
        "num_points",
        "test_psnr",
        "ms_ssim",
        "train_seconds",
        "fps",
        "source_model",
        "source",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Instant-GI train.txt metrics into a CSV file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Instant-GI/output"),
        help="Root directory to scan for train.txt files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("train_metrics.csv"),
        help="Output CSV path (default: %(default)s)",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = args.root
    if not root.exists():
        parser.error(f"Root directory {root} does not exist.")

    rows: List[Dict[str, object]] = []
    for train_file in iter_train_files(root):
        try:
            rows.append(parse_train_file(train_file))
        except Exception as exc:  # noqa: BLE001 - surface parsing issues to user
            print(f"Skipping {train_file}: {exc}", file=sys.stderr)

    if not rows:
        raise SystemExit("No valid train.txt files were parsed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
