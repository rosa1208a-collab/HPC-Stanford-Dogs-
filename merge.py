#!/usr/bin/env python3
"""Combine worker part files and create a small edge-image mosaic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageOps

RESULT_FIELDS = [
    "index",
    "total_tasks",
    "strategy",
    "image",
    "source_bytes",
    "width",
    "height",
    "process_ms",
    "edge_ratio",
    "status",
    "error",
    "output",
]


def load_parts(results_dir: Path, expected_tasks: int | None) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    csv_paths = sorted(results_dir.glob("part-*.csv"))
    json_paths = sorted(results_dir.glob("part-*.json"))
    if expected_tasks is not None:
        expected = {f"part-{index:04d}" for index in range(expected_tasks)}
        actual = {path.stem for path in csv_paths}
        missing = sorted(expected - actual)
        actual_metadata = {path.stem for path in json_paths}
        missing_metadata = sorted(expected - actual_metadata)
        if missing:
            raise FileNotFoundError(f"Faltan archivos de resultados: {', '.join(missing)}")
        if missing_metadata:
            raise FileNotFoundError(
                f"Faltan metadatos de resultados: {', '.join(missing_metadata)}"
            )
    if not csv_paths:
        raise FileNotFoundError(f"No se encontraron partes en {results_dir}")

    rows: list[dict[str, str]] = []
    for part in csv_paths:
        with part.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in json_paths
    ]
    return rows, metadata


def create_mosaic(rows: list[dict[str, str]], results_dir: Path, target: Path, limit: int) -> int:
    images: list[Image.Image] = []
    for row in rows:
        if row["status"] != "ok" or not row["output"]:
            continue
        path = results_dir / row["output"]
        if path.is_file():
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if len(images) == limit:
            break
    if not images:
        return 0

    tile_width = 160
    tile_height = 160
    columns = min(5, len(images))
    rows_count = math.ceil(len(images) / columns)
    mosaic = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
    for index, image in enumerate(images):
        tile = ImageOps.contain(image, (tile_width - 8, tile_height - 8))
        x = (index % columns) * tile_width + (tile_width - tile.width) // 2
        y = (index // columns) * tile_height + (tile_height - tile.height) // 2
        mosaic.paste(tile, (x, y))
    target.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(target)
    return len(images)


def merge(results_dir: Path, expected_tasks: int | None, mosaic_count: int) -> dict[str, object]:
    rows, metadata = load_parts(results_dir, expected_tasks)
    rows.sort(key=lambda row: (int(row["index"]), row["image"]))
    with (results_dir / "resultados.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    worker_times = [float(part["worker_seconds"]) for part in metadata]
    average_time = statistics.mean(worker_times) if worker_times else 0.0
    imbalance = max(worker_times) / average_time if average_time else None
    image_count = sum(row["status"] == "ok" for row in rows)
    failure_count = sum(row["status"] == "error" for row in rows)
    mosaic_path = results_dir / "mosaico.png"
    mosaic_images = create_mosaic(rows, results_dir, mosaic_path, mosaic_count)

    summary: dict[str, object] = {
        "task_count": len(metadata),
        "image_rows": len(rows),
        "processed": image_count,
        "failed": failure_count,
        "load_imbalance": round(imbalance, 6) if imbalance is not None else None,
        "worker_times_seconds": worker_times,
        "mosaic": str(mosaic_path.name) if mosaic_images else None,
        "mosaic_images": mosaic_images,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("datos/salida/demo"))
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--mosaic-count", type=int, default=25)
    args = parser.parse_args()
    if args.expected_tasks is not None and args.expected_tasks < 1:
        parser.error("--expected-tasks debe ser mayor que cero.")
    if args.mosaic_count < 1:
        parser.error("--mosaic-count debe ser mayor que cero.")
    try:
        summary = merge(args.results_dir, args.expected_tasks, args.mosaic_count)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error al unir resultados: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
