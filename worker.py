#!/usr/bin/env python3
"""Process one deterministic block of Stanford Dogs images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CSV_FIELDS = [
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


def block_for_task(paths: list[Path], index: int, total_tasks: int) -> list[Path]:
    if total_tasks < 1:
        raise ValueError("total_tasks debe ser mayor que cero.")
    if index < 0 or index >= total_tasks:
        raise ValueError(f"index debe estar entre 0 y {total_tasks - 1}.")
    start = len(paths) * index // total_tasks
    end = len(paths) * (index + 1) // total_tasks
    return paths[start:end]


def process_image(
    source: Path,
    input_dir: Path,
    edge_dir: Path,
    threshold: int,
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV no pudo decodificar la imagen.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)[1]
    edges = edges.astype(np.uint8)

    relative = source.relative_to(input_dir).with_suffix(".png")
    target = edge_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), edges):
        raise OSError(f"No se pudo escribir la imagen procesada: {target}")

    height, width = edges.shape
    duration = time.perf_counter() - started
    row: dict[str, object] = {
        "source_bytes": source.stat().st_size,
        "width": width,
        "height": height,
        "process_ms": round(duration * 1_000, 3),
        "edge_ratio": round(cv2.countNonZero(edges) / edges.size, 6),
        "status": "ok",
        "error": "",
        "output": str(Path("edges") / relative),
    }
    return row, duration


def write_part(
    output_dir: Path,
    index: int,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"part-{index:04d}"
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_worker(
    index: int,
    total_tasks: int,
    input_dir: Path,
    output_dir: Path,
    threshold: int,
) -> dict[str, object]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {input_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    list_started = time.perf_counter()
    images = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    list_seconds = time.perf_counter() - list_started
    assigned = block_for_task(images, index, total_tasks)

    rows: list[dict[str, object]] = []
    process_seconds = 0.0
    failed = 0
    for source in assigned:
        try:
            result, duration = process_image(source, input_dir, output_dir / "edges", threshold)
            process_seconds += duration
            print(f"OK {source.relative_to(input_dir)} ({result['process_ms']} ms)")
        except (OSError, ValueError, cv2.error) as exc:
            failed += 1
            try:
                source_bytes = source.stat().st_size
            except OSError:
                source_bytes = 0
            result = {
                "source_bytes": source_bytes,
                "width": "",
                "height": "",
                "process_ms": 0,
                "edge_ratio": "",
                "status": "error",
                "error": str(exc),
                "output": "",
            }
            print(f"ERROR {source}: {exc}", file=sys.stderr)

        rows.append(
            {
                "index": index,
                "total_tasks": total_tasks,
                "strategy": "block",
                "image": source.relative_to(input_dir).as_posix(),
                **result,
            }
        )

    worker_seconds = time.perf_counter() - list_started
    metadata: dict[str, object] = {
        "index": index,
        "total_tasks": total_tasks,
        "strategy": "block",
        "node": os.environ.get("NODE_NAME", socket.gethostname()),
        "assigned": len(assigned),
        "processed": len(assigned) - failed,
        "failed": failed,
        "list_seconds": round(list_seconds, 6),
        "process_seconds": round(process_seconds, 6),
        "worker_seconds": round(worker_seconds, 6),
        "started_at": started_at,
    }
    write_part(output_dir, index, rows, metadata)
    print(
        f"Índice {index}: {len(assigned) - failed}/{len(assigned)} procesadas; "
        f"partes guardadas en {output_dir}"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=int,
        default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")),
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=int(os.environ.get("TOTAL_TASKS", "1")),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "datos/entrada/stanforddogs")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "datos/salida/demo")),
    )
    parser.add_argument("--threshold", type=int, default=80)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold debe estar entre 0 y 255.")

    try:
        run_worker(
            args.index,
            args.total_tasks,
            args.input_dir,
            args.output_dir,
            args.threshold,
        )
    except (OSError, ValueError) as exc:
        print(f"Error en worker: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
