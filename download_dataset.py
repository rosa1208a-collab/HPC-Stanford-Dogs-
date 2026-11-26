#!/usr/bin/env python3
"""Stream Stanford Dogs and save a reproducible 5,000-image sample."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

DATASET_URL = "https://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
DEFAULT_COUNT = 5_000
DEFAULT_SEED = 1208


def image_report(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        return image.width, image.height


def download_sample(project_dir: Path, count: int, seed: int) -> dict[str, object]:
    input_root = project_dir / "datos" / "entrada"
    input_root.mkdir(parents=True, exist_ok=True)
    destination = input_root / "stanforddogs"
    if destination.exists():
        raise FileExistsError(
            f"{destination} ya existe; no se sobrescribirá. "
            "Mueve o renombra esa carpeta si quieres volver a descargar."
        )

    staging = Path(tempfile.mkdtemp(prefix=".stanforddogs-", dir=input_root))
    selected: dict[int, dict[str, object]] = {}
    chooser = random.Random(seed)
    seen = 0

    try:
        request = urllib.request.Request(
            DATASET_URL, headers={"User-Agent": "stanford-dogs-lab/1.0"}
        )
        print(f"Descargando y seleccionando {count:,} imágenes desde Stanford Dogs...")
        with urllib.request.urlopen(request, timeout=60) as response:
            with tarfile.open(fileobj=response, mode="r|") as archive:
                for member in archive:
                    if not member.isfile() or Path(member.name).suffix.lower() not in {
                        ".jpg",
                        ".jpeg",
                    }:
                        continue

                    current = seen
                    seen += 1
                    slot = current if current < count else chooser.randrange(current + 1)
                    if slot >= count:
                        continue

                    source = archive.extractfile(member)
                    if source is None:
                        raise OSError(f"No se pudo leer {member.name} del archivo.")
                    with source, (staging / f"{slot:05d}.jpg").open("wb") as target:
                        shutil.copyfileobj(source, target)
                    selected[slot] = {
                        "source": member.name,
                        "bytes": member.size,
                    }

                    if seen % 2_000 == 0:
                        print(f"  {seen:,} imágenes revisadas...")

        if seen < count or len(selected) != count:
            raise RuntimeError(
                f"El archivo contiene {seen:,} imágenes; se requieren {count:,}."
            )

        image_rows: list[dict[str, object]] = []
        for index in range(count):
            image_path = staging / f"{index:05d}.jpg"
            width, height = image_report(image_path)
            image_rows.append(
                {
                    "file": image_path.name,
                    "source": selected[index]["source"],
                    "bytes": image_path.stat().st_size,
                    "width": width,
                    "height": height,
                }
            )

        total_bytes = sum(int(row["bytes"]) for row in image_rows)
        report: dict[str, object] = {
            "dataset": "Stanford Dogs",
            "source_url": DATASET_URL,
            "seed": seed,
            "selected_images": count,
            "archive_images_seen": seen,
            "total_bytes": total_bytes,
            "average_bytes": total_bytes / count,
            "min_bytes": min(int(row["bytes"]) for row in image_rows),
            "max_bytes": max(int(row["bytes"]) for row in image_rows),
            "min_width": min(int(row["width"]) for row in image_rows),
            "max_width": max(int(row["width"]) for row in image_rows),
            "min_height": min(int(row["height"]) for row in image_rows),
            "max_height": max(int(row["height"]) for row in image_rows),
            "images": image_rows,
        }

        staging.rename(destination)
        report_path = project_dir / "datos" / "dataset_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        env_path = project_dir / "datos" / "dataset.env"
        env_path.write_text(
            "DATASET_NAME=stanforddogs\n"
            "DATASET_DIR=datos/entrada/stanforddogs\n"
            f"DATASET_IMAGES={count}\n",
            encoding="utf-8",
        )
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count debe ser mayor que cero.")

    try:
        report = download_sample(args.project_dir, args.count, args.seed)
    except Exception as exc:
        print(f"Error al descargar Stanford Dogs: {exc}", file=sys.stderr)
        return 1

    print(
        f"Listo: {report['selected_images']:,} imágenes, "
        f"{report['total_bytes']:,} bytes en datos/entrada/stanforddogs."
    )
    print("Revisa datos/dataset_report.json para tamaños y dimensiones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
