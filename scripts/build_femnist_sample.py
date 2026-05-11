import argparse
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def relabel_class(class_code: str) -> int:
    if class_code.isdigit() and int(class_code) < 40:
        return int(class_code) - 30
    if int(class_code, 16) <= 90:
        return int(class_code, 16) - 55
    return int(class_code, 16) - 61


def normalize_image(image_bytes: bytes) -> list[float]:
    image = Image.open(BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.flatten().tolist()


def collect_writer_samples(
    by_write_zip: Path, num_writers: int, max_images_per_writer: int
) -> tuple[dict[str, list[dict]], set[str]]:
    writer_samples: dict[str, list[dict]] = defaultdict(list)
    target_hashes: set[str] = set()

    with zipfile.ZipFile(by_write_zip) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.endswith(".png"):
                continue

            parts = entry.filename.split("/")
            if len(parts) < 5 or parts[0] != "by_write":
                continue

            writer_id = parts[2]
            if writer_id not in writer_samples and len(writer_samples) >= num_writers:
                continue
            if len(writer_samples[writer_id]) >= max_images_per_writer:
                continue

            image_bytes = archive.read(entry)
            image_hash = hashlib.md5(image_bytes).hexdigest()
            writer_samples[writer_id].append(
                {
                    "hash": image_hash,
                    "bytes": image_bytes,
                    "path": entry.filename,
                }
            )
            target_hashes.add(image_hash)

            if len(writer_samples) >= num_writers and all(
                len(samples) >= max_images_per_writer
                for samples in writer_samples.values()
            ):
                break

    return dict(writer_samples), target_hashes


def map_hashes_to_classes(by_class_zip: Path, target_hashes: set[str]) -> dict[str, str]:
    hash_to_class: dict[str, str] = {}
    processed = 0

    with zipfile.ZipFile(by_class_zip) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.endswith(".png"):
                continue

            image_bytes = archive.read(entry)
            image_hash = hashlib.md5(image_bytes).hexdigest()
            if image_hash in target_hashes:
                parts = entry.filename.split("/")
                if len(parts) >= 2:
                    hash_to_class[image_hash] = parts[1]
                    if len(hash_to_class) == len(target_hashes):
                        break

            processed += 1
            if processed % 50000 == 0:
                print(
                    f"Scanned {processed} by_class images, matched "
                    f"{len(hash_to_class)}/{len(target_hashes)} target hashes..."
                )

    return hash_to_class


def build_split(
    writer_samples: dict[str, list[dict]],
    hash_to_class: dict[str, str],
    train_frac: float,
    seed: int,
) -> tuple[dict, dict]:
    rng = random.Random(seed)
    train = {"users": [], "num_samples": [], "user_data": {}}
    test = {"users": [], "num_samples": [], "user_data": {}}

    for writer_id, samples in writer_samples.items():
        labeled = []
        for sample in samples:
            class_code = hash_to_class.get(sample["hash"])
            if class_code is None:
                continue
            labeled.append(
                {
                    "x": normalize_image(sample["bytes"]),
                    "y": relabel_class(class_code),
                }
            )

        if len(labeled) < 2:
            continue

        rng.shuffle(labeled)
        split_idx = max(1, min(len(labeled) - 1, int(len(labeled) * train_frac)))
        train_samples = labeled[:split_idx]
        test_samples = labeled[split_idx:]

        train["users"].append(writer_id)
        train["num_samples"].append(len(train_samples))
        train["user_data"][writer_id] = {
            "x": [item["x"] for item in train_samples],
            "y": [item["y"] for item in train_samples],
        }

        test["users"].append(writer_id)
        test["num_samples"].append(len(test_samples))
        test["user_data"][writer_id] = {
            "x": [item["x"] for item in test_samples],
            "y": [item["y"] for item in test_samples],
        }

    return train, test


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-write-zip", required=True)
    parser.add_argument("--by-class-zip", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-writers", type=int, default=10)
    parser.add_argument("--max-images-per-writer", type=int, default=40)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_write_zip = Path(args.by_write_zip)
    by_class_zip = Path(args.by_class_zip)
    out_dir = Path(args.out_dir)

    writer_samples, target_hashes = collect_writer_samples(
        by_write_zip=by_write_zip,
        num_writers=args.num_writers,
        max_images_per_writer=args.max_images_per_writer,
    )
    print(
        f"Collected {sum(len(v) for v in writer_samples.values())} images from "
        f"{len(writer_samples)} writers."
    )

    hash_to_class = map_hashes_to_classes(by_class_zip, target_hashes)
    print(f"Matched {len(hash_to_class)} of {len(target_hashes)} selected images.")

    train, test = build_split(
        writer_samples=writer_samples,
        hash_to_class=hash_to_class,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    train_path = out_dir / "train" / "sampled_small_train.json"
    test_path = out_dir / "test" / "sampled_small_test.json"
    write_json(train, train_path)
    write_json(test, test_path)

    summary = {
        "writers_requested": args.num_writers,
        "writers_kept": len(train["users"]),
        "max_images_per_writer": args.max_images_per_writer,
        "train_samples": sum(train["num_samples"]),
        "test_samples": sum(test["num_samples"]),
    }
    write_json(summary, out_dir / "sample_summary.json")

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
