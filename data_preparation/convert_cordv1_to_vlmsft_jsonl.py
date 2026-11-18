#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Convert CORD-v1 parquet shards into local-vlm JSONL split files.

CORD-v1 stores the receipt image as raw bytes inside a struct column, which the
Qwen3-VL processor cannot read. This converter

* extracts every image to ``<dst>/images/<split>_<index>.<ext>``,
* serializes the CORD ground truth into the ``<s_key>...</s_key>`` /
  ``<sep/>`` target text the Qwen CORD recipes use, and
* writes one processor-native conversation per line:

    {"conversation": [
        {"role": "user", "content": [
            {"type": "image", "image": "<absolute image path>"},
            {"type": "text", "text": "Describe this image."}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "<serialized ground truth>"}]}]}

The output directory (``<dst>/train.jsonl``, ``validation.jsonl``,
``test.jsonl``) is consumed by Megatron-Bridge's ``--dataset local-vlm`` preset:

    DATASET=local-vlm ./train_qwen3_vl_sft.sh <recipe> sft "" "" \\
      dataset.source.load_kwargs.data_files.train=<dst>/train.jsonl

``local-vlm`` disables validation and test by default because the preset is
parameterized by the single ``data_files.train`` path; enabling them requires
pointing ``dataset.validation_source`` / ``dataset.test_source`` at the
corresponding files and setting ``dataset.do_validation`` / ``dataset.do_test``.

The script only depends on pyarrow (it never imports megatron.bridge), so it
runs in any container that has the dataset dependencies installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit("This program requires pyarrow. Install it with: pip install pyarrow") from exc


DEFAULT_SRC_DIR = Path("/path/to/cord-v1/data")
DEFAULT_DST_DIR = Path("/path/to/cord-v1-sft")
DEFAULT_PROMPT = "Describe this image."
SPLITS = ("train", "validation", "test")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def json2token(obj: Any, sort_json_key: bool = True) -> str:
    """Serialize one CORD ground-truth object into the Qwen target sequence.

    Mirrors ``megatron.bridge.data.token_utils.json2token`` (itself from NeMo's
    ``automodel_datasets.py``): dictionaries become ``<s_key>value</s_key>``
    (keys sorted in reverse when ``sort_json_key`` is set), lists are joined
    with ``<sep/>``, and scalars use ``str``. Keeping a local copy avoids
    importing megatron.bridge from this data-preparation step while preserving
    the exact target text produced at training time.
    """
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        return "".join(f"<s_{key}>" + json2token(obj[key], sort_json_key) + f"</s_{key}>" for key in keys)
    if isinstance(obj, list):
        return "<sep/>".join(json2token(item, sort_json_key) for item in obj)
    return str(obj)


def ground_truth_to_text(value: Any, split: str, index: int) -> str:
    """Parse one ``ground_truth`` column and serialize its ``gt_parse`` object."""
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, dict):
        parsed = value
    else:
        raise ValueError(f"{split}[{index}]: unsupported ground_truth type {type(value).__name__}")

    if isinstance(parsed, dict) and "gt_parses" in parsed:
        gt_parse = parsed["gt_parses"][0]
    elif isinstance(parsed, dict) and "gt_parse" in parsed:
        gt_parse = parsed["gt_parse"]
    else:
        raise ValueError(f"{split}[{index}]: ground_truth has neither 'gt_parses' nor 'gt_parse'")
    return json2token(gt_parse, sort_json_key=True)


def extract_image(example: dict[str, Any], split: str, index: int, image_dir: Path) -> str:
    """Write one embedded image to disk and return its absolute path."""
    image = example.get("image")
    if not isinstance(image, dict) or image.get("bytes") is None:
        raise ValueError(f"{split}[{index}]: expected an image struct with non-empty 'bytes'")
    payload = image["bytes"]
    suffix = ".png" if payload.startswith(PNG_MAGIC) else ".jpg"
    image_path = image_dir / f"{split}_{index:06d}{suffix}"
    image_path.write_bytes(payload)
    return str(image_path)


def parquet_rows(parquet_files: list[Path], batch_size: int) -> Iterator[dict[str, Any]]:
    """Yield rows from parquet shards in deterministic filename order."""
    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=True):
            for row in batch.to_pylist():
                yield row


def convert_split(
    src_dir: Path,
    dst_dir: Path,
    split: str,
    *,
    prompt: str,
    batch_size: int,
    overwrite: bool,
) -> int:
    """Convert one split and return the number of rows written."""
    parquet_files = sorted(src_dir.glob(f"{split}-*.parquet"))
    if not parquet_files:
        print(f"[skip] {split}: no {split}-*.parquet under {src_dir}")
        return 0

    output_file = dst_dir / f"{split}.jsonl"
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}\nUse --overwrite to replace it.")

    image_dir = dst_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as destination:
        for example in parquet_rows(parquet_files, batch_size):
            image_path = extract_image(example, split, row_count, image_dir)
            target_text = ground_truth_to_text(example.get("ground_truth"), split, row_count)
            row = {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": target_text},
                        ],
                    },
                ]
            }
            destination.write(json.dumps(row, ensure_ascii=False))
            destination.write("\n")
            row_count += 1

    print(f"{split}: {row_count:,} rows -> {output_file}")
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CORD-v1 parquet shards to local-vlm conversation JSONL."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help=f"Directory with the cord-v1 parquet shards (default: {DEFAULT_SRC_DIR}).",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=DEFAULT_DST_DIR,
        help=f"Output directory for JSONL files and images (default: {DEFAULT_DST_DIR}).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"User prompt paired with every image (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help=f"Splits to convert (default: {' '.join(SPLITS)}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Parquet rows read per batch (default: 64).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing <split>.jsonl outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = args.src_dir.expanduser().resolve()
    dst_dir = args.dst_dir.expanduser().resolve()

    if not src_dir.is_dir():
        print(f"Error: source directory not found: {src_dir}", file=sys.stderr)
        return 1
    if dst_dir.exists() and not dst_dir.is_dir():
        print(f"Error: output path is not a directory: {dst_dir}", file=sys.stderr)
        return 1
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    try:
        for split in args.splits:
            total += convert_split(
                src_dir,
                dst_dir,
                split,
                prompt=args.prompt,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Wrote {total:,} local-vlm conversations under {dst_dir}")
    print("Run SFT with:")
    print('  DATASET=local-vlm ./train_qwen3_vl_sft.sh <RECIPE> sft "" "" \\')
    print(f"    dataset.source.load_kwargs.data_files.train={dst_dir / 'train.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
