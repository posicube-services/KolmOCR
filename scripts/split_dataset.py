"""Split a grouped dataset (pdf/png/md/html) into train/eval subsets."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


EXT_DIR_MAP: Dict[str, Sequence[str]] = {
    "pdf": [".pdf"],
    "image": [".png"],
    "md": [".md"],
    "html": [".html"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split qwen235_result-like folders into train/eval with consistent base names."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root folder containing subdirectories (pdf, image, md, html) to split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of files to assign to the training subset (default 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for consistent splits.",
    )
    return parser.parse_args()


def collect_basenames(directory: Path, extensions: Iterable[str]) -> Set[str]:
    basenames: Set[str] = set()
    if not directory.is_dir():
        return basenames
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in extensions:
            basenames.add(entry.stem)
    return basenames


def gather_basenames(root: Path) -> List[str]:
    result: Set[str] = set()
    for subdir, extensions in EXT_DIR_MAP.items():
        result |= collect_basenames(root / subdir, extensions)
    return sorted(result)


def move_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.rename(dest)
    except OSError:
        shutil.copy2(src, dest)
        src.unlink()


def split_basenames(basenames: List[str], train_ratio: float, seed: int) -> Dict[str, List[str]]:
    random_generator = random.Random(seed)
    random_generator.shuffle(basenames)
    train_count = int(len(basenames) * train_ratio)
    train = basenames[:train_count]
    eval_ = basenames[train_count:]
    return {"train": train, "eval": eval_}


def relocate_files(
    root: Path, subset: str, basenames: Iterable[str], ext_map: Dict[str, Sequence[str]]
) -> None:
    for base in basenames:
        base_dest_dir = (root / subset / base)
        for subdir, extensions in ext_map.items():
            source_dir = root / subdir
            for ext in extensions:
                src = source_dir / f"{base}{ext}"
                if not src.exists():
                    continue
                move_file(src, base_dest_dir / src.name)


def main() -> None:
    args = parse_args()
    root = args.dataset_dir
    if not root.exists():
        raise SystemExit(f"Dataset directory not found: {root}")

    basenames = gather_basenames(root)
    if not basenames:
        raise SystemExit("No files found to split.")

    subsets = split_basenames(basenames, args.train_ratio, args.seed)
    for subset_name, base_list in subsets.items():
        relocate_files(root, subset_name, base_list, EXT_DIR_MAP)
        print(f"Moved {len(base_list)} basenames into {subset_name}/")

    print("Split complete.")


if __name__ == "__main__":
    main()
