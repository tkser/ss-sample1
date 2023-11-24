import argparse
from pathlib import Path


def fix_label(label_glob: str) -> None:
    current_dir = Path.cwd()
    label_paths = current_dir.glob(label_glob)
    for label_path in label_paths:
        if "_" in label_path.stem:
            label_path.rename(label_path.parent / (label_path.stem.split("_")[1] + label_path.suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label_glob", type=str)
    args = parser.parse_args()

    fix_label(args.label_glob)
