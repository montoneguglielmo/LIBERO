#!/usr/bin/env python3

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download LIBERO dataset in LeRobot format"
    )

    parser.add_argument(
        "--repo-id",
        default="physical-intelligence/libero",
        help="Hugging Face dataset repo ID",
    )

    parser.add_argument(
        "--output-dir",
        "--outputdir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory where the dataset will be saved",
    )

    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip downloading videos",
    )

    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "*.json",
        "*.jsonl",
        "data/**/*.parquet",
    ]

    if not args.no_videos:
        allow_patterns.append("data/**/videos/**")

    print("==== LIBERO DOWNLOAD ====")
    print(f"Requested subset : {args.dataset}")
    print(f"Repo ID          : {args.repo_id}")
    print(f"Output dir       : {output_dir}")
    print(f"Videos           : {'no' if args.no_videos else 'yes'}")
    print("=========================")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )

    print("Download complete.")
    print("NOTE: Dataset subset is selected at load time, not download time.")


if __name__ == "__main__":
    main()