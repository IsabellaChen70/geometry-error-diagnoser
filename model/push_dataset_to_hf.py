"""Upload the v6 transform-diagnosis dataset to the Hugging Face Hub.

This script only PREPARES and (when explicitly asked) performs an upload. It
never hardcodes a token: the token is read from the ``HF_TOKEN`` environment
variable. Nothing is uploaded unless you pass a real ``--repo-id`` and omit
``--dry-run``.

Examples
--------
Dry run (no token, no network) -- lists exactly what would be uploaded::

    python3 model/push_dataset_to_hf.py --dry-run

Real upload of the larger public sample::

    export HF_TOKEN=hf_xxx
    python3 model/push_dataset_to_hf.py --repo-id your-username/geometry-transform-diagnosis-v6

Upload a different local tree (e.g. a cluster-pulled full v6 curriculum)::

    python3 model/push_dataset_to_hf.py \
        --repo-id your-username/geometry-transform-diagnosis-v6 \
        --path ~/transform_diagnosis_data_v6
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

PLACEHOLDER_REPO_ID = "REPLACE_ME/geometry-transform-diagnosis-v6"
DEFAULT_PATH = os.path.join(REPO_ROOT, "dataset_public")
DEFAULT_CARD = os.path.join(REPO_ROOT, "DATASET_CARD.md")
TOKEN_ENV = "HF_TOKEN"
IGNORE_SUFFIXES = (".tmp", ".DS_Store", ".pyc")
IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def gather_files(root: str) -> Tuple[List[Tuple[str, int]], int]:
    """Return ``[(relpath, bytes), ...]`` (sorted) and the total byte count."""
    files: List[Tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        for name in sorted(filenames):
            if name.endswith(IGNORE_SUFFIXES):
                continue
            abspath = os.path.join(dirpath, name)
            rel = os.path.relpath(abspath, root)
            files.append((rel, os.path.getsize(abspath)))
    files.sort()
    total = sum(size for _, size in files)
    return files, total


def _print_plan(
    args: argparse.Namespace,
    files: List[Tuple[str, int]],
    total: int,
    card_active: bool,
) -> None:
    print("Hugging Face dataset upload plan")
    print(f"  repo_id   : {args.repo_id}")
    print(f"  repo_type : {args.repo_type}")
    print(f"  private   : {args.private}")
    print(f"  local path: {args.path}")
    if card_active:
        print(f"  card      : {args.card} -> README.md")
    elif args.card:
        print(f"  card      : {args.card} -> README.md  (MISSING!)")
    print(f"  files     : {len(files)} ({_human(total)})")
    for rel, size in files:
        print(f"    {rel}  ({_human(size)})")


def run(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.path):
        raise SystemExit(f"local path not found: {args.path}")
    files, _ = gather_files(args.path)
    if not files:
        raise SystemExit(f"no files to upload under {args.path}")

    # When a card is supplied it is uploaded as README.md, so drop any top-level
    # README.md from the folder upload to avoid clobbering the card.
    card_active = os.path.isfile(args.card)
    if card_active:
        files = [(rel, size) for rel, size in files if rel != "README.md"]
    total = sum(size for _, size in files)

    _print_plan(args, files, total, card_active)

    if args.dry_run:
        print("\nDRY RUN: nothing uploaded. Re-run without --dry-run to publish.")
        return 0

    if args.repo_id == PLACEHOLDER_REPO_ID:
        raise SystemExit(
            "refusing to upload to the placeholder repo id; pass a real --repo-id"
        )
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(
            f"no token found; set {TOKEN_ENV} in the environment before uploading"
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "huggingface_hub is required for a real upload: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    ignore_patterns = ["*.tmp", "**/__pycache__/*", "*.DS_Store"]
    if card_active:
        ignore_patterns.append("README.md")
        api.upload_file(
            path_or_fileobj=args.card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message="Add dataset card",
        )
    api.upload_folder(
        folder_path=args.path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        ignore_patterns=ignore_patterns,
        commit_message="Upload v6 transform-diagnosis dataset",
    )
    print(f"\nUploaded {len(files)} files to {args.repo_type}:{args.repo_id}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=PLACEHOLDER_REPO_ID,
                        help="target HF repo id, e.g. your-username/geometry-transform-diagnosis-v6")
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help="local folder to upload (default: dataset_public/)")
    parser.add_argument("--card", default=DEFAULT_CARD,
                        help="dataset card uploaded as README.md (default: DATASET_CARD.md)")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be uploaded; no token or network needed")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
