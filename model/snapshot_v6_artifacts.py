"""Snapshot legacy metadata/results before v6 cluster runs.

Dry-run is the default.  ``--execute`` copies only small metadata, result files,
and adapter manifests; it never copies/deletes datasets, renders, model weights,
or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

HOME = Path.home()
DEFAULT_OUT = HOME / "slm_v6_snapshot"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(source_data: Path, artifact_roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for name in ("summary.json", "manifest.json", "manifest_v6.json"):
        path = source_data / name
        if path.is_file():
            files.add(path.resolve())
    for root in artifact_roots:
        if not root.is_dir():
            continue
        for pattern in ("results*.json", "records*.jsonl"):
            files.update(path.resolve() for path in root.glob(pattern) if path.is_file())
        for pattern in (
            "outputs*/trainer_state.json",
            "outputs*/checkpoint-*/trainer_state.json",
            "outputs*/checkpoint-*/adapter_config.json",
        ):
            files.update(path.resolve() for path in root.glob(pattern) if path.is_file())
        for adapter in root.glob("lora_adapters*"):
            if not adapter.is_dir():
                continue
            for name in (
                "adapter_config.json",
                "tokenizer_config.json",
                "processor_config.json",
                "special_tokens_map.json",
                "trainer_state.json",
            ):
                path = adapter / name
                if path.is_file():
                    files.add(path.resolve())
    return sorted(files)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data",
        default=str(HOME / "transform_diagnosis_data"),
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="directory containing legacy results/adapters; repeatable (default: $HOME)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the copy; without this flag the command is a dry run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_data = Path(args.source_data).expanduser().resolve()
    roots = [Path(value).expanduser().resolve() for value in args.artifact_root] or [HOME]
    output = Path(args.out).expanduser().resolve()
    files = discover(source_data, roots)
    print(f"{'EXECUTE' if args.execute else 'DRY RUN'}: {len(files)} artifact file(s)")
    for path in files:
        print(f"  {path}")
    if not args.execute:
        print(f"would copy into {output}; re-run with --execute")
        return 0
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing nonempty snapshot directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"files": []}
    for index, source in enumerate(files):
        relative = Path(f"{index:04d}_{source.parent.name}") / source.name
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest["files"].append({
            "source": str(source),
            "snapshot": str(relative),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        })
    manifest_path = output / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
