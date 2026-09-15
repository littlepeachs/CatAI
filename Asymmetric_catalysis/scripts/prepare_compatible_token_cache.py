#!/usr/bin/env python3
"""Create a read-only token-cache view for an exactly rematerialized backbone.

The atom tensors are reused only after the rematerialized fused backbone state
hash has been checked separately against the frozen manifest.  This helper
updates provenance paths/hashes and symlinks the immutable cache payloads; it
does not alter any atom representation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


FILES = (
    "component_atom_tokens.npz",
    "component_metadata.csv",
    "reaction_component_map.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("source manifest or rematerialized checkpoint missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_path = source / "component_atom_tokens.npz"
    if manifest.get("output_token_sha256") != sha256(token_path):
        raise RuntimeError("source token payload hash mismatch")

    output.mkdir(parents=True)
    for name in FILES:
        source_file = source / name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        os.symlink(source_file, output / name)

    manifest["jmp_checkpoint"] = str(checkpoint)
    manifest["jmp_checkpoint_sha256"] = sha256(checkpoint)
    manifest["output_token_path"] = str(output / "component_atom_tokens.npz")
    manifest["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["seed_repeat_cache_view"] = True
    manifest["seed_repeat_source_manifest"] = str(manifest_path)
    manifest["seed_repeat_source_manifest_sha256"] = sha256(manifest_path)
    manifest["representation_recomputed"] = False
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
