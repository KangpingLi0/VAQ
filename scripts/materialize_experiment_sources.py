"""Restore frozen experiment sources from pinned Git commits and local patches.

Example (the lmms-eval checkout must contain the pinned revision)::

    python scripts/materialize_experiment_sources.py \
        internvl2_8b_methods_20261115 --lmms-repo /path/to/lmms-eval

No existing experiment source directory is changed. Data, checkpoints and
environment-specific paths in the original launchers still need provisioning.
InternVL additionally uses timm==0.9.16 in its isolated deps directory.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def archive(repository, revision, destination):
    payload = subprocess.check_output([
        "git", "-C", str(repository), "archive", "--format=tar", revision,
    ])
    with tarfile.open(fileobj=io.BytesIO(payload)) as bundle:
        for item in bundle.getmembers():
            path = Path(item.name)
            if path.is_absolute() or ".." in path.parts or item.issym() or item.islnk():
                raise ValueError(f"Unsafe archive member: {item.name}")
        bundle.extractall(destination)


def verify(source, expected):
    for name, digest in expected.items():
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--lmms-repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if Path(args.campaign).name != args.campaign:
        parser.error("campaign must be a directory name, without path components")
    base = ROOT / "experiments" / args.campaign
    manifest = json.loads((base / "source_manifest.json").read_text())
    destination = args.destination or base / "source"
    if destination.exists():
        raise SystemExit(f"Refusing to replace existing source directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vaq-source-", dir=destination.parent) as temporary:
        source = Path(temporary) / "source"
        source.mkdir()
        archive(ROOT, manifest["project_revision"], source)
        lmms = source / "3rdparty" / "lmms-eval-new"
        lmms.mkdir(parents=True, exist_ok=True)
        archive(args.lmms_repo, manifest["lmms_revision"], lmms)
        for directory, patch in [(source, "project.patch"), (lmms, "lmms_eval.patch")]:
            patch_path = base / patch
            if patch_path.stat().st_size:
                subprocess.run([
                    "git", "apply", "--no-index", "--whitespace=nowarn",
                    str(patch_path),
                ], cwd=directory, check=True)
        verify(source, manifest["sha256"])
        shutil.move(str(source), str(destination))
    print(f"Restored and verified {len(manifest['sha256'])} source files: {destination}")


if __name__ == "__main__":
    main()
