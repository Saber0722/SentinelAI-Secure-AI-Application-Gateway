"""Download the public evaluation datasets and open-source guard models (explicit, one-time setup).

    uv sync --extra guards
    uv run --extra guards python scripts/download_external.py

Datasets go to data/external/ and models to models/guards/ (both git-ignored). A manifest with the
SHA-256 of every downloaded file is written to data/external/manifest.json so runs are traceable.
Nothing here runs during normal operation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from sentinelai.evaluation.external import DATASETS, GUARD_MODELS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external"
MODEL_DIR = ROOT / "models" / "guards"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest: dict = {"downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"), "datasets": {}, "models": {}}
    for ds in DATASETS:
        files = {}
        for filename in ds.files:
            path = Path(hf_hub_download(ds.repo, filename, repo_type="dataset", local_dir=DATA_DIR / ds.key))
            files[filename] = sha256(path)
            print(f"dataset {ds.repo}/{filename}  sha256={files[filename][:12]}")
        manifest["datasets"][ds.key] = {"repo": ds.repo, "license": ds.license, "files": files}
    for model in GUARD_MODELS:
        path = Path(snapshot_download(model.repo, local_dir=MODEL_DIR / model.key,
                                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))  # fmt: skip
        weights = sorted(path.glob("*.safetensors"))
        manifest["models"][model.key] = {"repo": model.repo, "license": model.license,
                                         "weights_sha256": {w.name: sha256(w) for w in weights}}  # fmt: skip
        print(f"model   {model.repo}  ({sum(w.stat().st_size for w in weights) / 2**20:.0f} MiB)")
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
