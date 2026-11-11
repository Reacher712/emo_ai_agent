"""Download original official weights, pinning the resolved revision."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / ".hf-cache"))
# Standard HTTPS downloads are easier to resume/diagnose on this Windows host.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import HfApi, snapshot_download

REPO = "Tongyi-MAI/Z-Image-Turbo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    info = HfApi().model_info(REPO, revision=args.revision)
    destination = ROOT / "models" / "Z-Image-Turbo"
    print(f"Official repository: {REPO}\nRevision: {info.sha}\nDestination: {destination}", flush=True)
    snapshot_download(
        repo_id=REPO, revision=info.sha, local_dir=destination,
        allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                        "transformer/*", "vae/*", "README.md", "LICENSE*"],
        max_workers=args.workers,
    )
    manifest = {"repository": REPO, "revision": info.sha, "quantization": None, "lora": None}
    (destination / "deployment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()

