# SPDX-License-Identifier: Apache-2.0
"""Download encoder/decoder JIT checkpoints for the tokenizers in the registry.

Only encoder.jit and decoder.jit are fetched (the full autoencoder.jit is not
needed for reconstruction eval). None of these repos are gated, so no HF token
is required.

Usage:
    python eval_quality/download_checkpoints.py [--checkpoint_dir checkpoints]
        [--tokenizers CI8x8-360p ...] [--include-legacy]
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from tokenizers_registry import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--tokenizers", nargs="*", default=None, help="subset of tokenizer names; default: all")
    parser.add_argument("--include-legacy", action="store_true", help="also download the legacy 0.1 fill-in checkpoints")
    args = parser.parse_args()

    for tok in select(args.tokenizers, args.include_legacy):
        local_dir = Path(args.checkpoint_dir) / tok["hf_repo"].split("/")[-1]
        print(f"Downloading {tok['hf_repo']} -> {local_dir}")
        snapshot_download(
            repo_id=tok["hf_repo"],
            local_dir=local_dir,
            allow_patterns=["encoder.jit", "decoder.jit"],
        )
    print("Done.")


if __name__ == "__main__":
    main()
