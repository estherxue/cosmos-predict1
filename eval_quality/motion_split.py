# SPDX-License-Identifier: Apache-2.0
"""High/low-motion split analysis between two video tokenizers.

Splits clips by the per-clip motion score stored in metrics.json (mean |frame
diff|), then compares the PSNR drop from tokenizer A to B per group, plus the
Spearman rank correlation between motion and drop.

Usage:
    python eval_quality/motion_split.py results_pair/metrics.json 0.1-CV4x8x8 0.1-CV8x8x8
"""

import json
import sys


def main():
    metrics_path, name_a, name_b = sys.argv[1:4]
    d = json.load(open(metrics_path))
    ra, rb = (next(r for r in d["results"] if r["name"] == n) for n in (name_a, name_b))
    seqs = sorted(ra["samples"], key=lambda s: ra["samples"][s]["motion"])
    half = len(seqs) // 2

    for label, group in (("low-motion", seqs[:half]), ("high-motion", seqs[half:])):
        motion = sum(ra["samples"][s]["motion"] for s in group) / len(group)
        pa = sum(ra["samples"][s]["psnr"] for s in group) / len(group)
        pb = sum(rb["samples"][s]["psnr"] for s in group) / len(group)
        print(f"{label:12s} (n={len(group)}): motion {motion:5.2f}  "
              f"{name_a} {pa:5.2f} dB  {name_b} {pb:5.2f} dB  drop {pa - pb:+.2f} dB")

    drop = {s: ra["samples"][s]["psnr"] - rb["samples"][s]["psnr"] for s in seqs}
    print("\nlargest per-clip drops:")
    for s in sorted(seqs, key=lambda s: -drop[s])[:5]:
        print(f"  {drop[s]:+.2f} dB  {s:24s} motion {ra['samples'][s]['motion']:.2f}")

    n = len(seqs)
    mrank = {s: i for i, s in enumerate(seqs)}
    drank = {s: i for i, s in enumerate(sorted(seqs, key=lambda s: drop[s]))}
    rho = 1 - 6 * sum((mrank[s] - drank[s]) ** 2 for s in seqs) / (n * (n**2 - 1))
    print(f"\nSpearman rho(motion, PSNR drop) = {rho:.3f}  (n={n})")


if __name__ == "__main__":
    main()
