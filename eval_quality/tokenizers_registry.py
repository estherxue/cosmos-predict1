# SPDX-License-Identifier: Apache-2.0
"""Registry of Cosmos tokenizer checkpoints evaluated in this study.

`compression` is the nominal spatio-temporal downsampling factor
(T_ratio * H_ratio * W_ratio), not a bit-rate compression ratio.
"""

TOKENIZERS = [
    # -- Cosmos-Tokenize1 series (the ones shipped with cosmos-predict1) --
    dict(name="CI8x8-360p", hf_repo="nvidia/Cosmos-Tokenize1-CI8x8-360p", family="CI", kind="image", compression=64, series="tokenize1"),
    dict(name="CI16x16-360p", hf_repo="nvidia/Cosmos-Tokenize1-CI16x16-360p", family="CI", kind="image", compression=256, series="tokenize1"),
    dict(name="DI8x8-360p", hf_repo="nvidia/Cosmos-Tokenize1-DI8x8-360p", family="DI", kind="image", compression=64, series="tokenize1"),
    dict(name="DI16x16-360p", hf_repo="nvidia/Cosmos-Tokenize1-DI16x16-360p", family="DI", kind="image", compression=256, series="tokenize1"),
    dict(name="CV4x8x8-360p", hf_repo="nvidia/Cosmos-Tokenize1-CV4x8x8-360p", family="CV", kind="video", compression=256, series="tokenize1"),
    dict(name="CV8x8x8-720p", hf_repo="nvidia/Cosmos-Tokenize1-CV8x8x8-720p", family="CV", kind="video", compression=512, series="tokenize1"),
    dict(name="DV4x8x8-360p", hf_repo="nvidia/Cosmos-Tokenize1-DV4x8x8-360p", family="DV", kind="video", compression=256, series="tokenize1"),
    dict(name="DV8x16x16-720p", hf_repo="nvidia/Cosmos-Tokenize1-DV8x16x16-720p", family="DV", kind="video", compression=2048, series="tokenize1"),
    # -- Legacy Cosmos-0.1 series: one training recipe across all compression
    # rates (no per-variant resolution split), so within-series comparisons are
    # clean compression ablations --
    dict(name="0.1-CV4x8x8", hf_repo="nvidia/Cosmos-0.1-Tokenizer-CV4x8x8", family="CV", kind="video", compression=256, series="legacy"),
    dict(name="0.1-CV8x8x8", hf_repo="nvidia/Cosmos-0.1-Tokenizer-CV8x8x8", family="CV", kind="video", compression=512, series="legacy"),
    dict(name="0.1-CV8x16x16", hf_repo="nvidia/Cosmos-0.1-Tokenizer-CV8x16x16", family="CV", kind="video", compression=2048, series="legacy"),
    dict(name="0.1-DV4x8x8", hf_repo="nvidia/Cosmos-0.1-Tokenizer-DV4x8x8", family="DV", kind="video", compression=256, series="legacy"),
    dict(name="0.1-DV8x8x8", hf_repo="nvidia/Cosmos-0.1-Tokenizer-DV8x8x8", family="DV", kind="video", compression=512, series="legacy"),
    dict(name="0.1-DV8x16x16", hf_repo="nvidia/Cosmos-0.1-Tokenizer-DV8x16x16", family="DV", kind="video", compression=2048, series="legacy"),
]


def select(names=None, include_legacy=False):
    """Return registry entries, defaulting to the Tokenize1 series.

    Args:
        names: optional list of tokenizer names; overrides include_legacy.
        include_legacy: also include the legacy 0.1 fill-in checkpoints.
    """
    if names:
        by_name = {t["name"]: t for t in TOKENIZERS}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise ValueError(f"Unknown tokenizer(s) {unknown}. Available: {sorted(by_name)}")
        return [by_name[n] for n in names]
    return [t for t in TOKENIZERS if include_legacy or t["series"] == "tokenize1"]
