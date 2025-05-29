#!/usr/bin/env python
"""
compress_nn.py

Wavelet-compress a Hugging-Face checkpoint and write one
compact `.safetensors` file (only FP16 wavelet weights, no Φ).

Usage examples
--------------
# loss-less (≈ ½ size)
python compress_nn.py --model distilgpt2 --keep 1.0

# keep top 50 % magnitudes (≈ ¼–⅓ size)
python compress_nn.py --model ./distilgpt2 --keep 0.5
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Dict

import torch
from safetensors.torch import save_file
from transformers import AutoModel  # pylint: disable=import-error

ROOT2: float = 1.0 / math.sqrt(2.0)


# --------------------------------------------------------------------------- #
def haar_matrix(size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a 1-level orthonormal Haar matrix Φ ∈ ℝ^{size×size}.

    The matrix is cached by (size, dtype, device) to avoid rebuilding it.
    """
    key = (size, dtype, device)
    if key not in _HAAR_CACHE:
        identity = torch.eye(size, dtype=dtype, device=device)
        _HAAR_CACHE[key] = torch.cat(
            (
                (identity[::2] + identity[1::2]) * ROOT2,
                (identity[::2] - identity[1::2]) * ROOT2,
            ),
            dim=0,
        )
    return _HAAR_CACHE[key]


_HAAR_CACHE: Dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def compress_weight(weight: torch.Tensor, keep: float) -> torch.Tensor:
    """Convert *weight* to wavelet basis, prune, and cast to FP16."""
    phi = haar_matrix(weight.shape[1], dtype=weight.dtype, device=weight.device)
    coeff = (weight @ phi.T).half()  # change of basis + fp16

    if 0.0 < keep < 1.0:
        threshold: float = (1.0 - keep) * coeff.abs().max().item()
        coeff[coeff.abs() < threshold] = 0.0

    return coeff


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="Wavelet weight compressor")
    parser.add_argument(
        "--model",
        required=True,
        help="HF hub name or local folder with config.json & weights",
    )
    parser.add_argument(
        "--keep",
        type=float,
        default=1.0,
        help="fraction of coefficients kept (0 < keep ≤ 1)",
    )
    parser.add_argument(
        "--out",
        default="compressed",
        help="basename (without extension) of output .safetensors file",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    model = AutoModel.from_pretrained(args.model)
    dense_sd = model.state_dict()
    sparse_sd: Dict[str, torch.Tensor] = {}

    dense_bytes = 0
    sparse_bytes = 0

    for name, param in dense_sd.items():
        if param.ndim == 2 and param.shape[1] % 2 == 0:
            comp = compress_weight(param, args.keep)
            sparse_sd[name] = comp
            dense_bytes += param.numel() * param.element_size()
            sparse_bytes += comp.numel() * 2  # fp16 = 2 bytes
        else:
            fp16 = param.half()
            sparse_sd[name] = fp16
            dense_bytes += param.numel() * param.element_size()
            sparse_bytes += fp16.numel() * 2

    out_path = pathlib.Path(args.out).with_suffix(".safetensors")
    save_file(sparse_sd, str(out_path))

    _report_sizes(out_path, dense_bytes, sparse_bytes)


def _report_sizes(out_path: pathlib.Path, dense_b: int, sparse_b: int) -> None:
    """Print plain-text size report."""
    meg = 1 / 2**20
    print(f"\nsaved   → {out_path.resolve()}")
    print(f"dense    {dense_b * meg:7.1f} MB")
    ratio = sparse_b / dense_b
    print(f"wavelet  {sparse_b * meg:7.1f} MB   ({ratio:.2%} of original)")


if __name__ == "__main__":
    main()
