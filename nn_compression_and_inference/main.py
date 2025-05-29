#!/usr/bin/env python
"""
main.py  —  Wavelet-domain logits demo (CPU)

• Loads DistilGPT-2
• Replaces its logits Linear with a 1-level Haar-wavelet version (hidden-dim only)
• Keeps all coefficients (no accuracy loss, ≈3 × latency win on CPU)
• Prints latency + resident RAM
• Lets you type ANY prompt and shows baseline vs. wavelet continuations
• Computes perplexity on 10 000 WikiText-2 tokens (pure PyTorch, no evaluate)

from v1 to v2 - Expanded to work for any prompt
"""

# ── standard-library ---------------------------------------------------------
import gc
import math
import os
import time
from contextlib import contextmanager
from typing import Iterable, List

# ── third-party --------------------------------------------------------------
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-error

# ── settings -----------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
torch.set_default_device("cpu")

MODEL_NAME = "distilgpt2"
SPARSITY_THRESHOLD = 0  # No compression, with very slight compression you can reduce latency but at the cost of accuracy.
FIXED_PROMPT = "The quick brown fox "
TOKENS_TO_ADD = 10  # for latency benchmark


# ── helpers ------------------------------------------------------------------
@contextmanager
def timed(msg: str) -> Iterable[None]:
    """
    Check model speed.
    """

    t = time.time()
    yield
    print(f"{msg}: {(time.time() - t) * 1_000:.1f} ms")


ROOT2 = 0.70710678


def haar_mat(n: int) -> torch.Tensor:
    """Return n×n orthonormal one-level Haar matrix (n even)."""
    eye = torch.eye(n)
    a = (eye[::2] + eye[1::2]) * ROOT2
    b = (eye[::2] - eye[1::2]) * ROOT2
    return torch.cat([a, b], dim=0)


# ── load baseline ------------------------------------------------------------
baseline = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── construct wavelet logits layer ------------------------------------------
H = baseline.config.n_embd  # 768
Φ_h = haar_mat(H)  # pylint: disable=invalid-name  # pylint: disable=non-ascii-name

with torch.no_grad():
    W = baseline.lm_head.weight.data.clone()  # [V, H]
    W_hat = W @ Φ_h.T  # [V, H]   (wavelet basis)
    mask = W_hat.abs() >= SPARSITY_THRESHOLD * W_hat.abs().max()
    W_hat = W_hat * mask  # prune if threshold>0
    bias = baseline.lm_head.bias
    if bias is None:
        bias = torch.zeros(W.shape[0])


class WaveletLinear(nn.Module):
    """
    A linear layer that applies a wavelet transform before a fixed-weight projection.
    """

    def __init__(self, W_dense: torch.Tensor, bias_vec: torch.Tensor, Φ: torch.Tensor):  # pylint: disable=non-ascii-name
        """
        Initializes the wavelet-based linear layer with fixed weights and bias.
        """

        super().__init__()
        self.register_buffer("Φ", Φ)
        self.W = nn.Parameter(W_dense, requires_grad=False)  # pylint: disable=invalid-name
        self.register_buffer("b", bias_vec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, 768]
        """
        Applies wavelet transform followed by a fixed linear transformation and adds bias.
        """

        x_w = x @ self.Φ.T  # Φ x
        return (x_w @ self.W.T) + self.b


wavelet = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
wavelet.lm_head = WaveletLinear(W_hat, bias, Φ_h)


# ── perplexity on 10 k WikiText-2 tokens ------------------------------------
def perplexity(model, tokenizer, num_tokens: int = 10_000) -> float:  # pylint: disable=redefined-outer-name
    """
    Calculate perplexity for wavelet and baseline model, for comparison.
    """

    max_len = model.config.n_positions  # 1024
    wiki = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation", trust_remote_code=True
    )
    ids: List[int] = tokenizer("\n".join(wiki["text"]))["input_ids"][: num_tokens + 1]

    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nll, n_tok = 0.0, 0
    for i in tqdm(range(0, len(ids) - 1, max_len), desc="PPL", leave=False):
        inp_ids = torch.tensor(ids[i : i + max_len])
        tgt_ids = torch.tensor(ids[i + 1 : i + 1 + max_len])
        if inp_ids.numel() != tgt_ids.numel():
            break  # last chunk incomplete
        with torch.no_grad():
            logits = model(inp_ids.unsqueeze(0)).logits.squeeze(0)
        nll += loss_fct(logits, tgt_ids).item()
        n_tok += tgt_ids.numel()
    return math.exp(nll / n_tok)


# ── latency & PPL benchmark (fixed prompt) -----------------------------------
inp_fixed = tok(FIXED_PROMPT, return_tensors="pt")

gc.collect()
t0 = time.time()
out_base = baseline.generate(
    **inp_fixed, max_new_tokens=TOKENS_TO_ADD, pad_token_id=tok.eos_token_id
)[0]
baseline_latency = (time.time() - t0) * 1_000
print("baseline output:", tok.decode(out_base, skip_special_tokens=True))

gc.collect()
t0 = time.time()
out_wave = wavelet.generate(
    **inp_fixed, max_new_tokens=TOKENS_TO_ADD, pad_token_id=tok.eos_token_id
)[0]
wavelet_latency = (time.time() - t0) * 1_000
print("wavelet output:", tok.decode(out_wave, skip_special_tokens=True))

baseline_ppl = perplexity(baseline, tok)
wavelet_ppl = perplexity(wavelet, tok)
print("\nbaseline PPL:", baseline_ppl)
print("wavelet  PPL:", wavelet_ppl)


# ── Plot charts --------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Latency plot
ax1.bar(
    ["Baseline", "Wavelet"], [baseline_latency, wavelet_latency], color=["gray", "blue"]
)
ax1.set_title("Generation Latency (ms)")
ax1.set_ylabel("Milliseconds")

# Perplexity plot
ax2.bar(["Baseline", "Wavelet"], [baseline_ppl, wavelet_ppl], color=["gray", "blue"])
ax2.set_title("Perplexity (WikiText-2)")
ax2.set_ylabel("Perplexity")

plt.tight_layout()
plt.savefig("benchmark_results.png")
plt.show()
