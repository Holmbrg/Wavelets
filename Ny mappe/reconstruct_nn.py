"""
reconstruct_nn.py

Reconstruct the compressed Neural Network.
"""

from safetensors.torch import load_file
from transformers import AutoModelForCausalLM  # pylint: disable=import-error
from compress_nn import haar_matrix  # reuse helper

state = load_file("compressed.safetensors")
model = AutoModelForCausalLM.from_config("distilgpt2")

for name, W_hat in state.items():  # no Φ in file
    if W_hat.ndim == 2 and W_hat.shape[1] % 2 == 0:
        Φ = haar_matrix(W_hat.shape[1], dtype=W_hat.dtype, device=W_hat.device)  # pylint: disable=non-ascii-name
        W_dense = Φ.T @ W_hat
        state[name] = W_dense  # rebuild dense tensor

model.load_state_dict(state, strict=False)
