import torch
import torch.nn as nn


class LinearReLU(nn.Module):
    def __init__(self, in_features=64, out_features=32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)   # a = Wx + b

    def forward(self, x):
        return torch.relu(self.linear(x))                    # non-linear function


def fake_quant(t, scale, mode="round"):
    """Symmetric INT8 quantize -> dequantize (zero-point 0, range [-128, 127]).

    mode="round": round-to-nearest (the usual definition).
    mode="trunc": round-toward-zero, which is what MLIR's --lower-quant-ops
                  emits (arith.fptosi). compare.py checks against both.
    """
    s = t / scale
    s = torch.round(s) if mode == "round" else torch.trunc(s)
    return torch.clamp(s, -128, 127) * scale
