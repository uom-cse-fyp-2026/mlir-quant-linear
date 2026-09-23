"""Fix the weights, compute INT8 scales, and save the reference outputs."""
import json
import os

import torch

from model import LinearReLU, fake_quant

os.makedirs("build", exist_ok=True)
torch.manual_seed(0)
model = LinearReLU().eval()
torch.save(model.state_dict(), "build/model.pt")

x = torch.randn(1, 64)

# Symmetric per-tensor scales, zero-point = 0.
# The activation range includes x itself so no value falls outside [-128, 127]
# (out-of-range float->i8 conversion is undefined after lowering).
calib = torch.randn(128, 64)
act_scale = max(calib.abs().max().item(), x.abs().max().item()) / 127
w_scale = model.linear.weight.abs().max().item() / 127

with torch.no_grad():
    W, b = model.linear.weight, model.linear.bias
    ref_fp32 = model(x)
    ref_q = {
        mode: torch.relu(fake_quant(x, act_scale, mode) @ fake_quant(W, w_scale, mode).T + b)
        for mode in ("round", "trunc")
    }

with open("build/scales.json", "w") as f:
    json.dump({"act_scale": act_scale, "w_scale": w_scale}, f, indent=2)
torch.save({"x": x, "ref_fp32": ref_fp32,
            "ref_q_round": ref_q["round"], "ref_q_trunc": ref_q["trunc"]},
           "build/io.pt")
print(f"act_scale={act_scale:.9g}  w_scale={w_scale:.9g}")
