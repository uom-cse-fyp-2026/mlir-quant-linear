"""Compare mlir-runner output with the PyTorch references."""
import re
import sys

import torch


def parse(path):
    txt = open(path).read()
    data = txt[txt.index("data ="):]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", data)
    return torch.tensor([float(n) for n in nums])


io = torch.load("build/io.pt")
fp32, q = parse("build/out_fp32.txt"), parse("build/out_q.txt")
ref_fp32 = io["ref_fp32"].flatten()
ref_round, ref_trunc = io["ref_q_round"].flatten(), io["ref_q_trunc"].flatten()
ATOL = 1e-4

ok_fp32 = torch.allclose(fp32, ref_fp32, atol=ATOL)
ok_round = torch.allclose(q, ref_round, atol=ATOL)
ok_trunc = torch.allclose(q, ref_trunc, atol=ATOL)
err = (q - fp32).abs().max().item()

print("MLIR fp32  vs PyTorch fp32                :", ok_fp32)
print("MLIR quant vs PyTorch fake-quant (round)  :", ok_round,
      f"(max diff {(q - ref_round).abs().max().item():.3e})")
print("MLIR quant vs PyTorch fake-quant (trunc)  :", ok_trunc,
      f"(max diff {(q - ref_trunc).abs().max().item():.3e})")
print(f"quantization error (max |q - fp32|)       : {err:.6e}")

ok = ok_fp32 and (ok_round or ok_trunc) and err > 0
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
