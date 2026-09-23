"""Export the model to MLIR: torch dialect (for reading) and linalg-on-tensors
(input to the quantize-linear pass). A @main that feeds the calibration input
and prints the result is appended so the IR can be executed with mlir-runner."""
import torch
from torch_mlir import fx

from model import LinearReLU

model = LinearReLU().eval()
model.load_state_dict(torch.load("build/model.pt"))
x = torch.load("build/io.pt")["x"]

# High-level view (torch dialect) - for reading/understanding only
m = fx.export_and_import(model, x, output_type="torch", func_name="forward")
with open("build/1_torch.mlir", "w") as f:
    f.write(str(m))

# Lowered view (linalg on tensors) - this is what the pass works on
m = fx.export_and_import(model, x, output_type="linalg-on-tensors", func_name="forward")
text = str(m)

vals = ", ".join(f"{v:.9e}" for v in x.flatten().tolist())
main = f"""
  func.func private @printMemrefF32(tensor<*xf32>) attributes {{llvm.emit_c_interface}}
  func.func @main() {{
    %x = arith.constant dense<[[{vals}]]> : tensor<1x64xf32>
    %y = call @forward(%x) : (tensor<1x64xf32>) -> tensor<1x32xf32>
    %u = tensor.cast %y : tensor<1x32xf32> to tensor<*xf32>
    call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
    return
  }}
"""
# Insert main before the module's closing brace (and before any {-# resources #-} block)
cut = text.find("{-#")
head, tail = (text, "") if cut == -1 else (text[:cut], text[cut:])
end = head.rstrip().rfind("}")
with open("build/2_linalg.mlir", "w") as f:
    f.write(head[:end] + main + head[end:] + tail)
print("wrote build/1_torch.mlir and build/2_linalg.mlir")
