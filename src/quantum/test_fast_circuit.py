"""
Check that the PyTorch statevector circuit reproduces the PennyLane reference
(outputs and gradients w.r.t. inputs and weights), then compare speed.

    python -m src.quantum.test_fast_circuit
"""
import time

import torch

from .quantum_layer import QuantumLayer, N_QUBITS

torch.manual_seed(0)

fast = QuantumLayer(backend="torch")
ref = QuantumLayer(backend="pennylane")
ref.qlayer.weights.data.copy_(fast.qlayer.weights.data)

# Encoder outputs are unbounded, so test well outside [0, 2pi) too.
x = (torch.randn(16, N_QUBITS) * 3).requires_grad_(True)
x_ref = x.detach().clone().requires_grad_(True)

out_fast = fast(x)
out_ref = ref(x_ref).to(out_fast.dtype)

g = torch.randn_like(out_fast)
out_fast.backward(g)
out_ref.backward(g)

checks = {
    "outputs": (out_fast, out_ref),
    "d/d inputs": (x.grad, x_ref.grad),
    "d/d weights": (fast.qlayer.weights.grad, ref.qlayer.weights.grad),
}
ok = True
for name, (a, b) in checks.items():
    err = (a - b.to(a.dtype)).abs().max().item()
    passed = err < 1e-5
    ok &= passed
    print(f"{name:12s} max |diff| = {err:.2e}  {'PASS' if passed else 'FAIL'}")

single = torch.randn(N_QUBITS)
err = (fast(single) - ref(single).to(torch.float32)).abs().max().item()
ok &= err < 1e-5
print(f"{'unbatched':12s} max |diff| = {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")

# Speed: forward + backward on a QLSTM-sized batch.
for name, layer in [("pennylane", ref), ("torch", fast)]:
    xb = torch.randn(8, N_QUBITS, requires_grad=True)
    reps = 5 if name == "pennylane" else 200
    start = time.perf_counter()
    for _ in range(reps):
        layer(xb).sum().backward()
    per_call = (time.perf_counter() - start) / reps
    print(f"{name:10s} {per_call * 1e3:8.2f} ms per forward+backward (batch 8)")

assert ok, "Torch circuit does not match the PennyLane reference"
print("\nAll equivalence checks passed.")
