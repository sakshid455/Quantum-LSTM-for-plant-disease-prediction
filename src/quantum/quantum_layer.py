import os
import math

import pennylane as qml
import torch
import torch.nn as nn

# Number of qubits
N_QUBITS = 8

# Number of StronglyEntanglingLayers repetitions
N_LAYERS = 3

# "torch" = exact batched statevector simulation in PyTorch (fast, default).
# "pennylane" = PennyLane default.qubit reference implementation.
BACKEND = os.environ.get("QLSTM_BACKEND", "torch")

# Quantum simulator
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode classical data
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))

    # Trainable quantum layer
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # Measure expectation values
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    ]


def _cnot_permutation(n_qubits, control, target):
    """Basis-index permutation of CNOT(control, target); wire 0 is the most significant bit."""
    idx = torch.arange(2 ** n_qubits)
    c_bit = (idx >> (n_qubits - 1 - control)) & 1
    return idx ^ (c_bit << (n_qubits - 1 - target))


def _rot_matrix(phi, theta, omega):
    """PennyLane Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi), batched over leading dims."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    plus = torch.exp(-0.5j * (phi + omega))
    minus = torch.exp(0.5j * (phi - omega))
    return torch.stack([
        torch.stack([plus * c, -minus * s], dim=-1),
        torch.stack([minus.conj() * s, plus.conj() * c], dim=-1),
    ], dim=-2)


_ROW, _COL = "abcdefghijklm", "nopqrstuvwxyz"


def _kron_wires(m):
    """Kronecker product over the wire axis: (..., k, 2, 2) -> (..., 2^k, 2^k), wire 0 most significant."""
    k = m.shape[-3]
    subs = ",".join(f"...{_ROW[i]}{_COL[i]}" for i in range(k))
    out = torch.einsum(f"{subs}->...{_ROW[:k]}{_COL[:k]}", *m.unbind(-3))
    return out.reshape(*m.shape[:-3], 2 ** k, 2 ** k)


def _kron_vectors(v):
    """Kronecker product over the wire axis: (..., k, 2) -> (..., 2^k)."""
    k = v.shape[-2]
    subs = ",".join(f"...{_ROW[i]}" for i in range(k))
    out = torch.einsum(f"{subs}->...{_ROW[:k]}", *v.unbind(-2))
    return out.reshape(*v.shape[:-2], 2 ** k)


class TorchStatevectorCircuit(nn.Module):
    """
    Exact simulation of AngleEmbedding(X) + StronglyEntanglingLayers + <Z_i>,
    numerically identical to quantum_circuit above.

    The wires are split into two halves A and B, so the state is held as a
    (batch, 2^a, 2^b) matrix. A layer of single-qubit rotations is then
    U_A @ state @ U_B^T, where U_A / U_B are Kronecker products of the Rot
    gates on each half, and the layer's CNOT ring is one basis permutation.
    This keeps the op count small, which dominates runtime at these sizes.
    """

    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        # Same parameter name, shape and initialisation as qml.qnn.TorchLayer.
        self.weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3) * 2 * math.pi)

        ranges = [(l % (n_qubits - 1)) + 1 for l in range(n_layers)]
        for l, r in enumerate(ranges):
            perm = torch.arange(2 ** n_qubits)
            for i in range(n_qubits):
                perm = perm[_cnot_permutation(n_qubits, i, (i + r) % n_qubits)]
            self.register_buffer(f"cnot_ring_{l}", perm, persistent=False)

        idx = torch.arange(2 ** n_qubits)
        bits = torch.stack([(idx >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)], dim=1)
        self.register_buffer("z_signs", (1 - 2 * bits).float(), persistent=False)

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        n, batch = self.n_qubits, x.shape[0]
        a = n // 2

        half = x.double() / 2
        # RX(x)|0> = cos(x/2)|0> - i sin(x/2)|1>, per wire
        amps = torch.stack([torch.cos(half).to(torch.complex128), -1j * torch.sin(half)], dim=-1)
        state = _kron_vectors(amps[:, :a]).unsqueeze(-1) * _kron_vectors(amps[:, a:]).unsqueeze(1)

        w = self.weights.double()
        rots = _rot_matrix(w[..., 0], w[..., 1], w[..., 2])  # (layers, wires, 2, 2)
        u_a = _kron_wires(rots[:, :a])
        u_b_t = _kron_wires(rots[:, a:]).transpose(-1, -2)
        for l in range(w.shape[0]):
            state = u_a[l] @ state @ u_b_t[l]
            state = state.reshape(batch, -1)[:, getattr(self, f"cnot_ring_{l}")]
            state = state.reshape(batch, 2 ** a, 2 ** (n - a))

        state = state.reshape(batch, -1)
        probs = state.real ** 2 + state.imag ** 2
        out = (probs @ self.z_signs.double()).to(x.dtype)
        return out.squeeze(0) if squeeze else out


class QuantumLayer(nn.Module):

    def __init__(self, backend=None):
        super().__init__()

        backend = backend or BACKEND

        if backend == "torch":
            self.qlayer = TorchStatevectorCircuit()
        elif backend == "pennylane":
            weight_shapes = {
                "weights": (N_LAYERS, N_QUBITS, 3)
            }

            self.qlayer = qml.qnn.TorchLayer(
                quantum_circuit,
                weight_shapes
            )
        else:
            raise ValueError(f"Unknown QLSTM_BACKEND: {backend}")

    def forward(self, x):
        return self.qlayer(x)
