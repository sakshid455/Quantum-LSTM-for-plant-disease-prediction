import pennylane as qml
import torch
import torch.nn as nn

# Number of qubits
N_QUBITS = 4

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


class QuantumLayer(nn.Module):

    def __init__(self):
        super().__init__()

        weight_shapes = {
            "weights": (2, N_QUBITS, 3)
        }

        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes
        )

    def forward(self, x):
        return self.qlayer(x)