"""Hybrid quantum LSTM (QLSTM) and its classical LSTM baseline.

QLSTM follows Chen, Yoo & Fang (2020), "Quantum Long Short-Term Memory" (arXiv:2009.01783): each of the four LSTM
gates (forget, input, cell-update, output) is a variational quantum circuit (VQC). At every time step the classical
vector [x_t, h_{t-1}] is compressed to n_qubits angles, each VQC returns n_qubits Pauli-Z expectations, and a
linear layer maps them to the hidden size. The circuits are simulated with PennyLane and trained end to end with
PyTorch through qml.qnn.TorchLayer.
"""
from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


def make_vqc(n_qubits: int, n_layers: int) -> nn.Module:
    """AngleEmbedding + StronglyEntanglingLayers, measured in Z on every wire. Supports batched inputs."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}
    return qml.qnn.TorchLayer(circuit, shapes)


class QLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 5, horizon: int = 6, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.compress = nn.Linear(n_features + hidden, n_qubits)
        self.vqc = nn.ModuleDict({g: make_vqc(n_qubits, n_layers) for g in ("forget", "input", "cell", "output")})
        self.expand = nn.ModuleDict({g: nn.Linear(n_qubits, hidden) for g in self.vqc})
        self.head = nn.Linear(hidden, horizon)

    def _gate(self, name: str, v: torch.Tensor) -> torch.Tensor:
        return self.expand[name](self.vqc[name](v).to(v.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = x.new_zeros(b, self.hidden)
        c = x.new_zeros(b, self.hidden)
        for step in range(t):
            v = torch.atan(self.compress(torch.cat([x[:, step], h], dim=1)))  # bounded angles
            f = torch.sigmoid(self._gate("forget", v))
            i = torch.sigmoid(self._gate("input", v))
            g = torch.tanh(self._gate("cell", v))
            o = torch.sigmoid(self._gate("output", v))
            c = f * c + i * g
            h = o * torch.tanh(c)
        return self.head(h)


class ClassicalLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 5, horizon: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def build(name: str, n_features: int, hidden: int, horizon: int, n_qubits: int = 4, n_layers: int = 2) -> nn.Module:
    if name == "qlstm":
        return QLSTM(n_features, hidden, horizon, n_qubits, n_layers)
    if name == "lstm":
        return ClassicalLSTM(n_features, hidden, horizon)
    raise ValueError(name)
