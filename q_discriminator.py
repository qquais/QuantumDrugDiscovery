import pennylane as qml
import sys
import torch
import torch.nn.functional as F

ATOM_NUM = 45
n_circuites = ATOM_NUM
n_qubits = 5

n_measured_wire = 1
n_2nd_qubits = 9#int(n_circuites * n_measured_wire)#int(np.ceil(np.sqrt(n_qubits * n_circuites)))
n_2nd_circuits = int(n_circuites/n_2nd_qubits)

n_3rd_circuits = 1
n_3rd_qubits = int( n_measured_wire * n_2nd_circuits / n_3rd_circuits)

print((n_circuites, n_qubits), (n_2nd_circuits, n_2nd_qubits), (n_3rd_circuits, n_3rd_qubits))
dev = qml.device("default.qubit", wires=n_qubits)
dev1 = qml.device("default.qubit", wires=n_2nd_qubits)
dev2 = qml.device("default.qubit", wires=n_3rd_qubits)

MEASURED_QUBIT_IDX = 2#int(sys.argv[1])
MEASURED_QUBIT_2ND_IDX = 7#int(sys.argv[2])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.001, normalize=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

@qml.qnode(dev1, interface="torch", diff_method="backprop")
def qnode_(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_2nd_qubits), pad_with=0.001, normalize=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_2nd_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_2nd_qubits)]

@qml.qnode(dev2, interface="torch", diff_method="backprop")
def qnode__(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_3rd_qubits), pad_with=0.001, normalize=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_3rd_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_3rd_qubits)]


n_layers = 1
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
n_2nd_layers = 3
n_3rd_layers = 1
weight_shapes_2nd = {"weights": (n_2nd_layers, n_2nd_qubits, 3)}
weight_shapes_3rd = {"weights": (n_3rd_layers, n_3rd_qubits, 3)}

# Output sizes after measurement change (all qubits measured)
_qnode_out_size    = n_qubits       # 5
_qnode__out_size   = n_2nd_qubits   # 9
_qnode___out_size  = n_3rd_qubits   # 5


class HybridModel(torch.nn.Module):
    def __init__(self, LAYER3 = False):
        super().__init__()
        self.qlayer_1 = torch.nn.ModuleList([qml.qnn.TorchLayer(qnode, weight_shapes) for i in range(ATOM_NUM)])
        self.qlayer_21 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
        self.LAYER3 = LAYER3
        if self.LAYER3:
            self.qlayer_22 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_23 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_24 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_25 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_31 = qml.qnn.TorchLayer(qnode__, weight_shapes_3rd)
            # LAYER3: 5 x n_2nd_qubits outputs -> n_3rd_qubits -> 1
            self.output_layer = torch.nn.Linear(_qnode___out_size, 1)
        else:
            self.output_layer = torch.nn.Linear(_qnode__out_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.split(x, 1, dim=1)

        for i, l in enumerate(self.qlayer_1):
            tmp = self.qlayer_1[i](x[i])
            if i > 0:
                out = torch.cat([out, tmp], axis=2)
            else:
                out = tmp

        x = torch.squeeze(out, 1)
        if self.LAYER3:
            x = torch.split(x, 9, dim=1)
            out1 = self.qlayer_21(x[0])
            out2 = self.qlayer_22(x[1])
            out3 = self.qlayer_23(x[2])
            out4 = self.qlayer_24(x[3])
            out5 = self.qlayer_25(x[4])
            out = torch.cat([out1, out2, out3, out4, out5], axis=1)
            out = self.qlayer_31(out)
        else:
            out = self.qlayer_21(x)
        return self.sigmoid(self.output_layer(out))


# --- KaoQuantumDisc ---
# 9-qubit quantum discriminator matching the reference architecture.
# Input: upper-triangular bonds + atoms -> (batch, 225) -> pad to 512
# 9 qubits, 3 StronglyEntanglingLayers, measure ONE qubit (qubit 4)
# Training: weight clamping +-0.01, loss = d_real + d_fake (same sign)
# Output: (batch, 1) scalar PauliZ expectation

_kao_n_qubits = 9
_kao_n_layers = 3
_kao_measured_qubit = 4
_kao_dev = qml.device("default.qubit", wires=_kao_n_qubits)

@qml.qnode(_kao_dev, interface="torch", diff_method="backprop")
def _kao_qnode(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(_kao_n_qubits),
                                     pad_with=0.001, normalize=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(_kao_n_qubits))
    return [qml.expval(qml.PauliZ(wires=_kao_measured_qubit))]

_kao_weight_shapes = {"weights": (_kao_n_layers, _kao_n_qubits, 3)}


class KaoQuantumDisc(torch.nn.Module):
    """9-qubit quantum discriminator — single qubit 4 measured, learned output scaling."""
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(_kao_qnode, _kao_weight_shapes)
        self.fc = torch.nn.Linear(1, 1)  # learnable scale+bias; unbounds the [-1,1] PauliZ output

    def forward(self, x):
        # x: (batch, 45, 5) upper-tri bonds+atoms, or (batch, 225) flat
        x = x.reshape(x.shape[0], -1).float()
        return self.fc(self.qlayer(x))  # (batch, 1) — unbounded critic score


# Keep SimpleQuantumDisc as alias for backwards compatibility
SimpleQuantumDisc = KaoQuantumDisc


def sanity_check_quantum_disc(model, device='cpu'):
    """Verify: outputs differ, gradients non-zero. Input: (batch, 225) flat."""
    print("[sanity_check] Running quantum discriminator sanity check...", flush=True)
    model.eval()
    x1 = torch.randn(2, 225).to(device)
    x2 = torch.randn(2, 225).to(device)
    x1.requires_grad_(True)

    out1 = model(x1)
    out2 = model(x2)

    assert out1.shape == (2, 1), f"Bad output shape: {out1.shape}"
    assert not torch.allclose(out1, out2, atol=1e-4), "Outputs identical for different inputs!"

    loss = out1.sum()
    loss.backward()
    assert x1.grad is not None and x1.grad.abs().sum() > 0, "Input gradients are zero!"

    print(f"[sanity_check] PASSED — out1={out1.detach().squeeze().tolist()}, "
          f"out2={out2.detach().squeeze().tolist()}", flush=True)
    model.train()
