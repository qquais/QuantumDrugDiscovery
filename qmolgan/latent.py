"""Latent noise sources for the generator, and their classical controls.

The v1 pipeline compared a VQC noise source against a classical
Gaussian that differed in *four* ways at once: distribution family, support
(bounded vs unbounded), latent dimension, and intrinsic stochastic
dimension. No claim about "quantum advantage" survives that confound, which
is the central confound the v1 audit identified.

This module makes every one of those axes a separately controllable knob so
that the ablations can isolate them:

  gaussian   z ~ N(0, I_d)                      unbounded, d independent dims
  uniform    z ~ U(-1, 1)^d                     bounded,   d independent dims
  rank2      z = f(u1, u2), u ~ U(-1, 1)^2      bounded,   2 intrinsic dims
  trig       classical trigonometric surrogate  bounded,   2 intrinsic dims,
             with the same trainable-parameter count as the VQC
  vqc        Kao et al. 2023 variational circuit (entangling)
  vqc_noent  the same circuit with the CNOT blocks removed

The `rank2`/`trig` controls matter because the Kao circuit draws only TWO
scalars (z1, z2 ~ U(-1,1)) per sample and encodes them on every wire: the
n_qubits circuit outputs are a deterministic function of two random numbers,
so the quantum latent has intrinsic dimension 2 regardless of qubit count.
Any diversity difference against a 4-D Gaussian is therefore expected from
dimensionality alone, before entanglement is invoked as an explanation.
Moreover a Pauli-Z expectation of this circuit family is a finite
trigonometric polynomial in the encoded angles, so `trig` is not a loose
analogy — it is a classical model of the same function class.
"""

import math

import numpy as np
import torch
import torch.nn as nn

LATENT_KINDS = ('gaussian', 'uniform', 'rank2', 'trig', 'vqc', 'vqc_noent')


class LatentSampler(nn.Module):
    """Common interface: ``sample(batch)`` -> (batch, dim) float tensor on device.

    Subclasses that carry trainable parameters expose them through the usual
    ``nn.Module`` machinery, so the solver can hand them to an optimiser with
    a per-group learning rate and checkpoint them with ``state_dict()``.
    """

    kind = 'abstract'
    trainable = False

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def sample(self, batch_size, device=None):
        raise NotImplementedError

    def describe(self):
        return {'kind': self.kind, 'dim': self.dim, 'trainable': self.trainable,
                'n_params': sum(p.numel() for p in self.parameters())}


# ---------------------------------------------------------------------------
# Classical baselines and controls
# ---------------------------------------------------------------------------

class GaussianLatent(LatentSampler):
    """The MolGAN default: z ~ N(0, I). Unbounded, full-rank."""

    kind = 'gaussian'

    def sample(self, batch_size, device=None):
        z = torch.randn(batch_size, self.dim)
        return z.to(device) if device is not None else z


class UniformLatent(LatentSampler):
    """Bounded control: z ~ U(low, high)^d, matching the VQC's [-1, 1] support
    without matching its rank. Isolates 'boundedness' from 'quantum'."""

    kind = 'uniform'

    def __init__(self, dim, low=-1.0, high=1.0):
        super().__init__(dim)
        self.low, self.high = float(low), float(high)

    def sample(self, batch_size, device=None):
        z = torch.rand(batch_size, self.dim) * (self.high - self.low) + self.low
        return z.to(device) if device is not None else z

    def describe(self):
        return {**super().describe(), 'low': self.low, 'high': self.high}


class Rank2Latent(LatentSampler):
    """Intrinsic-dimension control: two random scalars, fixed random lift to d
    bounded dimensions. Matches the VQC's rank-2 stochasticity and [-1, 1]
    support with no trainable parameters at all."""

    kind = 'rank2'

    def __init__(self, dim, seed=0):
        super().__init__(dim)
        g = torch.Generator().manual_seed(int(seed))
        # Fixed (buffer, not parameter) random projection so the control is
        # deterministic across seeds of the *training* run and travels with
        # the checkpoint.
        self.register_buffer('freq', torch.rand(dim, 2, generator=g) * 2.0 + 0.5)
        self.register_buffer('phase', torch.rand(dim, generator=g) * 2 * math.pi)

    def sample(self, batch_size, device=None):
        u = torch.rand(batch_size, 2) * 2 - 1
        theta = torch.asin(u)                                   # (B, 2)
        z = torch.cos(theta @ self.freq.cpu().T + self.phase.cpu())
        return z.to(device) if device is not None else z


class TrigLatent(LatentSampler):
    """Classical trigonometric surrogate of the VQC.

    A Pauli-Z expectation of the Kao circuit is a finite trigonometric
    polynomial in the two encoded angles, with frequency support set by the
    circuit depth. This module learns exactly such a polynomial with a
    matched trainable-parameter budget, giving the strongest available
    classical control: if it matches the VQC, the VQC's contribution is its
    function class and not its quantumness.
    """

    kind = 'trig'
    trainable = True

    def __init__(self, dim, n_freq=3, seed=0):
        super().__init__(dim)
        self.n_freq = int(n_freq)
        g = torch.Generator().manual_seed(int(seed))
        # coeff: (dim, n_freq) amplitudes; freq: (n_freq, 2) integer-ish
        # frequencies; phase: (n_freq,).
        self.coeff = nn.Parameter(torch.rand(dim, self.n_freq, generator=g) * 2 - 1)
        self.phase = nn.Parameter(torch.rand(self.n_freq, generator=g) * 2 * math.pi)
        self.register_buffer(
            'freq', torch.arange(1, self.n_freq + 1, dtype=torch.float32).unsqueeze(-1)
                      .repeat(1, 2) * torch.tensor([1.0, 1.0]))

    def sample(self, batch_size, device=None):
        u = torch.rand(batch_size, 2) * 2 - 1
        theta = torch.asin(u)                                   # (B, 2)
        basis = torch.cos(theta @ self.freq.cpu().T + self.phase.cpu())   # (B, n_freq)
        z = torch.tanh(basis @ self.coeff.cpu().T)              # (B, dim), bounded
        return z.to(device) if device is not None else z

    def describe(self):
        return {**super().describe(), 'n_freq': self.n_freq}


# ---------------------------------------------------------------------------
# Quantum
# ---------------------------------------------------------------------------

def build_gen_circuit(qubits, layers, entangle=True, device_name='default.qubit',
                      diff_method='backprop'):
    """The Kao et al. 2023 QuMolGAN noise circuit, with entanglement optional.

    Structure per sample:
      1. draw z1, z2 ~ U(-1, 1)  (the ONLY source of sample-to-sample noise)
      2. encode RY(arcsin z1), RZ(arcsin z2) on every wire
      3. ``layers`` blocks of RY(w_i) on each wire, then (if ``entangle``)
         CNOT - RZ(w_{i+q}) - CNOT between adjacent wires
      4. measure <Z_i> on every wire -> z in [-1, 1]^qubits

    Setting ``entangle=False`` removes step 3's CNOT-RZ-CNOT blocks, leaving a
    product circuit. That is the ablation that tests the paper's central
    mechanistic claim ("entangled states explore better latent regions"),
    which the v1 analysis asserted without measuring.
    """
    import pennylane as qml

    dev = qml.device(device_name, wires=qubits)

    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def gen_circuit(w):
        z1 = float(torch.rand(1).item()) * 2 - 1
        z2 = float(torch.rand(1).item()) * 2 - 1
        for i in range(qubits):
            qml.RY(np.arcsin(z1), wires=i)
            qml.RZ(np.arcsin(z2), wires=i)
        for _ in range(layers):
            for i in range(qubits):
                qml.RY(w[i], wires=i)
            if entangle:
                for i in range(qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[i + qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

    return gen_circuit


def n_circuit_weights(qubits, layers, entangle=True):
    """Parameter count of the circuit above.

    The entangling variant consumes ``qubits`` RY weights plus ``qubits - 1``
    RZ weights per layer, but the original Kao indexing reads them from one
    flat vector of length ``layers * (2 * qubits - 1)`` and reuses the same
    slice every layer; that shape is preserved here so old checkpoints load.
    """
    return layers * (2 * qubits - 1)


class VQCLatent(LatentSampler):
    """Variational quantum circuit as the generator's noise source."""

    kind = 'vqc'
    trainable = True

    def __init__(self, qubits, layers, entangle=True, seed=None, weights=None,
                 device_name='default.qubit'):
        super().__init__(qubits)
        self.qubits, self.layers, self.entangle = int(qubits), int(layers), bool(entangle)
        self.kind = 'vqc' if entangle else 'vqc_noent'
        n_w = n_circuit_weights(self.qubits, self.layers, self.entangle)
        if weights is None:
            g = np.random.RandomState(seed) if seed is not None else np.random
            weights = g.rand(n_w) * 2 * np.pi - np.pi
        self.weights = nn.Parameter(torch.tensor(np.asarray(weights, dtype=np.float64)))
        self._circuit = build_gen_circuit(self.qubits, self.layers, self.entangle,
                                          device_name=device_name)

    def sample(self, batch_size, device=None):
        # PennyLane returns a list of per-wire tensors when the qnode has
        # multiple expval outputs; stack them into one vector per sample.
        rows = []
        for _ in range(batch_size):
            out = self._circuit(self.weights)
            rows.append(torch.stack(out) if isinstance(out, (list, tuple)) else out)
        z = torch.stack(tuple(rows)).float()
        return z.to(device) if device is not None else z

    def describe(self):
        return {**super().describe(), 'qubits': self.qubits, 'layers': self.layers,
                'entangle': self.entangle}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_latent(kind, dim=None, qubits=None, layers=3, seed=0, n_freq=3,
                device_name='default.qubit'):
    """Build a latent sampler by name. ``dim`` defaults to ``qubits`` for VQCs."""
    kind = kind.lower()
    if kind in ('vqc', 'vqc_noent'):
        if qubits is None:
            qubits = dim
        if dim is not None and int(dim) != int(qubits):
            raise ValueError(
                f'z_dim ({dim}) must equal qubits ({qubits}) for a VQC latent: '
                'the circuit emits one expectation value per wire.')
        return VQCLatent(qubits, layers, entangle=(kind == 'vqc'), seed=seed,
                         device_name=device_name)
    if dim is None:
        raise ValueError(f'latent kind {kind!r} requires an explicit dim')
    if kind == 'gaussian':
        return GaussianLatent(dim)
    if kind == 'uniform':
        return UniformLatent(dim)
    if kind == 'rank2':
        return Rank2Latent(dim, seed=seed)
    if kind == 'trig':
        return TrigLatent(dim, n_freq=n_freq, seed=seed)
    raise ValueError(f'unknown latent kind {kind!r}; expected one of {LATENT_KINDS}')


def latent_statistics(sampler, n=4096, seed=0):
    """Empirical statistics of a latent source, for the latent-space analysis
    the v1 analysis asserted but never measured.

    Returns per-dimension mean/std, the correlation matrix, the eigenvalue
    spectrum of the covariance (which exposes the rank-2 structure of the VQC
    directly), participation ratio as an effective-dimension summary, and a
    differential-entropy estimate.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        z = sampler.sample(n).double().cpu().numpy()
    mean = z.mean(0)
    std = z.std(0)
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(std, std)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig_pos = np.clip(eig, 0, None)
    total = eig_pos.sum()
    participation = float((total ** 2) / np.sum(eig_pos ** 2)) if total > 0 else float('nan')
    # Kozachenko-Leonenko style entropy proxy: log-det of the covariance is
    # enough to show a degenerate (rank-deficient) latent without extra deps.
    sign, logdet = np.linalg.slogdet(cov + 1e-12 * np.eye(cov.shape[0]))
    return {
        'n': int(n),
        'mean': mean.tolist(),
        'std': std.tolist(),
        'corr': np.nan_to_num(corr).tolist(),
        'cov_eigenvalues': eig.tolist(),
        'participation_ratio': participation,
        'log_det_cov': float(logdet) if sign > 0 else float('-inf'),
        'effective_rank_99pct': int(np.searchsorted(np.cumsum(eig_pos) / total, 0.99) + 1)
                                 if total > 0 else 0,
    }
