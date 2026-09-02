"""qmolgan — reusable components for the classical/quantum MolGAN study.

Everything that is shared between training, offline evaluation, figure
generation and the unit tests lives here, so that a metric can never be
computed one way during training and a different way during evaluation.
That divergence is what produced the withdrawn v1 numbers (see
docs/ERRATA.md).
"""

__all__ = ['chem', 'latent', 'rewards', 'protocol', 'generate']
