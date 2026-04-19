from rdkit import Chem
from rdkit.Chem import QED, Crippen
import sascorer  # you already have SA_score.pkl.gz etc., so you should have this script too

import torch


# === IMPORTANT ===
# Use the same atom and bond setup as in your preprocessing.
# This is a common QM9-style example; if your dataset uses a different list, edit here.
ATOM_LIST = ["C", "N", "O", "F", "S", "Cl"]

# edge index 0 = no bond, 1..len(BOND_LIST) = these bond types
BOND_LIST = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def graph_to_mol_single(node_idx, edge_idx, n_nodes):
    """
    Convert a single graph to an RDKit Mol.

    node_idx: 1D tensor/array of shape (n_nodes,)  with atom-type indices
    edge_idx: 2D tensor/array of shape (n_nodes, n_nodes) with bond-type indices
              0 = no bond, 1..K = bond types from BOND_LIST.
    """
    mol = Chem.RWMol()
    atom_indices = []

    # 1) Add atoms
    for i in range(n_nodes):
        atom_type = int(node_idx[i])
        if atom_type < 0 or atom_type >= len(ATOM_LIST):
            return None
        atom = Chem.Atom(ATOM_LIST[atom_type])
        atom_indices.append(mol.AddAtom(atom))

    # 2) Add bonds (upper triangle only, then RDKit treats as undirected)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            btype_idx = int(edge_idx[i, j])
            if btype_idx == 0:
                continue  # no bond
            if not (1 <= btype_idx <= len(BOND_LIST)):
                return None
            bond_type = BOND_LIST[btype_idx - 1]
            try:
                mol.AddBond(atom_indices[i], atom_indices[j], bond_type)
            except Exception:
                return None

    # 3) Sanitize the molecule (check valence, aromaticity, etc.)
    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return mol


def decode_batch_to_mols(nodes_tensor: torch.Tensor,
                         edges_tensor: torch.Tensor):
    """
    Convert a batch of graphs to RDKit molecules.

    nodes_tensor: (N, n_nodes)   indices for atom types
    edges_tensor: (N, n_nodes, n_nodes) indices for bond types
    """
    nodes_np = nodes_tensor.cpu().numpy()
    edges_np = edges_tensor.cpu().numpy()

    n_samples, n_nodes = nodes_np.shape
    mols = []

    for i in range(n_samples):
        mol = graph_to_mol_single(nodes_np[i], edges_np[i], n_nodes)
        mols.append(mol)

    return mols


def _safe_mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def evaluate_molecules(mols):
    """
    Compute validity, uniqueness, QED, logP, and SA over a list of RDKit mols.
    """
    total = len(mols)
    valid_mols = [m for m in mols if m is not None]
    n_valid = len(valid_mols)

    # SMILES to measure uniqueness
    smiles_list = []
    for m in valid_mols:
        try:
            s = Chem.MolToSmiles(m)
            smiles_list.append(s)
        except Exception:
            continue

    unique_smiles = set(smiles_list)

    validity = n_valid / total if total > 0 else 0.0
    uniqueness = len(unique_smiles) / n_valid if n_valid > 0 else 0.0

    qed_vals, logp_vals, sa_vals = [], [], []

    for m in valid_mols:
        try:
            qed_vals.append(QED.qed(m))
            logp_vals.append(Crippen.MolLogP(m))
            sa_vals.append(sascorer.calculateScore(m))
        except Exception:
            continue

    metrics = {
        "num_samples": total,
        "num_valid": n_valid,
        "validity": validity,
        "uniqueness": uniqueness,
        "QED_mean": _safe_mean(qed_vals),
        "logP_mean": _safe_mean(logp_vals),
        "SA_mean": _safe_mean(sa_vals),
    }
    return metrics
