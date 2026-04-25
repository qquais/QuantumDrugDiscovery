import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import random
import numpy as np

from rdkit import RDLogger
from utils.args import get_GAN_config
from utils.utils_io import get_date_postfix
from torch.backends import cudnn

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def main():
    cudnn.benchmark = True
    config = get_GAN_config()

    # ---- Kao et al. 2023 Table 1 — 150-epoch QuMolGAN (quantum noise generator) ----

    # Dataset
    # config.mol_data_dir = 'data/qm9_5k.sparsedataset'
    config.mol_data_dir = '/scratch/gilbreth/quaiqa01/QuantumDrugDiscovery/data/qm9_5k_py37.sparsedataset'

    # Quantum (ON — QuMolGAN noise generator, Kao et al. 2023 Table 1)
    config.quantum = True
    config.qubits = 4               # must equal z_dim: circuit returns qubits values → z
    config.layer = 3
    config.update_qc = True
    config.qc_lr = 0.04

    # Training
    config.mode = 'train'
    config.complexity = 'hr'
    config.g_conv_dim = [16]        # 'hr' → [16]
    config.batch_size = 16
    config.z_dim = 4                # must equal qubits (circuit output dim = z dim)
    config.num_epochs = 30
    config.n_critic = 5
    config.critic_type = 'D'
    config.lambda_wgan = 1.0        # pure WGAN (alpha=1.0)
    config.lambda_gp = 10.0
    config.decay_every_epoch = None
    config.g_lr = 0.001
    config.d_lr = 0.001
    config.use_quantum_disc = False

    # Quantum circuit (defined but never called when quantum=False)
    try:
        import pennylane as qml
        dev = qml.device('default.qubit', wires=config.qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def gen_circuit(w):
            z1 = random.uniform(-1, 1)
            z2 = random.uniform(-1, 1)
            for i in range(config.qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            for l in range(config.layer):
                for i in range(config.qubits):
                    qml.RY(w[i], wires=i)
                for i in range(config.qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[i + config.qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(config.qubits)]

        config.gen_circuit = gen_circuit
    except Exception as e:
        print(f"PennyLane unavailable ({e}); proceeding without gen_circuit.")
        config.gen_circuit = None

    # Directories
    run_id = get_date_postfix()
    config.saving_dir = os.path.join('results/quantum/GAN', run_id)
    config.log_dir_path = os.path.join(config.saving_dir, 'train', 'log_dir')
    config.model_dir_path = os.path.join(config.saving_dir, 'train', 'model_dir')
    config.img_dir_path = os.path.join(config.saving_dir, 'train', 'img_dir')
    for d in [config.log_dir_path, config.model_dir_path, config.img_dir_path]:
        os.makedirs(d, exist_ok=True)

    log_name = os.path.join(config.log_dir_path, f'{run_id}_logger.log')
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(config)

    print(config)

    from solver import Solver
    solver = Solver(config, logging)
    solver.train_and_validate()


if __name__ == '__main__':
    main()
