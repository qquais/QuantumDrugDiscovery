import os
import logging

from rdkit import RDLogger

from utils.args import get_GAN_config
from utils.utils_io import get_date_postfix

# Remove flooding logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from solver import Solver
from torch.backends import cudnn

import pennylane as qml
import random
import numpy as np

# def main(config):

#     # For fast training
#     cudnn.benchmark = True

#     # Timestamp
#     if config.mode == 'train':
#         a_train_time = get_date_postfix()
#         config.saving_dir = os.path.join(config.saving_dir, a_train_time)
#         config.log_dir_path = os.path.join(config.saving_dir, config.mode, 'log_dir')
#         config.model_dir_path = os.path.join(config.saving_dir, config.mode, 'model_dir')
#         config.img_dir_path = os.path.join(config.saving_dir, config.mode, 'img_dir')
#     else:
#         a_test_time = get_date_postfix()
#         config.saving_dir = os.path.join(config.saving_dir)
#         config.log_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'log_dir')
#         config.model_dir_path = os.path.join(config.saving_dir, 'model_dir')
#         config.img_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'img_dir')


#     # Create directories if not exist
#     if not os.path.exists(config.log_dir_path):
#         os.makedirs(config.log_dir_path)
#     if not os.path.exists(config.model_dir_path):
#         os.makedirs(config.model_dir_path)
#     if not os.path.exists(config.img_dir_path):
#         os.makedirs(config.img_dir_path)

#     # Logger
#     if config.mode == 'train':
#         log_p_name = os.path.join(config.log_dir_path, a_train_time + '_logger.log')
#         from imp import reload
#         reload(logging)
#         logging.basicConfig(filename=log_p_name, level=logging.INFO)
#         logging.info(config)
#     elif config.mode == 'test':
#         log_p_name = os.path.join(config.log_dir_path, a_test_time + '_logger.log')
#         from imp import reload
#         reload(logging)
#         logging.basicConfig(filename=log_p_name, level=logging.INFO)
#         logging.info(config)
#     else:
#         raise NotImplementedError

#     # Solver for training and test MolGAN
#     if config.mode == 'train':
#         solver = Solver(config, logging)
#     elif config.mode == 'test':
#         solver = Solver(config, logging)
#     else:
#         raise NotImplementedError

#     solver.train_and_validate()

# if __name__ == '__main__':

#     config = get_GAN_config()

#     # GPU
#     os.environ["CUDA_VISIBLE_DEVICES"]="5"


#     # Dataset
#     # molecule dataset dir
#     config.mol_data_dir = r'data/gdb9_9nodes.sparsedataset'
#     #config.mol_data_dir = r'data/qm9_5k.sparsedataset'


#     # Quantum
#     # quantum circuit to generate inputs of MolGAN
#     config.quantum = False
#     # number of qubit of quantum circuit
#     config.qubits = 8
#     # number of layer of quantum circuit
#     config.layer = 3
#     # update the parameters of quantum circuit
#     config.update_qc = False
#     # the learning rate of quantum circuit
#     # None: same learning rate as g_lr
#     config.qc_lr = 0.04
#     # to use pretrained quantum circuit or not
#     # config.qc_pretrained = False


#     # Training
#     config.mode = 'train'
#     # the complexity of generator
#     config.complexity = 'mr'
#     # batch size
#     config.batch_size = 16
#     # input noise dimension
#     config.z_dim = 8
#     # number of epoch
#     config.num_epochs = 300
#     # n_critic
#     config.n_critic = 3
#     # critic type
#     config.critic_type = 'D'
#     # 1.0 for pure WGAN and 0.0 for pure RL
#     config.lambda_wgan = 1
#     # weight decay
#     config.decay_every_epoch = 60
#     config.gamma = 0.1


#     # Testing
#     #config.mode = "test"
#     #config.complexity = 'mr'
#     #config.test_sample_size = 5000
#     #config.z_dim = 8
#     #config.test_epoch = 30
#     # MolGAN
#     #config.saving_dir = r"results/GAN/20211014_151730/train"
#     # Quantum
#     #config.saving_dir = r"results/quantum-GAN/20211130_102404/train"


#     if config.complexity == 'nr':
#         config.g_conv_dim = [128, 256, 512]
#     elif config.complexity == 'mr':
#         config.g_conv_dim = [128]
#     elif config.complexity == 'hr':
#         config.g_conv_dim = [16]
#     else:
#         raise ValueError("Please enter an valid model complexity from 'mr', 'hr' or 'nr'!")


#     # Quantum directory
#     if config.quantum and config.mode == 'train':
#         config.saving_dir = 'results/quantum-GAN'

#     # Quantum Circuit
#     dev = qml.device('default.qubit', wires=config.qubits)
#     @qml.qnode(dev, interface='torch', diff_method='backprop')
#     def gen_circuit(w):
#         # random noise as generator input
#         z1 = random.uniform(-1, 1)
#         z2 = random.uniform(-1, 1)
#         # construct generator circuit for both atom vector and node matrix
#         for i in range(config.qubits):
#             qml.RY(np.arcsin(z1), wires=i)
#             qml.RZ(np.arcsin(z2), wires=i)
#         for l in range(config.layer):
#             for i in range(config.qubits):
#                 qml.RY(w[i], wires=i)
#             for i in range(config.qubits-1):
#                 qml.CNOT(wires=[i, i+1])
#                 qml.RZ(w[i+config.qubits], wires=i+1)
#                 qml.CNOT(wires=[i, i+1])
#         return [qml.expval(qml.PauliZ(i)) for i in range(config.qubits)]

#     config.gen_circuit = gen_circuit

#     print(config)

#     main(config)


def main():
    """Load and analyze saved checkpoints without retraining."""

    # Load configuration
    config = get_GAN_config()

    # Ensure gen_circuit exists in config to prevent missing attribute errors
    if not hasattr(config, "gen_circuit"):
        config.gen_circuit = None

    # Ensure required directory paths exist in config
    if not hasattr(config, "model_dir_path"):
        # Auto-find the latest run under config.saving_dir (set via --saving_dir or default in args.py)
        gan_base = config.saving_dir.rstrip("/")
        if os.path.isdir(gan_base):
            runs = sorted([d for d in os.listdir(gan_base) if os.path.isdir(os.path.join(gan_base, d))], reverse=True)
            if runs:
                latest = os.path.join(gan_base, runs[0], "train")
                config.log_dir_path = os.path.join(latest, "log_dir")
                config.model_dir_path = os.path.join(latest, "model_dir")
                config.img_dir_path = os.path.join(latest, "img_dir")
                print(f"Using latest run: {latest}")
            else:
                raise FileNotFoundError(f"No training runs found under {gan_base}")
        else:
            raise FileNotFoundError(f"Results directory not found: {gan_base}")


    # Initialize Solver (but do not train)
    solver = Solver(config)

    # Step 1: List available checkpoints
    print("\n🔍 Step 1: Listing available checkpoints...")
    solver.list_checkpoints()

    # Step 2: Analyze specific checkpoints
    print("\n📊 Step 2: Analyzing selected checkpoints...")
    solver.analyze_checkpoint(1)   # Analyze epoch 1
    solver.analyze_checkpoint(100) # Analyze epoch 100
    solver.analyze_checkpoint(300) # Analyze epoch 300

    # Step 3: Generate a molecule from a checkpoint
    print("\n🧪 Step 3: Generating Molecules from Checkpoints...")
    solver.generate_molecule(300)  # Generate from epoch 300

if __name__ == "__main__":
    main()
