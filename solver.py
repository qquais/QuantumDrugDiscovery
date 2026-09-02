"""Training loop for classical and quantum MolGAN variants.

Rewritten from the original `solver_legacy.py` (kept for reference) to fix
the reporting defects found in the v1 audit (docs/ERRATA.md):

* Training-time "scores" were computed on the *training batch* of 16
  molecules every 10 steps. Uniqueness over 16 samples is near 1 by
  construction; that is where the withdrawn 73% uniqueness came from. This
  version evaluates a fixed, seeded validation sample of `val_n` molecules
  once per epoch and writes it to history.csv, and the header of that file
  says what n was.
* Reported SA/logP were the [0,1]-normalised reward-space versions. Raw
  values are logged now; normalisation lives only inside the reward.
* Per-metric means were taken over `np.nonzero(v)`, silently dropping every
  molecule that scored exactly 0 and biasing every average upward.
* The latent source is now a `qmolgan.latent.LatentSampler`, so classical,
  bounded-classical, rank-matched-classical and quantum runs share one code
  path and differ only in the sampler.
* Circuit weights are checkpointed as `{epoch}-Z.ckpt` state dicts instead of
  appended CSV rows (which desynced from epochs whenever a run was resumed).
"""

import csv
import datetime
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from data.sparse_molecular_dataset import SparseMolecularDataset
from models.models import Generator, Discriminator
from qmolgan import chem, generate as gen_utils, rewards as reward_utils
from qmolgan.latent import make_latent
from utils.logger import Logger
from utils.utils import MolecularMetrics, save_mol_img


class Solver(object):
    """Trains one MolGAN variant and logs an honest per-epoch history."""

    def __init__(self, config, log=None):
        self.config = config
        self.log = log

        # ---- Data -------------------------------------------------------
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)
        self.train_smiles = set(gen_utils.training_smiles(self.data))

        # ---- Model shape -------------------------------------------------
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.dropout = config.dropout
        self.post_method = config.post_method

        # ---- Objective ---------------------------------------------------
        self.la = config.lambda_wgan
        self.la_gp = config.lambda_gp
        self.reward_mode = config.reward_mode
        self.metric = config.metric
        self.enable_rl_loss = config.enable_rl_loss
        self.reward_weights = {k: float(getattr(config, k))
                               for k in reward_utils.REWARD_KEYS}
        self.rw_clip = (config.rw_clip_min, config.rw_clip_max)

        # ---- Training ----------------------------------------------------
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = max(1, len(self.data) // self.batch_size)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.gamma = config.gamma
        self.decay_every_epoch = config.decay_every_epoch
        self.n_critic = config.n_critic if self.la > 0 else 1
        self.critic_type = config.critic_type
        self.mode = config.mode
        self.resume_epoch = config.resume_epoch
        self.model_save_step = config.model_save_step
        self.seed = config.seed

        # Honest per-epoch validation: a fixed number of freshly generated
        # molecules, always the same count, always the same noise seed, so the
        # per-epoch curve is comparable across epochs, runs and models.
        self.val_n = config.val_n
        self.val_seed = config.val_seed
        self.val_every = max(1, config.val_every)

        self.gumbel_temp_start = config.gumbel_temp_start
        self.gumbel_temp_end = config.gumbel_temp_end

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}', flush=True)

        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path
        self.history_path = os.path.join(config.saving_dir, 'history.csv')

        self.use_tensorboard = config.use_tensorboard
        self.logger = Logger(self.log_dir_path) if (self.mode == 'train'
                                                    and self.use_tensorboard) else None

        self.build_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_model(self):
        self.G = Generator(self.g_conv_dim, self.z_dim, self.data.vertexes,
                           self.data.bond_num_types, self.data.atom_num_types,
                           self.dropout).to(self.device)
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1,
                               dropout_rate=self.dropout).to(self.device)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1,
                               dropout_rate=self.dropout).to(self.device)

        cfg = self.config
        self.latent = make_latent(cfg.latent, dim=cfg.z_dim, qubits=cfg.qubits,
                                  layers=cfg.layer, seed=cfg.seed, n_freq=cfg.n_freq)
        print(f'Latent source: {json.dumps(self.latent.describe())}', flush=True)

        # The latent source's parameters (if any) get their own learning rate:
        # a VQC needs a much larger step than the generator's dense stack.
        latent_params = [p for p in self.latent.parameters() if p.requires_grad]
        if latent_params and cfg.update_latent:
            lr = cfg.qc_lr if cfg.qc_lr else self.g_lr
            self.g_optimizer = torch.optim.RMSprop(
                [{'params': list(self.G.parameters())},
                 {'params': latent_params, 'lr': lr}], lr=self.g_lr)
        else:
            for p in latent_params:
                p.requires_grad_(False)
            self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), self.g_lr)

        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), self.d_lr)
        self.v_optimizer = torch.optim.RMSprop(self.V.parameters(), self.g_lr)

        for model, name in ((self.G, 'G'), (self.D, 'D'), (self.V, 'V')):
            n = sum(p.numel() for p in model.parameters())
            print(f'{name}: {n} parameters')
            if self.log is not None:
                self.log.info(f'{name}: {n} parameters')

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoints(self, epoch):
        """Save G/D/V and the latent state for ``epoch`` (1-based)."""
        torch.save(self.G.state_dict(), os.path.join(self.model_dir_path, f'{epoch}-G.ckpt'))
        torch.save(self.D.state_dict(), os.path.join(self.model_dir_path, f'{epoch}-D.ckpt'))
        torch.save(self.V.state_dict(), os.path.join(self.model_dir_path, f'{epoch}-V.ckpt'))
        torch.save(self.latent.state_dict(), os.path.join(self.model_dir_path, f'{epoch}-Z.ckpt'))

    def restore(self, epoch):
        for net, tag in ((self.G, 'G'), (self.D, 'D'), (self.V, 'V')):
            path = os.path.join(self.model_dir_path, f'{epoch}-{tag}.ckpt')
            net.load_state_dict(torch.load(path, map_location=self.device))
        gen_utils.load_latent_state(self.latent, self.model_dir_path, epoch)
        print(f'Restored epoch {epoch} from {self.model_dir_path}')

    # ------------------------------------------------------------------
    # Pieces of the objective
    # ------------------------------------------------------------------

    @staticmethod
    def postprocess(inputs, method, temperature=1.0):
        def listify(x):
            return x if isinstance(x, (list, tuple)) else [x]

        if method == 'soft_gumbel':
            out = [F.gumbel_softmax(e.contiguous().view(-1, e.size(-1)) / temperature,
                                    hard=False).view(e.size()) for e in listify(inputs)]
        elif method == 'hard_gumbel':
            out = [F.gumbel_softmax(e.contiguous().view(-1, e.size(-1)) / temperature,
                                    hard=True).view(e.size()) for e in listify(inputs)]
        else:
            out = [F.softmax(e / temperature, -1) for e in listify(inputs)]
        return out if len(out) > 1 else out[0]

    def label2onehot(self, labels, dim):
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def gradient_penalty(self, y, x):
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight,
                                   retain_graph=True, create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        return torch.mean((torch.sqrt(torch.sum(dydx ** 2, dim=1)) - 1) ** 2)

    def reward(self, mols):
        """RL reward. Weighted mode uses normalised SA/logP by design; those
        normalised values are never reported as metrics."""
        if self.reward_mode == 'weighted':
            return reward_utils.weighted_reward(mols, self.train_smiles,
                                                self.reward_weights, clip=self.rw_clip)
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):
            if m == 'logp':
                rr = rr * MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr = rr * MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr = rr * MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr = rr * MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'unique':
                rr = rr * MolecularMetrics.unique_scores(mols)
            elif m == 'validity':
                rr = rr * MolecularMetrics.valid_scores(mols)
            elif m == 'diversity':
                rr = rr * MolecularMetrics.diversity_scores(mols, self.data)
            else:
                raise RuntimeError(f'{m} is not a known reward metric')
        return np.asarray(rr, dtype=np.float32).reshape(-1, 1)

    def decode(self, nodes_hat, edges_hat):
        edges_hard = torch.max(edges_hat, -1)[1].detach().cpu().numpy()
        nodes_hard = torch.max(nodes_hat, -1)[1].detach().cpu().numpy()
        return chem.decode_batch(nodes_hard, edges_hard, self.data)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def temperature(self, epoch):
        span = max(1, self.num_epochs - 1)
        return self.gumbel_temp_start + (self.gumbel_temp_end - self.gumbel_temp_start) * (epoch / span)

    def train_and_validate(self):
        self.start_time = time.time()
        start_epoch = 0
        if self.resume_epoch is not None and self.mode == 'train':
            start_epoch = self.resume_epoch
            self.restore(self.resume_epoch)
        elif self.config.test_epoch is not None and self.mode == 'test':
            self.restore(self.config.test_epoch)

        if self.mode == 'test':
            metrics = self.validate(self.config.test_epoch or start_epoch)
            print(json.dumps(metrics, indent=2))
            return

        self._init_history()
        for epoch in range(start_epoch, self.num_epochs):
            losses = self.train_one_epoch(epoch)
            if (epoch + 1) % self.model_save_step == 0:
                self.save_checkpoints(epoch + 1)
            if (epoch + 1) % self.val_every == 0 or epoch + 1 == self.num_epochs:
                metrics = self.validate(epoch + 1)
                self._append_history(epoch + 1, losses, metrics)
                self._report(epoch, losses, metrics)

    def train_one_epoch(self, epoch):
        temp = self.temperature(epoch)
        losses = defaultdict(list)

        for a_step in range(self.num_steps):
            cur_step = self.num_steps * epoch + a_step
            real_mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
            a = torch.from_numpy(a).to(self.device).long()
            x = torch.from_numpy(x).to(self.device).long()
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            z = self.latent.sample(a.size(0), device=self.device).float()

            # ---- Discriminator (WGAN-GP) --------------------------------
            logits_real, _ = self.D(a_tensor, None, x_tensor)
            edges_logits, nodes_logits = self.G(z)
            edges_hat, nodes_hat = self.postprocess((edges_logits, nodes_logits),
                                                    self.post_method, temp)
            logits_fake, _ = self.D(edges_hat, None, nodes_hat)

            eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor
                      + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            grad_logits, _ = self.D(x_int0, None, x_int1)
            grad_penalty = (self.gradient_penalty(grad_logits, x_int0)
                            + self.gradient_penalty(grad_logits, x_int1))

            d_loss_real = torch.mean(logits_real)
            d_loss_fake = torch.mean(logits_fake)
            # E[D(fake)] - E[D(real)] + lambda * GP. The sign of the first two
            # terms is the Wasserstein estimate; summing them (as some earlier
            # implementations do) cancels the critic's gradient signal.
            loss_D = -d_loss_real + d_loss_fake + self.la_gp * grad_penalty

            train_d = ((cur_step == 0 or cur_step % self.n_critic != 0)
                       if self.critic_type == 'D'
                       else (cur_step != 0 and cur_step % self.n_critic == 0))
            if train_d:
                self.zero_grad()
                loss_D.backward()
                self.d_optimizer.step()

            # ---- Generator + value net ----------------------------------
            edges_logits, nodes_logits = self.G(z)
            edges_hat, nodes_hat = self.postprocess((edges_logits, nodes_logits),
                                                    self.post_method, temp)
            logits_fake, _ = self.D(edges_hat, None, nodes_hat)
            value_real, _ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
            value_fake, _ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)

            loss_G = torch.mean(-logits_fake)
            loss_RL = torch.mean(-value_fake)
            if self.la < 1.0:
                # Value net regresses the reward of the generated molecules and
                # of the real molecules in THIS batch (real_mols), so the two
                # regression targets come from the same draw as the graphs the
                # critic just saw.
                reward_f = torch.from_numpy(
                    self.reward(self.decode(nodes_hat, edges_hat))).to(self.device)
                reward_r = torch.from_numpy(
                    self.reward(list(real_mols))).to(self.device)
                loss_V = torch.mean(torch.abs(value_real - reward_r)
                                    + torch.abs(value_fake - reward_f))
            else:
                loss_V = torch.tensor(0.0, device=self.device)

            train_g = ((cur_step != 0 and cur_step % self.n_critic == 0)
                       if self.critic_type == 'D'
                       else (cur_step == 0 or cur_step % self.n_critic != 0))
            if train_g:
                self.zero_grad()
                if self.la < 1.0 and self.enable_rl_loss:
                    alpha = torch.abs(loss_G.detach() / (loss_RL.detach() + 1e-8)).detach()
                    step_G = self.la * loss_G + (1.0 - self.la) * alpha * loss_RL
                    step_G.backward(retain_graph=True)
                    loss_V.backward()
                    self.g_optimizer.step()
                    self.v_optimizer.step()
                else:
                    (self.la * loss_G).backward()
                    self.g_optimizer.step()

            losses['D/loss'].append(loss_D.item())
            losses['D/real'].append(d_loss_real.item())
            losses['D/fake'].append(d_loss_fake.item())
            losses['D/gp'].append(grad_penalty.item())
            losses['G/loss'].append(loss_G.item())
            losses['RL/loss'].append(loss_RL.item())
            losses['V/loss'].append(float(loss_V.item()))

            if self.logger is not None:
                for tag, val in (('D/loss', loss_D.item()), ('G/loss', loss_G.item())):
                    self.logger.scalar_summary(tag, val, cur_step)

        if self.decay_every_epoch and epoch != 0 and (epoch + 1) % self.decay_every_epoch == 0:
            self.update_lr(self.gamma)

        return {k: float(np.mean(v)) for k, v in losses.items()}

    def zero_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

    def update_lr(self, gamma):
        for opt in (self.d_optimizer, self.g_optimizer, self.v_optimizer):
            for group in opt.param_groups:
                group['lr'] *= gamma

    # ------------------------------------------------------------------
    # Honest validation
    # ------------------------------------------------------------------

    def validate(self, epoch):
        """Generate a fixed, seeded sample of `val_n` molecules and score them
        with the same metric code the offline evaluation uses.

        The noise seed is constant across epochs so the per-epoch curve is not
        contaminated by noise-draw variance, and it is offset far from the
        protocol's selection/report streams so that in-training monitoring can
        never be mistaken for, or leak into, a reported number.
        """
        self.G.eval()
        mols = gen_utils.sample_molecules(
            self.G, self.latent, self.data, self.val_n,
            batch_size=min(256, self.val_n), seed=self.val_seed,
            post_method=self.post_method, device=self.device)
        self.G.train()

        smi = [chem.canonical_smiles(m) for m in mols]
        valid = [s for m, s in zip(mols, smi) if chem.is_valid(m) and s]
        clean = [s for m, s in zip(mols, smi) if chem.is_clean_valid(m) and s]
        clean_mols = [m for m in mols if chem.is_clean_valid(m)]
        props = chem.raw_properties(clean_mols)
        n = len(mols)

        def frac(sub, denom):
            return len(sub) / denom if denom else float('nan')

        metrics = {
            'n_sampled': n,
            'validity': frac(valid, n),
            'clean_validity': frac(clean, n),
            'uniqueness': frac(set(valid), len(valid)) if valid else float('nan'),
            'uniqueness_clean': frac(set(clean), len(clean)) if clean else float('nan'),
            'novelty': (sum(s not in self.train_smiles for s in valid) / len(valid)
                        if valid else float('nan')),
            'novelty_clean': (sum(s not in self.train_smiles for s in clean) / len(clean)
                              if clean else float('nan')),
        }
        for key in chem.PROPERTY_KEYS:
            vals = props[key]
            metrics[key] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float('nan')

        if self.img_dir_path and clean_mols:
            save_mol_img(clean_mols[:8], os.path.join(self.img_dir_path, f'mol-{epoch}.png'))
        return metrics

    # ------------------------------------------------------------------
    # History / logging
    # ------------------------------------------------------------------

    HISTORY_FIELDS = ('epoch', 'n_sampled', 'validity', 'clean_validity', 'uniqueness',
                      'uniqueness_clean', 'novelty', 'novelty_clean',
                      'QED', 'logP', 'SA', 'MW',
                      'D/loss', 'D/real', 'D/fake', 'D/gp', 'G/loss', 'RL/loss', 'V/loss')

    def _init_history(self):
        if os.path.exists(self.history_path):
            return
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.HISTORY_FIELDS)

    def _append_history(self, epoch, losses, metrics):
        row = {'epoch': epoch, **metrics, **losses}
        with open(self.history_path, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(k, '') for k in self.HISTORY_FIELDS])

    def _report(self, epoch, losses, metrics):
        et = str(datetime.timedelta(seconds=time.time() - self.start_time))[:-7]
        head = (f'[{et}] epoch {epoch + 1}/{self.num_epochs}  '
                f'(val n={metrics["n_sampled"]}, seed={self.val_seed})')
        body = ('  ' + '  '.join(f'{k}={metrics[k]:.4f}' for k in
                                 ('validity', 'clean_validity', 'uniqueness_clean',
                                  'novelty_clean', 'QED', 'SA')
                                 if np.isfinite(metrics.get(k, np.nan))))
        loss_line = '  ' + '  '.join(f'{k}={v:.3f}' for k, v in losses.items())
        print(head + '\n' + body + '\n' + loss_line, flush=True)
        if self.log is not None:
            self.log.info(head + body + loss_line)
