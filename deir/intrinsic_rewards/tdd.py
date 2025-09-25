import gym
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from deir.common_models.cnns import CNN_FEATURE_EXTRACTOR
from deir.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from deir.common_models.mlps import *
from deir.utils.enum_types import NormType
from deir.utils.common_func import init_module_with_name
from deir.utils.running_mean_std import RunningMeanStd
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor


def mrn_distance(x, y):

    eps = 1e-8
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = torch.max(F.relu(x_prefix - y_prefix), axis=-1).values
    l2_component = torch.sqrt(torch.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    return max_component + l2_component


class TDDModel(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        tdd_aggregate_fn: bool = 'min',
        tdd_energy_fn: str = 'mrn_pot',
        tdd_loss_fn: str = 'infonce',
        tdd_logsumexp_coef: float = 0,
        offpolicy_data: int = 0,
        batch_size=None,
        device='cuda',
    ):
        model_cnn_features_extractor_kwargs = dict(model_cnn_features_extractor_kwargs)
        if isinstance(model_cnn_features_extractor_class, str):
            model_cnn_features_extractor_class = CNN_FEATURE_EXTRACTOR[model_cnn_features_extractor_class]
        if model_cnn_features_extractor_kwargs is None:
            model_cnn_features_extractor_kwargs = {}

        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)
        self.aggregate_fn = tdd_aggregate_fn
        self.energy_fn = tdd_energy_fn
        self.loss_fn = tdd_loss_fn
        self.offpolicy_data = offpolicy_data
        self.temperature = 1.
        self.knn_k = 10
        self.logsumexp_coef = tdd_logsumexp_coef

        assert batch_size is not None
        self.batch_size = batch_size
        self.device = device

        self._build()
        self._init_modules()
        self._init_optimizers()

    def _init_modules(self) -> None:
        module_names = {
            self.model_cnn_extractor: 'model_cnn_extractor',
            self.potential_net: 'model_potential_net',
            self.encoder: 'model_net',
        }
        for module, name in module_names.items():
            init_module_with_name(name, module)

    def _init_optimizers(self) -> None:
        param_dicts = dict(self.named_parameters(recurse=True)).items()
        self.model_params = [
            param for name, param in param_dicts
        ]
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        self.model_optimizer = self.optimizer_class(
            self.model_params, lr=self.model_learning_rate, **self.optimizer_kwargs)

    def _build(self) -> None:

        self.model_cnn_features_extractor_kwargs.update(dict(
            features_dim=self.model_features_dim,
        ))
        self.model_cnn_extractor = \
            self.model_cnn_features_extractor_class(
                self.observation_space,
                **self.model_cnn_features_extractor_kwargs
            )
        self.encoder = ModelOutputHeads(
            feature_dim=self.model_features_dim,
            latent_dim=self.model_latents_dim,
            activation_fn=self.activation_fn,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
            output_dim=64,
        )
        self.potential_net = ModelOutputHeads(
            feature_dim=self.model_features_dim,
            latent_dim=self.model_latents_dim,
            activation_fn=self.activation_fn,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
            output_dim=1,
        )

    def forward(self,
                curr_obs: Tensor, future_obs: Tensor
                ):

        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(future_obs)
        phi_x = self.encoder(curr_cnn_embs)
        phi_y = self.encoder(next_cnn_embs)
        c_y = self.potential_net(next_cnn_embs)
        device = phi_x.device

        if self.energy_fn == 'l2':
            logits = - torch.sqrt(((phi_x[:, None] - phi_y[None, :])**2).sum(dim=-1) + 1e-8)
        elif self.energy_fn == 'cos':
            s_norm = torch.linalg.norm(phi_x, axis=-1, keepdims=True)
            g_norm = torch.linalg.norm(phi_y, axis=-1, keepdims=True)
            phi_x_norm = phi_x / s_norm
            phi_y_norm = phi_y / g_norm
            phi_x_norm = phi_x_norm / self.temperature
            logits = torch.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
        elif self.energy_fn == 'dot':
            logits = torch.einsum("ik,jk->ij", phi_x, phi_y)
        elif self.energy_fn == 'mrn':
            logits = - mrn_distance(phi_x[:, None], phi_y[None, :])
        elif self.energy_fn == 'mrn_pot':
            logits = c_y.T - mrn_distance(phi_x[:, None], phi_y[None, :])

        batch_size = logits.size(0)
        I = torch.eye(batch_size, device=device)

        if self.loss_fn == 'infonce':
            contrastive_loss = F.cross_entropy(logits, I)
        elif self.loss_fn == 'infonce_backward':
            contrastive_loss = F.cross_entropy(logits.T, I)
        elif self.loss_fn == 'infonce_symmetric':
            contrastive_loss = (F.cross_entropy(logits, I) + F.cross_entropy(logits.T, I)) / 2
        elif self.loss_fn == 'dpo':
            positive = torch.diag(logits)
            diffs = positive[:, None] - logits
            contrastive_loss = -F.logsigmoid(diffs)

        contrastive_loss = torch.mean(contrastive_loss)

        logs = {
            'contrastive_loss': contrastive_loss,
            'logits_pos': torch.diag(logits).mean(),
            'logits_neg': torch.mean(logits * (1 - I)),
            'logits_logsumexp': torch.mean((torch.logsumexp(logits + 1e-6, axis=1)**2)),
            'categorical_accuracy': torch.mean((torch.argmax(logits, axis=1) == torch.arange(batch_size, device=device)).float()),
        }
        return contrastive_loss, logs

    def get_temporal_distance(self, obs, next_obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == self.single_feature_dim:
            single_sample = True
            obs = obs.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)
        else:
            single_sample = False

        with torch.no_grad():

            curr_cnn_embs = self._get_cnn_embeddings(obs)
            next_cnn_embs = self._get_cnn_embeddings(next_obs)
            phi_x = self.encoder(curr_cnn_embs)
            phi_y = self.encoder(next_cnn_embs)

            if self.energy_fn == 'l2':
                distance = torch.sqrt(((phi_x - phi_y)**2).sum(dim=-1) + 1e-8)
            elif self.energy_fn == 'cos':
                s_norm = torch.linalg.norm(phi_x, axis=-1, keepdims=True)
                g_norm = torch.linalg.norm(phi_y, axis=-1, keepdims=True)
                phi_x_norm = phi_x / s_norm
                phi_y_norm = phi_y / g_norm
                phi_x_norm = phi_x_norm / self.temperature
                distance = -torch.mean(phi_x_norm * phi_y_norm, dim=-1)
            elif self.energy_fn == 'dot':
                distance = -torch.mean(phi_x * phi_y, dim=-1)
            elif self.energy_fn in ['mrn', 'mrn_pot']:
                distance = mrn_distance(phi_x, phi_y)

        if single_sample:
            return distance[0].item()
        return distance

    def get_intrinsic_rewards(self,
                              curr_obs, next_obs, last_mems, curr_act, curr_dones, obs_history, stats_logger
                              ):
        with torch.no_grad():

            batch_size = curr_obs.size(0)
            curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
            next_cnn_embs = self._get_cnn_embeddings(next_obs)

            int_rews = np.zeros(batch_size, dtype=np.float32)
            for env_id in range(batch_size):

                curr_obs_emb = curr_cnn_embs[env_id].view(1, -1)
                next_obs_emb = next_cnn_embs[env_id].view(1, -1)
                obs_embs = obs_history[env_id]
                new_embs = [curr_obs_emb, next_obs_emb] if obs_embs is None else [obs_embs, next_obs_emb]
                obs_embs = torch.cat(new_embs, dim=0)
                obs_history[env_id] = obs_embs
                phi_x = self.encoder(obs_history[env_id][:-1])
                phi_y = self.encoder(obs_history[env_id][-1].unsqueeze(0))

                if self.energy_fn == 'l2':
                    dists = torch.sqrt(((phi_x[:, None] - phi_y[None, :])**2).sum(dim=-1) + 1e-8)
                elif self.energy_fn == 'cos':
                    x_norm = torch.linalg.norm(phi_x, axis=-1, keepdims=True)
                    y_norm = torch.linalg.norm(phi_y, axis=-1, keepdims=True)
                    phi_x_norm = phi_x / x_norm
                    phi_y_norm = phi_y / y_norm
                    phi_x_norm = phi_x_norm / self.temperature
                    dists = - torch.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
                elif self.energy_fn == 'dot':
                    dists = - torch.einsum("ik,jk->ij", phi_x, phi_y)
                elif 'mrn' in self.energy_fn:
                    dists = mrn_distance(phi_x, phi_y)

                if self.aggregate_fn == 'min':
                    int_rew = dists.min().item()
                    int_rews[env_id] += int_rew
                elif self.aggregate_fn == 'quantile10':
                    int_rews[env_id] += torch.quantile(dists, 0.1).item()
                elif self.aggregate_fn == 'knn':
                    if len(dists) <= self.knn_k:
                        knn_dists = dists
                    else:
                        knn_dists, _ = torch.topk(dists, self.knn_k, largest=False)
                    int_rews[env_id] += knn_dists[-1].item()

        logs = {
            'dists_mean': dists.mean(),
            'dists_min': dists.min(),
            'dists_max': dists.max(),
        }
        stats_logger.add(
            **logs,
        )
        return int_rews, None

    def optimize(self, replay_buffer, logger, step, dqn_gradient_update=1):
        n_update = 0
        if hasattr(replay_buffer, "get"):
            for rollout_data, offpolicy_data in replay_buffer.get(self.batch_size):
                n_update += 1
                if self.offpolicy_data:
                    contrastive_loss, logs = \
                        self.forward(
                            offpolicy_data.observations,
                            offpolicy_data.future_observations,
                        )
                else:
                    contrastive_loss, logs = \
                        self.forward(
                            rollout_data.observations,
                            rollout_data.future_observations,
                        )
        elif hasattr(replay_buffer, "sample_ir"):
            for index in range(dqn_gradient_update):
                raise NotImplementedError

        if n_update == 0:
            print("no data for tdd update")
            return
        loss = contrastive_loss + self.logsumexp_coef * logs['logits_logsumexp']
        for k, v in logs.items():
            logger.log(f'train/td/{k}', v.item(), step)

        self.model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()
