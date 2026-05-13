from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class PriorTokenizer(nn.Module):
    def __init__(self, d_vlm: int = 960, patch: int = 8, n_cams: int = 2,
                 in_channels: int = 2):
        super().__init__()
        self.patch = patch
        self.n_cams = n_cams
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels, d_vlm)
        self.pos_embed = nn.Parameter(torch.zeros(patch * patch, d_vlm))
        self.cam_embed = nn.Embedding(n_cams, d_vlm)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.cam_embed.weight)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        B, C, Cin, H, W = raw.shape
        assert C == self.n_cams and Cin == self.in_channels
        x = raw.reshape(B * C, Cin, H, W)
        x = F.adaptive_avg_pool2d(x, self.patch)
        x = x.reshape(B, C, Cin, self.patch * self.patch).permute(0, 1, 3, 2)
        tokens = self.proj(x) + self.pos_embed[None, None]
        cam_ids = torch.arange(self.n_cams, device=tokens.device)
        tokens = tokens + self.cam_embed(cam_ids)[None, :, None, :]
        return tokens.reshape(B, C * self.patch * self.patch, -1)


class _FilmWrappedStateProj(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self._outer_ref: list = []

    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features

    def forward(self, state):
        x = self.base(state)
        outer = self._outer_ref[0]
        if outer._cur_prior_tokens is None:
            return x
        pooled = outer._cur_prior_tokens.mean(dim=1)
        gb = outer.film(pooled)
        gamma, beta = gb.chunk(2, dim=-1)
        if x.dtype != gamma.dtype:
            gamma, beta = gamma.to(x.dtype), beta.to(x.dtype)
        return x * (1 + outer.alpha * gamma) + outer.alpha * beta


class PCMSmolVLA(nn.Module):
    def __init__(
        self,
        policy: SmolVLAPolicy,
        in_channels: int,
        patch: int = 8,
        alpha_init: float = 1.0,
        aux_recon_dim: int = 6,
        aux_recon_steps: int = 10,
    ):
        super().__init__()
        self.policy = policy
        d_vlm = policy.model.vlm_with_expert.vlm.config.text_config.hidden_size
        self.d_vlm = d_vlm

        self.tokenizer = PriorTokenizer(d_vlm=d_vlm, patch=patch, in_channels=in_channels)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self.film = nn.Sequential(
            nn.Linear(d_vlm, 256), nn.GELU(),
            nn.Linear(256, 2 * d_vlm),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        self.aux_recon_dim = aux_recon_dim
        self.aux_recon_steps = aux_recon_steps
        if aux_recon_dim > 0:
            self.aux_head = nn.Sequential(
                nn.Linear(d_vlm, 256), nn.GELU(),
                nn.Linear(256, aux_recon_dim * aux_recon_steps),
            )
        else:
            self.aux_head = None

        for p in self.policy.parameters():
            p.requires_grad = False

        wrapped = _FilmWrappedStateProj(policy.model.state_proj)
        wrapped._outer_ref.append(self)
        policy.model.state_proj = wrapped

        self._cur_prior_tokens: Optional[torch.Tensor] = None

    def set_prior(self, raw: Optional[torch.Tensor]):
        self._cur_prior_tokens = self.tokenizer(raw) if raw is not None else None

    def forward(self, batch: dict, prior_raw: Optional[torch.Tensor] = None,
                eef_future: Optional[torch.Tensor] = None, aux_weight: float = 0.1):
        self.set_prior(prior_raw)
        main_loss, info = self.policy.forward(batch)
        aux_loss = torch.tensor(0.0, device=main_loss.device)
        if self.aux_head is not None and eef_future is not None and prior_raw is not None:
            pooled = self._cur_prior_tokens.mean(dim=1)
            pred = self.aux_head(pooled).reshape(-1, self.aux_recon_steps, self.aux_recon_dim)
            aux_loss = F.mse_loss(pred, eef_future.to(pred.dtype))
            info["aux_loss"] = float(aux_loss.detach().cpu())
        info["main_loss"] = float(main_loss.detach().cpu())
        self.set_prior(None)
        return main_loss + aux_weight * aux_loss, info

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
