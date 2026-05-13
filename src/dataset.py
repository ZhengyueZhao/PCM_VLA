from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


class LiberoSpatial(Dataset):
    def __init__(self, episode_ids: list[int], gap: int = 3, chunk_size: int = 50,
                 aux_recon_steps: int = 10, max_samples_per_ep: int | None = None):
        self.ds = LeRobotDataset("lerobot/libero", episodes=episode_ids,
                                 video_backend="pyav")
        self.gap = gap
        self.chunk_size = chunk_size
        self.aux_recon_steps = aux_recon_steps
        self.ep_ids = list(episode_ids)

        ep_col = self.ds.hf_dataset.select_columns(["episode_index"])
        ep_indices = [int(ep_col[i]["episode_index"]) for i in range(len(self.ds))]
        ep_ends_by_start: dict[int, int] = {}
        cur_ep, ep_start = ep_indices[0], 0
        for i in range(1, len(ep_indices)):
            if ep_indices[i] != cur_ep:
                ep_ends_by_start[ep_start] = i
                ep_start, cur_ep = i, ep_indices[i]
        ep_ends_by_start[ep_start] = len(ep_indices)

        self.samples = []
        cur_ep, ep_start = ep_indices[0], 0
        for i in range(len(ep_indices)):
            if ep_indices[i] != cur_ep:
                cur_ep, ep_start = ep_indices[i], i
            t = i - ep_start
            if t >= gap:
                self.samples.append((i, i - gap, ep_ends_by_start[ep_start]))

        if max_samples_per_ep is not None and max_samples_per_ep > 0:
            rng = np.random.default_rng(0)
            n = min(len(self.samples), max_samples_per_ep * len(self.ep_ids))
            idx = rng.choice(len(self.samples), size=n, replace=False)
            self.samples = [self.samples[j] for j in idx]
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j, ep_end = self.samples[idx]
        cur = self.ds[i]
        prev = self.ds[j]
        hf = self.ds.hf_dataset

        k_list = [min(i + k, ep_end - 1) for k in range(self.chunk_size)]
        acts_raw = hf.select(k_list)["action"]
        actions = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in acts_raw])
        is_pad = torch.tensor([i + k >= ep_end for k in range(self.chunk_size)], dtype=torch.bool)

        K = self.aux_recon_steps
        base = torch.as_tensor(hf[i]["observation.state"], dtype=torch.float32)[:6]
        kr = [min(i + k, ep_end - 1) for k in range(1, K + 1)]
        fut_raw = hf.select(kr)["observation.state"]
        fut = torch.stack([torch.as_tensor(x, dtype=torch.float32)[:6] for x in fut_raw]) - base[None]

        return {
            "image_t":  cur["observation.images.image"],
            "image2_t": cur["observation.images.image2"],
            "image_p":  prev["observation.images.image"],
            "image2_p": prev["observation.images.image2"],
            "state":    cur["observation.state"],
            "action_chunk":   actions,
            "actions_is_pad": is_pad,
            "eef_future":     fut,
            "task":  cur["task"],
            "index": torch.tensor([i], dtype=torch.long),
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        if k == "task":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def load_spatial_episode_ids() -> list[int]:
    p = _root() / "data" / "libero_spatial_episode_ids.json"
    return json.load(open(p))["spatial_ep_indices"]


def build_policy_batch(raw_batch, preprocessor, device):
    img_t  = raw_batch["image_t"].to(device).float() / 255.0
    img2_t = raw_batch["image2_t"].to(device).float() / 255.0
    img_p  = raw_batch["image_p"].to(device).float() / 255.0
    img2_p = raw_batch["image2_p"].to(device).float() / 255.0

    raw_dict = {
        "observation.images.image":  img_t,
        "observation.images.image2": img2_t,
        "observation.state":         raw_batch["state"].to(device).float(),
        "action":                    raw_batch["action_chunk"].to(device).float(),
        "actions_id_pad":            raw_batch["actions_is_pad"].to(device),
        "task":                      raw_batch["task"],
    }
    batch = preprocessor(raw_dict)
    return batch, (img_t, img2_t, img_p, img2_p)
