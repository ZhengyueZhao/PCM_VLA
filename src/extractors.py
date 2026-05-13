from __future__ import annotations
import importlib
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


class RAFTExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=self.weights, progress=False)
        self.transform = self.weights.transforms()
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        H, W = prev.shape[-2:]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            prev = F.pad(prev, (0, pad_w, 0, pad_h))
            curr = F.pad(curr, (0, pad_w, 0, pad_h))
        a, b = self.transform(prev, curr)
        flow = self.model(a, b)[-1]
        if pad_h or pad_w:
            flow = flow[..., :H, :W]
        return flow


DAV2_CONFIGS = {
    "small": {
        "repo": "Depth-Anything-V2-Small",
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "weight": "depth_anything_v2_vits.pth",
    },
    "base": {
        "repo": "Depth-Anything-V2-Base",
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "weight": "depth_anything_v2_vitb.pth",
    },
    "large": {
        "repo": "Depth-Anything-V2-Large",
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "weight": "depth_anything_v2_vitl.pth",
    },
}


class DAV2Extractor(nn.Module):
    def __init__(self, size: str = "small"):
        super().__init__()
        sys.path.insert(0, str(_root() / "third_party" / "Depth-Anything-V2"))
        DepthAnythingV2 = importlib.import_module("depth_anything_v2.dpt").DepthAnythingV2
        if size not in DAV2_CONFIGS:
            raise ValueError(f"Unknown DAV2 size={size}; choose one of {sorted(DAV2_CONFIGS)}")
        cfg = DAV2_CONFIGS[size]
        self.size = size
        self.model = DepthAnythingV2(
            encoder=cfg["encoder"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
        )
        ckpt_root = (
            _root() / "checkpoints" / "hf_cache"
            / f"models--depth-anything--{cfg['repo']}"
            / "snapshots"
        )
        matches = sorted(ckpt_root.glob(f"*/{cfg['weight']}"))
        if not matches:
            raise FileNotFoundError(
                f"DAV2-{size} ckpt not found under {ckpt_root}. "
                "Download the prior models following README.md first."
            )
        ckpt_path = matches[-1]
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        cv2 = importlib.import_module("cv2")
        B, _, H, _ = img.shape
        img_np = (img.detach().cpu().numpy() * 255).astype("uint8").transpose(0, 2, 3, 1)
        out = []
        for i in range(B):
            bgr = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR)
            d = self.model.infer_image(bgr, input_size=H)
            out.append(torch.from_numpy(d).float())
        return torch.stack(out).unsqueeze(1).to(img.device)


class VGGTExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        VGGT = importlib.import_module("vggt.models.vggt").VGGT
        snap = (_root() / "checkpoints" / "hf_cache" / "models--facebook--VGGT-1B"
                / "snapshots" / "860abec7937da0a4c03c41d3c269c366e82abdf9")
        if not snap.exists():
            raise FileNotFoundError(
                f"VGGT snapshot not found at {snap}. Download the prior models following README.md first."
            )
        self.model = VGGT.from_pretrained(str(snap))
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, agent: torch.Tensor, wrist: torch.Tensor):
        B, _, H, W = agent.shape
        H14, W14 = (H // 14) * 14, (W // 14) * 14
        a = F.interpolate(agent, size=(H14, W14), mode="bilinear", align_corners=False)
        w = F.interpolate(wrist, size=(H14, W14), mode="bilinear", align_corners=False)
        x = torch.stack([a, w], dim=1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = self.model(x)
        d = preds["depth"].float()
        d_a = F.interpolate(d[:, 0, ..., 0].unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
        d_w = F.interpolate(d[:, 1, ..., 0].unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
        return d_a, d_w


PRIOR_TO_CHANNELS = {"raft": 2, "dav2": 1, "vggt": 1}


def make_extractor(prior: str, device: torch.device, dav2_size: str = "small"):
    if prior == "raft":
        return RAFTExtractor().to(device), 2
    if prior == "dav2":
        return DAV2Extractor(size=dav2_size).to(device), 1
    if prior == "vggt":
        return VGGTExtractor().to(device), 1
    raise ValueError(f"Unknown prior={prior}")


def compute_prior(extractor, prior: str, img_t: torch.Tensor, img2_t: torch.Tensor,
                  img_p: torch.Tensor = None, img2_p: torch.Tensor = None) -> torch.Tensor:
    H = img_t.shape[-1]
    if prior == "raft":
        f_a = extractor(img_p, img_t)
        f_w = extractor(img2_p, img2_t)
        return torch.stack([f_a, f_w], dim=1) / max(1.0, H / 2.0)
    if prior == "dav2":
        d_a, d_w = extractor(img_t), extractor(img2_t)
    else:
        d_a, d_w = extractor(img_t, img2_t)

    def _std(x):
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return (x - mu) / sd

    return torch.stack([_std(d_a), _std(d_w)], dim=1)
