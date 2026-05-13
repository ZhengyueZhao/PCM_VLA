from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from src.dataset import LiberoSpatial, collate, load_spatial_episode_ids, build_policy_batch
from src.extractors import make_extractor, compute_prior, PRIOR_TO_CHANNELS
from src.model import PCMSmolVLA

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_HOME", str(_ROOT / "checkpoints" / "hf_cache"))


def make_lr_lambda(warmup: int, total: int, floor_ratio: float):
    def fn(step: int):
        if step < warmup:
            return step / max(1, warmup)
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor_ratio + 0.5 * (1 - floor_ratio) * (1 + math.cos(math.pi * t))
    return fn


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--prior", type=str, required=True,
                    choices=["none", "raft", "dav2", "vggt"])
    args.add_argument("--out_dir", type=str, required=True)

    args.add_argument("--n_episodes", type=int, default=432)
    args.add_argument("--use_all_libero", action="store_true",
                    help="Use all 1693 lerobot/libero episodes (4 suites). "
                         "Recipe for the released `baseline` ckpt; prior variants use libero_spatial only.")
    args.add_argument("--max_per_ep", type=int, default=0)
    args.add_argument("--gap", type=int, default=3)

    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--steps", type=int, default=15000)
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--warmup_steps", type=int, default=500)
    args.add_argument("--lr_min_ratio", type=float, default=0.1)

    args.add_argument("--alpha_init", type=float, default=1.0)
    args.add_argument("--patch", type=int, default=8)
    args.add_argument("--aux_recon_dim", type=int, default=6)
    args.add_argument("--aux_weight", type=float, default=0.1)
    args.add_argument("--dav2_size", type=str, default="small",
                    choices=["small", "base", "large"],
                    help="Depth-Anything-V2 model scale; used only with --prior dav2.")

    args.add_argument("--save_every", type=int, default=3000)
    args.add_argument("--log_every", type=int, default=200)
    args.add_argument("--device", type=str, default="cuda")
    args = args.parse_args()

    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    no_prior = (args.prior == "none")
    in_channels = 1 if no_prior else PRIOR_TO_CHANNELS[args.prior]

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
    af = policy.config.output_features.get("action")
    if af is not None and tuple(af.shape) != (7,):
        af.shape = (7,)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path="lerobot/smolvla_libero",
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    model = PCMSmolVLA(
        policy, in_channels=in_channels, patch=args.patch,
        alpha_init=args.alpha_init,
        aux_recon_dim=args.aux_recon_dim,
    ).to(device)

    for n, p in policy.named_parameters():
        if "lm_expert" in n:
            p.requires_grad = True

    if no_prior:
        for p in model.tokenizer.parameters(): p.requires_grad = False
        for p in model.film.parameters():      p.requires_grad = False
        if model.aux_head is not None:
            for p in model.aux_head.parameters(): p.requires_grad = False
        model.alpha.requires_grad = False
        model.alpha.data.fill_(0.0)

    extractor = None
    if not no_prior:
        extractor, _ = make_extractor(args.prior, device, dav2_size=args.dav2_size)

    if args.use_all_libero:
        spatial_ids = list(range(1693))
    else:
        spatial_ids = load_spatial_episode_ids()[: args.n_episodes]
    ds = LiberoSpatial(spatial_ids, gap=args.gap, chunk_size=50,
                       aux_recon_steps=10,
                       max_samples_per_ep=(args.max_per_ep or None))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(args.warmup_steps, args.steps, args.lr_min_ratio)
    )
    print(f"Training prior={args.prior}, steps={args.steps}, trainable={model.n_trainable()/1e6:.1f}M")

    step = 0
    log_rows = []
    t0 = time.time()
    model.train()
    data_iter = iter(dl)
    while step < args.steps:
        try:
            raw = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            raw = next(data_iter)

        batch, (img_t, img2_t, img_p, img2_p) = build_policy_batch(raw, preprocessor, device)
        prior_raw = None
        if not no_prior:
            prior_raw = compute_prior(extractor, args.prior, img_t, img2_t, img_p, img2_p)
        eef_future = raw["eef_future"].to(device).float() if args.aux_recon_dim > 0 else None

        loss, info = model(batch, prior_raw=prior_raw, eef_future=eef_future,
                           aux_weight=args.aux_weight)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 10.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            extra = ""
            if "main_loss" in info: extra += f"  main={info['main_loss']:.4f}"
            if "aux_loss" in info:  extra += f"  aux={info['aux_loss']:.4f}"
            print(f"step {step:5d}: loss={loss.item():.4f}{extra}  "
                  f"alpha={float(model.alpha):+.4f}  "
                  f"time={time.time()-t0:.0f}s  mem={mem:.1f}GB")
            log_rows.append({
                "step": step, "loss": float(loss.item()),
                "main_loss": info.get("main_loss"),
                "aux_loss": info.get("aux_loss"),
                "alpha": float(model.alpha),
                "elapsed_s": time.time() - t0,
            })
        step += 1

    ckpt = {
        "config": {
            "prior":         args.prior,
            "no_prior":      no_prior,
            "patch":         args.patch,
            "in_channels":   in_channels,
            "alpha_init":    args.alpha_init,
            "aux_recon_dim": args.aux_recon_dim,
            "aux_weight":    args.aux_weight,
            "dav2_size":     args.dav2_size,
        },
        "tokenizer": model.tokenizer.state_dict(),
        "alpha":     model.alpha.detach().cpu(),
        "film":      model.film.state_dict(),
        "lm_expert": {k: v.detach().cpu()
                      for k, v in model.policy.state_dict().items() if "lm_expert" in k},
    }
    if model.aux_head is not None:
        ckpt["aux_head"] = model.aux_head.state_dict()

    torch.save(ckpt, out / "pcm_vla_state.pt")
    with open(out / "train_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)
    size_mb = (out / "pcm_vla_state.pt").stat().st_size / 1e6
    print(f"Saved {out}/pcm_vla_state.pt ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
