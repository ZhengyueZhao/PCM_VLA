from __future__ import annotations
import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict, deque
from pathlib import Path
import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_policy_all, close_envs
from src.extractors import make_extractor
from src.model import PCMSmolVLA

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", str(_ROOT / "checkpoints" / "hf_cache"))


class PriorHook:
    def __init__(self, wrapper: PCMSmolVLA, extractor, prior: str, gap: int = 3):
        self.wrapper = wrapper
        self.extract = extractor
        self.prior = prior
        self.gap = gap
        self.buf1: dict[int, deque] = defaultdict(lambda: deque(maxlen=gap + 1))
        self.buf2: dict[int, deque] = defaultdict(lambda: deque(maxlen=gap + 1))

    def reset(self):
        self.buf1.clear()
        self.buf2.clear()

    def observe(self, cam1: torch.Tensor, cam2: torch.Tensor):
        for i in range(cam1.shape[0]):
            self.buf1[i].append(cam1[i].detach())
            self.buf2[i].append(cam2[i].detach())

    @staticmethod
    def _std(x):
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return (x - mu) / sd

    def maybe_set_prior(self, cam1: torch.Tensor, cam2: torch.Tensor):
        B = cam1.shape[0]
        if self.prior in ("dav2", "vggt"):
            if self.prior == "vggt":
                d1, d2 = self.extract(cam1, cam2)
            else:
                d1, d2 = self.extract(cam1), self.extract(cam2)
            raw = torch.stack([self._std(d1), self._std(d2)], dim=1)
            self.wrapper.set_prior(raw)
            return True
        if any(len(self.buf1[i]) <= self.gap for i in range(B)):
            self.wrapper.set_prior(None)
            return False
        prev1 = torch.stack([self.buf1[i][0] for i in range(B)]).to(cam1.device)
        prev2 = torch.stack([self.buf2[i][0] for i in range(B)]).to(cam2.device)
        f1 = self.extract(prev1, cam1)
        f2 = self.extract(prev2, cam2)
        flow = torch.stack([f1, f2], dim=1) / max(1.0, cam1.shape[-1] / 2.0)
        self.wrapper.set_prior(flow)
        return True


def patch_policy(wrapper: PCMSmolVLA, hook: PriorHook | None, use_prior: bool):
    policy = wrapper.policy
    orig_select = policy.select_action
    orig_reset = policy.reset

    def new_select_action(self, batch, *args, **kwargs):
        cam1 = batch.get("observation.images.camera1")
        cam2 = batch.get("observation.images.camera2")
        if cam1 is not None and cam2 is not None and hook is not None:
            hook.observe(cam1, cam2)
            if use_prior:
                hook.maybe_set_prior(cam1, cam2)
            else:
                wrapper.set_prior(None)
        return orig_select(batch, *args, **kwargs)

    def new_reset(self):
        if hook is not None: hook.reset()
        wrapper.set_prior(None)
        return orig_reset()

    policy.select_action = types.MethodType(new_select_action, policy)
    policy.reset = types.MethodType(new_reset, policy)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--ckpt", type=str, required=True)
    args.add_argument("--task_ids", type=str, default="[0,1,2,3,4,5,6,7,8,9]")
    args.add_argument("--n_episodes", type=int, default=50)
    args.add_argument("--batch_size", type=int, default=5)
    args.add_argument("--max_episodes_rendered", type=int, default=0)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--start_seed", type=int, default=1000)
    args = args.parse_args()

    device = torch.device("cuda")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
    af = policy.config.output_features.get("action")
    if af is not None and tuple(af.shape) != (7,):
        af.shape = (7,)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path="lerobot/smolvla_libero",
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": {
                "observation.images.image":  "observation.images.camera1",
                "observation.images.image2": "observation.images.camera2",
            }},
        },
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    use_prior = not cfg["no_prior"]

    if "lm_expert" in ckpt:
        policy.load_state_dict(ckpt["lm_expert"], strict=False)

    wrapper = PCMSmolVLA(
        policy, in_channels=cfg["in_channels"], patch=cfg["patch"],
        alpha_init=float(ckpt["alpha"]) if "alpha" in ckpt else cfg["alpha_init"],
        aux_recon_dim=cfg["aux_recon_dim"],
    ).to(device)
    wrapper.tokenizer.load_state_dict(ckpt["tokenizer"])
    wrapper.alpha.data = ckpt["alpha"].to(device)
    wrapper.film.load_state_dict(ckpt["film"])
    if "aux_head" in ckpt and wrapper.aux_head is not None:
        wrapper.aux_head.load_state_dict(ckpt["aux_head"])
    wrapper.eval()

    hook = None
    if use_prior:
        extractor, _ = make_extractor(
            cfg["prior"], device, dav2_size=cfg.get("dav2_size", "small")
        )
        hook = PriorHook(wrapper, extractor, cfg["prior"], gap=3)
    patch_policy(wrapper, hook, use_prior=use_prior)

    task_ids = json.loads(args.task_ids)
    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=task_ids,
        observation_height=256, observation_width=256,
        init_states=True, max_parallel_tasks=1,
    )
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False,
                    trust_remote_code=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config)

    print(f"Evaluating {Path(args.ckpt).parent.name}: {args.n_episodes} episodes/task, {len(task_ids)} tasks")
    t0 = time.time()
    with torch.no_grad():
        info = eval_policy_all(
            envs=envs, policy=policy,
            env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            max_episodes_rendered=args.max_episodes_rendered,
            videos_dir=out / "videos",
            start_seed=args.start_seed, max_parallel_tasks=1,
        )

    overall = info["overall"]["pc_success"]
    elapsed = time.time() - t0
    per_task = {}
    for t in info["per_task"]:
        s = t["metrics"]["successes"]
        pc = 100 * sum(s) / len(s)
        per_task[t["task_id"]] = pc
    close_envs(envs)

    out_json = {
        "ckpt":      args.ckpt,
        "config":    cfg,
        "use_prior": use_prior,
        "n_episodes_per_task": args.n_episodes,
        "overall":   overall,
        "per_task":  per_task,
        "elapsed_s": elapsed,
    }
    name = Path(args.ckpt).parent.name or "results"
    out_path = out / f"results_{name}.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Overall {overall:.1f}% ({elapsed:.0f}s). Saved {out_path}")


if __name__ == "__main__":
    main()
