# PCM-VLA

This repository contains the code needed to run PCM-VLA on LIBERO-Spatial.
Checkpoints are hosted on HuggingFace and should be downloaded separately.

## Setup

Create an environment and install PyTorch with CUDA support.

```bash
conda create -n pcm-vla python=3.10 -y
conda activate pcm-vla
pip install --pre torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Download the frozen prior models with the HuggingFace CLI.

```bash
huggingface-cli download depth-anything/Depth-Anything-V2-Small
huggingface-cli download depth-anything/Depth-Anything-V2-Base
huggingface-cli download depth-anything/Depth-Anything-V2-Large
huggingface-cli download facebook/VGGT-1B
```

Clone the required third-party code.

```bash
mkdir -p third_party
git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 \
    third_party/Depth-Anything-V2
git clone --depth 1 https://github.com/facebookresearch/vggt \
    third_party/vggt
```

RAFT-Small is loaded from `torchvision`, so it does not need a separate
foundation-model download command.

SmolVLA and LIBERO assets are downloaded lazily by LeRobot when training or
evaluation first runs:

```text
lerobot/smolvla_base
lerobot/smolvla_libero
lerobot/libero
```

## Checkpoints

Download released PCM-VLA checkpoints with the HuggingFace CLI:

```bash
mkdir -p ckpts/baseline ckpts/baseline_spatial ckpts/raft ckpts/dav2 ckpts/vggt
huggingface-cli download Zhengyue2/SmolVLA_base pcm_vla_state.pt \
    --local-dir ckpts/baseline
huggingface-cli download Zhengyue2/SmolVLA_base_spatial pcm_vla_state.pt \
    --local-dir ckpts/baseline_spatial
huggingface-cli download Zhengyue2/pcm-vla-raft pcm_vla_state.pt \
    --local-dir ckpts/raft
huggingface-cli download Zhengyue2/pcm-vla-dav2 pcm_vla_state.pt \
    --local-dir ckpts/dav2
huggingface-cli download Zhengyue2/pcm-vla-vggt pcm_vla_state.pt \
    --local-dir ckpts/vggt
```

Each checkpoint should be placed as:

```text
ckpts/<name>/pcm_vla_state.pt
```

## Training

Train the no-prior baseline with all LIBERO suites:

```bash
python -m src.train \
    --prior none \
    --use_all_libero \
    --steps 25000 \
    --warmup_steps 1000 \
    --out_dir ckpts/baseline
```

Train the main PCM-VLA variants on LIBERO-Spatial:

```bash
python -m src.train --prior raft --steps 15000 --out_dir ckpts/raft
python -m src.train --prior dav2 --steps 15000 --out_dir ckpts/dav2
python -m src.train --prior vggt --steps 15000 --out_dir ckpts/vggt
```

## Evaluation

Run LIBERO-Spatial evaluation for one checkpoint:

```bash
python -m src.eval \
    --ckpt ckpts/baseline/pcm_vla_state.pt \
    --task_ids "[0,1,2,3,4,5,6,7,8,9]" \
    --n_episodes 50 \
    --output_dir eval_outputs/baseline
```

A short smoke test can use fewer episodes:

```bash
python -m src.eval \
    --ckpt ckpts/baseline_spatial/pcm_vla_state.pt \
    --task_ids "[0]" \
    --n_episodes 5 \
    --output_dir eval_outputs/baseline_spatial_smoke
```
