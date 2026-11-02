<h1 align="center">ComplexSync: High-Fidelity and Real-Time Lip Sync in Complex Scenarios</h1>

<div align='center'>
    <a href='https://github.com/caijiaran1993' target='_blank'>Jiaran Cai</a><sup>*</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://github.com/XPMarcus' target='_blank'>Xingpei Ma</a><sup>*</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://github.com/huangshenneng' target='_blank'>Shenneng Huang</a><sup>*</sup>
</div>

<div align='center'>
    <small><sup>*</sup> Equal contribution</small>
</div>

<div align='center'>
Guangzhou Quwan Network Technology
</div>
<br>

<div align="center">
  <a href='https://arxiv.org/abs/2609.29225'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  <a href='https://github.com/Playmate111/ComplexSync'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
</div>

## 📰 News

- **`2026/11/02`**: 📦 The inference code of ComplexSync is now open-sourced in this repository. See [Usage](#usage) to run the released checkpoints.
- **`2026/11/02`**: 📊 The **Complex-Val** test set is now open-sourced on ModelScope: [ComplexSync-Val](https://modelscope.cn/datasets/alkdjfklaj/ComplexSync-Val).
- **`2026/10/15`**: 🎉🎉🎉 Our [paper](https://arxiv.org/abs/2609.29225) has been accepted to [WACV 2027](https://wacv.thecvf.com/). We plan to release the inference code, model weights, and the **Complex-Val** test set in the coming weeks. Stay tuned and thank you for your patience!
- **`2026/09/24`**: 🚀🚀🚀 Our [paper](https://arxiv.org/abs/2609.29225) is in public on arxiv.

## Overview

This repository is the **inference-only** release of ComplexSync, a talking-head video
generation model that synchronises lip motion to a driving audio track while preserving the
appearance of the source video.

Training code, training data pre-processing, the evaluation suite and the stage-2
distillation models are not included.

## Released models

Two checkpoints are released:

| Model | Sampling steps | Config | Weight file |
| ----- | -------------- | ------ | ----------- |
| Stage 1 | 12 | `configs/unet/inference_stage1.yaml` | `stage1.pt` |
| Stage 3 | 1 | `configs/unet/inference_stage3.yaml` | `stage3.pt` |

* **Stage 1** is the multi-step model. It runs 12 DDIM steps with three-way classifier-free
  guidance (audio, motion and unconditional branches) over sliding windows.
* **Stage 3** is the one-step model distilled on top of stage 1, additionally fine-tuned with
  occlusion restoration. It runs a single denoising step, which makes it roughly an order of
  magnitude faster than stage 1.

The stage-2 distillation checkpoint is not released.

Both checkpoints are hosted on ModelScope:
[alkdjfklaj/ComplexSync](https://modelscope.cn/models/alkdjfklaj/ComplexSync).

## Complex-Val

The **Complex-Val** test set is available on ModelScope:
[ComplexSync-Val](https://modelscope.cn/datasets/alkdjfklaj/ComplexSync-Val).

## Installation

```bash
conda create -n complexsync python=3.10 -y
conda activate complexsync
pip install -r requirements.txt
```

`ffmpeg` must be available on `PATH`:

```bash
conda install -c conda-forge ffmpeg
```

## Model assets

The repository does not ship weights. Download `stage1.pt` and `stage3.pt` from
[ModelScope](https://modelscope.cn/models/alkdjfklaj/ComplexSync) and place all assets under
a single model root (default: `models/`, overridable with `--model_root` or the
`MODEL_ROOT` environment variable):

```
models/
├── stage1.pt
├── stage3.pt
├── sd-vae-ft-mse/          # Stable Diffusion VAE (stabilityai/sd-vae-ft-mse)
└── whisper/
    ├── tiny.pt             # used when cross_attention_dim == 384
    └── small.pt            # used when cross_attention_dim == 768
```

Fetch the released checkpoints with the ModelScope CLI:

```bash
pip install modelscope
modelscope download --model alkdjfklaj/ComplexSync --local_dir models
```

or clone the model repository directly:

```bash
git lfs install
git clone https://www.modelscope.cn/models/alkdjfklaj/ComplexSync.git models
```

The VAE (`stabilityai/sd-vae-ft-mse`) and the Whisper encoder are **not** part of that model
repository; download them separately and place them in `models/sd-vae-ft-mse/` and
`models/whisper/` as shown above.

Both released configs use `cross_attention_dim: 384`, i.e. the Whisper `tiny` encoder.

Face detection / alignment uses InsightFace `buffalo_l`. It is loaded from
`~/.insightface` by default and is downloaded automatically on first use; override the
location with `--insightface_root` or the `INSIGHTFACE_ROOT` environment variable.

## Usage

> The commands below reference `examples/demo.mp4` and `examples/demo.wav`. Demo inputs are
> not included in this repository; point the flags at your own video and audio files.

Stage 1 (12 steps):

```bash
python -m scripts.inference \
    --model_type stage1 \
    --inference_ckpt_path models/stage1.pt \
    --video_path path/to/video.mp4 \
    --audio_path path/to/audio.wav \
    --video_out_path path/to/output.mp4
```

Stage 3 (1 step):

```bash
python -m scripts.inference \
    --model_type stage3 \
    --inference_ckpt_path models/stage3.pt \
    --video_path path/to/video.mp4 \
    --audio_path path/to/audio.wav \
    --video_out_path path/to/output.mp4
```

`--video_path` may also point to a directory of `.mp4` files, in which case `--audio_path`
is either a single audio file (reused for every video) or a directory containing files with
matching names, and `--video_out_path` is treated as an output directory.

Other options:

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--inference_steps` | 12 / 1 | Denoising steps; defaults to 12 for stage 1 and 1 for stage 3 |
| `--guidance_scale` | 3.0 | Audio guidance scale (stage 1) |
| `--seed` | 1247 | Random seed, `-1` for a random one |
| `--unet_config_path` | derived from `--model_type` | Override the UNet config |
| `--model_root` | `models` / `$MODEL_ROOT` | Root directory of the model assets |
| `--insightface_root` | `~/.insightface` / `$INSIGHTFACE_ROOT` | Root directory of the InsightFace model pack |

## How inference works

1. The driving audio is encoded with Whisper and split into per-video-frame audio features
   (50 Hz features aligned to the 25 FPS video).
2. Frames are cropped and aligned around the detected face; the mouth region is masked and
   the masked face is encoded into the VAE latent space.
3. The UNet predicts the clean latent for a sliding window of 16 frames (stage 1 averages
   overlapping predictions), conditioned on the masked frame latent, the reference frame
   latent and the audio features.
4. Latents are decoded, blended with the original pixels outside the mouth region, un-warped
   back to the original frame and written out together with the original audio.

## Notes

* The reference (appearance) latent is the aligned source frame at the same temporal index,
  i.e. a per-frame reference rather than a single static reference image.
* Stage 1 uses a 3-frame overlap between consecutive windows; stage 3 uses a stride of
  `num_frames - 1` and averages overlapping predictions.
* Videos are processed at 25 FPS. Use `ffmpeg` to resample beforehand if needed.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for third-party components and their licenses.
