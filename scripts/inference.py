# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

WHISPER_FILES = {
    768: "whisper/small.pt",
    384: "whisper/tiny.pt",
}


def build_pipeline(config, args, dtype):
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim not in WHISPER_FILES:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    whisper_model_path = os.path.join(args.model_root, WHISPER_FILES[config.model.cross_attention_dim])
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained(
        os.path.join(args.model_root, "sd-vae-ft-mse"), torch_dtype=dtype
    )
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    return LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        insightface_root=args.insightface_root,
    ).to("cuda")


def run(pipeline, config, args, dtype, video_path, audio_path, video_out_path):
    kwargs = dict(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=video_out_path,
        num_frames=config.data.num_frames,
        video_fps=config.data.video_fps,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
    )
    if args.model_type == "stage1":
        pipeline.stage1_forward(**kwargs)
    else:
        pipeline.noalign_occ_forward(**kwargs)


def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")
    if not os.path.exists(args.inference_ckpt_path):
        raise RuntimeError(f"Checkpoint '{args.inference_ckpt_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    pipeline = build_pipeline(config, args, dtype)

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    if os.path.isfile(args.video_path):
        run(pipeline, config, args, dtype, args.video_path, args.audio_path, args.video_out_path)
        print("infer done")

    elif os.path.isdir(args.video_path):
        video_names = sorted(os.listdir(args.video_path))
        for video_name in video_names:
            if not video_name.endswith(".mp4"):
                continue
            video_path = os.path.join(args.video_path, video_name)
            if os.path.isdir(args.audio_path):
                audio_path = os.path.join(args.audio_path, video_name.replace(".mp4", ".wav"))
            else:
                audio_path = args.audio_path
            if not os.path.exists(audio_path):
                print(f"{audio_path} does not exist")
                continue
            video_out_path = os.path.join(args.video_out_path, video_name)
            if os.path.exists(video_out_path):
                continue
            os.makedirs(os.path.dirname(video_out_path), exist_ok=True)

            try:
                run(pipeline, config, args, dtype, video_path, audio_path, video_out_path)
                print("infer done")
            except Exception as e:
                print(f"Failed to process {video_name}: {e}")
                continue
    else:
        raise RuntimeError(f"Video path '{args.video_path}' is neither a file nor a directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="stage1", choices=["stage1", "stage3"])
    parser.add_argument("--unet_config_path", type=str, default=None)
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--model_root", type=str, default=os.environ.get("MODEL_ROOT", "models"))
    parser.add_argument(
        "--insightface_root",
        type=str,
        default=os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface")),
    )
    args = parser.parse_args()

    if args.inference_steps is None:
        args.inference_steps = 12 if args.model_type == "stage1" else 1
    if args.unet_config_path is None:
        args.unet_config_path = f"configs/unet/inference_{args.model_type}.yaml"

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
