#!/bin/bash
# Stage 3 inference example (single denoising step).

python -m scripts.inference \
    --model_type stage3 \
    --inference_ckpt_path "${MODEL_ROOT:-models}/stage3.pt" \
    --video_path "examples/demo.mp4" \
    --audio_path "examples/demo.wav" \
    --video_out_path "outputs/stage3.mp4"
