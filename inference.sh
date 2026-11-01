#!/bin/bash
# Stage 1 inference example (12 denoising steps).

python -m scripts.inference \
    --model_type stage1 \
    --inference_ckpt_path "${MODEL_ROOT:-models}/stage1.pt" \
    --video_path "examples/demo.mp4" \
    --audio_path "examples/demo.wav" \
    --video_out_path "outputs/stage1.mp4"
