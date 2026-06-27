#!/bin/bash
# Learning Rate Sensitivity Experiments
# Model: microsoft/Phi-3.5-mini-instruct
# Data: tatsu-lab/alpaca, iters=100


python3.11 -m mlx_lm lora \
  --model microsoft/Phi-3.5-mini-instruct \
  --train \
  --data tatsu-lab/alpaca \
  --iters 100 \
  --learning-rate 1e-6 \
  --batch-size 4 \
  --num-layers 8


  python3.11 -m mlx_lm lora \
  --model microsoft/Phi-3.5-mini-instruct \
  --train \
  --data tatsu-lab/alpaca \
  --iters 100 \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --num-layers 8

# Baseline - stable convergence
  python3.11 -m mlx_lm lora \
  --model microsoft/Phi-3.5-mini-instruct \
  --train \
  --data tatsu-lab/alpaca \
  --iters 100 \
  --learning-rate 1e-4 \
  --batch-size 4 \
  --num-layers 8

  # High initial loss and slow convergence

  python3.11 -m mlx_lm lora \
  --model microsoft/Phi-3.5-mini-instruct \
  --train \
  --data tatsu-lab/alpaca \
  --iters 100 \
  --learning-rate 1e-3 \
  --batch-size 4 \
  --num-layers 8

# Too large - oscillation
  python3.11 -m mlx_lm lora \
  --model microsoft/Phi-3.5-mini-instruct \
  --train \
  --data tatsu-lab/alpaca \
  --iters 100 \
  --learning-rate 1e-2 \
  --batch-size 4 \
  --num-layers 8
# Gradient explosion
  python3.11 -m mlx_lm lora \
  --model microsoft/Phi-3.5-mini-instruct \
  --train \
  --data tatsu-lab/alpaca \
  --iters 100 \
  --learning-rate 1e-1 \
  --batch-size 4 \
  --num-layers 8
