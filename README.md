# LLaMA LoRA Fine-tuning: Learning Rate Sensitivity Experiments

## Overview
Fine-tuned Phi-3.5-mini-instruct on Alpaca dataset using LoRA, 
systematically exploring the effect of learning rate on training stability.

## Setup
- Model: microsoft/Phi-3.5-mini-instruct (3.8B)
- Method: LoRA (num-layers=8, trainable params: 0.082%)
- Data: tatsu-lab/alpaca
- Hardware: Apple M-series, 64GB unified memory
- Framework: MLX

## Experiments

| Learning Rate | Final Loss | Behavior |
lr=1e-1：NaN      💥 gradient explosion
lr=1e-2：21.8     💥 oscillation  and diverging 
lr=1e-3：7.25     ⚠️  inital loss very high
lr=1e-4：0.96     ✅ ✅optimized
lr=1e-5：0.994    ✅ less optimized
lr=1e-6：1.48     ⚠️ slower convergence

## Key Observations

**1. Gradient Explosion (lr=1e-1)**
Loss started as nan, meaning the update was too big  at the beginning stage and caused gradient explosion and overflow 

**2. Oscillation and Divergence (lr=1e-2)**
It did not cause loss explosion during the 100 iterations, but it caused oscillations and later loss increase, meaning the update was still too big and the result was diverging instead of converging.

**3. High Initial Loss (lr=1e-3)**
The initial loss is too high even though loss has been decreasing throughout 
the 100 iterations, ending higher than lr=1e-4. This suggests that even 
lr=1e-3 was too aggressive — the first few updates pushed the LoRA weights 
far from their initialization, and 100 iterations were insufficient to recover. 
This highlights why LoRA fine-tuning is sensitive to learning rate even with 
only 0.082% of parameters being trained.

**4. Stable Convergence (lr=1e-4)**
Initial loss was low and kept decreasing steadily throughout training, 
demonstrating that lr=1e-4 sits in the stable optimization region for 
this model and dataset combination.

**5. Too Small Learning Rate (lr=1e-5, lr=1e-6)**
Both converged in the right direction but ended with higher final loss 
than lr=1e-4 within 100 iterations. The updates were too small to make 
meaningful progress in limited steps. This is underfitting from the 
optimization side — not because the model lacks capacity, but because 
the optimizer isn't moving fast enough.

## Conclusion
The stable optimization zone for this setup is lr=1e-4 to 1e-5. 
This experiment validates why lr=1e-4 to 3e-4 is the standard recommended 
range for LoRA fine-tuning in most production settings — it balances 
convergence speed with training stability.
