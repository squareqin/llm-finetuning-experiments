# Custom LoRA Training Loop - Implementation Notes

## Why implement from scratch

The goal was to understand what happens inside mlx_lm's abstraction.
Key question: what does the library handle that we need to implement ourselves?

## Findings

### 1. Full fine-tuning causes OOM
Running full fine-tuning on Phi-3.5-mini (3.8B) consumed 60GB+ memory
on a 64GB machine, causing system instability.

Root cause: full fine-tuning stores gradients for all 3.8B parameters.
With Adam optimizer, memory requirement is approximately 3x model size
(weights + gradients + optimizer state).

Solution: LoRA reduces trainable parameters to 0.08% by adding low-rank
matrices A and B to attention layers, freezing all original weights.

### 2. Prompt masking is critical for SFT quality
Without masking, loss is computed over all tokens including instruction.
This causes two problems:
- Model wastes capacity learning to predict instruction format tokens
- Instruction loss dominates the gradient signal, masking response quality

Implementation: set labels to -100 for instruction and padding tokens.
MLX does not support ignore_index in cross_entropy, so manual masking
was implemented instead.

### 3. Training behavior comparison

| Version | Initial Loss | Final Loss | Notes |
|---------|-------------|------------|-------|
| mlx_lm CLI | 1.33 | 0.96 | Has warmup + gradient clipping |
| Custom (no masking) | 14 | 0.18 | Loss dominated by instruction tokens |
| Custom (with masking) | 17 | 0.67 | Pure response loss, no warmup |

### 4. Why custom version has higher initial loss
mlx_lm CLI uses warmup: learning rate starts near 0 and gradually increases.
This means the model is barely updated in early steps, so loss at step 10
reflects near-original model performance.

Custom version uses fixed lr=1e-4 from step 1, causing more aggressive
early updates. Initial loss reflects post-update state, which is higher.

### 5. Model architecture discovery
Phi-3.5-mini combines Q, K, V into a single qkv_proj (shape: 9216x3072),
unlike LLaMA which has separate q_proj, k_proj, v_proj.
This required inspecting the model structure before applying LoRA.

## What to add next
- Gradient clipping
- Learning rate warmup + cosine decay
- Validation loss tracking
- Save and load LoRA weights
