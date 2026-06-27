# Custom LoRA Training Loop - Implementation Notes

## Why implement from scratch

The goal was to understand what happens inside mlx_lm's abstraction.
Key question: what does the library handle that we need to implement ourselves?

## Implementation Journey

### Step 1: Full fine-tuning → OOM
First attempt was a full fine-tuning loop without LoRA.
Result: system ran out of memory (60GB+ on a 64GB machine).

Root cause: full fine-tuning stores gradients for all 3.8B parameters.
With Adam optimizer, memory requirement is approximately 3-4x model size
(weights + gradients + optimizer state).

Solution: LoRA reduces trainable parameters to ~3.1M (0.08% of 3.8B)
by adding low-rank matrices A and B to attention layers,
freezing all original weights.

### Step 2: LoRA implementation
Implemented LoRALinear from scratch:

```python
def __call__(self, x):
    return self.original(x) + self.lora_b(self.lora_a(x)) * self.scale
```

Key design decisions:
- B initialized to zeros: ensures LoRA output is zero at training start,
  so model behavior is unchanged before training begins
- A initialized randomly: provides effective random projection from step 1
- Applied to qkv_proj only (Phi3 combines Q/K/V into one matrix,
  unlike LLaMA which has separate q_proj, k_proj, v_proj)

### Step 3: Discovering Phi3 architecture difference
Attempted to apply LoRA to q_proj and v_proj (standard LLaMA approach).
Got AttributeError: 'Attention' object has no attribute 'q_proj'.

Inspected model structure:
```python
print(model.model.layers[0].self_attn.keys())
# dict_keys(['qkv_proj', 'o_proj', 'rope'])
```

Phi3 merges Q, K, V into single qkv_proj (shape: 9216x3072).
Applied LoRA to qkv_proj instead.

### Step 4: Prompt masking
Without masking, loss is computed over all tokens including instruction.
Problems:
- Model wastes capacity learning instruction format tokens
- Instruction loss dominates gradient signal

Implementation: set labels to -100 for instruction and padding tokens.
MLX does not support ignore_index in cross_entropy, 
so implemented manual masking:

```python
mask = (shifted_labels != -100).astype(mx.float32)
loss = (loss * mask).sum() / mask.sum()
```

### Step 5: Verifying freeze works correctly
Concern: MLX's trainable_parameters() API returned unreliable counts.
Verified by comparing weights before and after training:

lora_b before training:  [[0, 0, 0, 0], [0, 0, 0, 0]]   ✅ zeros

original weight changed:  False                            ✅ freeze works

lora_b after training:   [[-0.010, -0.004, 0.008, ...]]  ✅ LoRA updated
Conclusion: freeze is working correctly. Only LoRA parameters are updated.

### Step 6: Gradient clipping
Added manual gradient clipping (MLX does not have built-in clip_grad_norm):

```python
def clip_grad_norm(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_norm = mx.sqrt(sum(
        mx.sum(g * g) for _, g in leaves if isinstance(g, mx.array)
    ))
    scale = mx.minimum(max_norm / (total_norm + 1e-6), 1.0)
    return tree_map(
        lambda g: g * scale if isinstance(g, mx.array) else g, grads
    )
```

### Step 7: Shuffle vs sequential batching
Sequential batching caused oscillation: adjacent batches had similar data,
leading to unstable gradient directions.

Random sampling (inspired by Karpathy's nanoGPT get_batch):
```python
indices = random.sample(range(len(all_tokens)), BATCH_SIZE)
```
Result: more stable loss curve.

## Training Results Comparison

| Version | Initial Loss | Final Loss | Notes |
|---------|-------------|------------|-------|
| mlx_lm CLI | 1.33 | 0.96 | warmup + clipping built-in |
| Custom, no masking | 14 | 0.18 | instruction loss dominates |
| Custom, with masking | 17 | 0.67 | pure response loss |
| Custom, masking + shuffle | 2.25 | 1.17 | verified freeze correct |
| Custom, masking + shuffle+  increase to 300 steps| 2.25 | 1.4 | late trend improving |
| Custom, masking + shuffle + + increase to 300 stepslr scheduler || 2.25 | 1.4 |late trend improving  |

Note: higher initial loss in custom versions is expected —
mlx_lm CLI uses lr warmup which keeps early loss artificially low.

### Step 8: Learning Rate Scheduler

Added cosine decay with warmup to address late-stage oscillation.

```python
def get_lr(step, num_steps, max_lr=1e-4, min_lr=1e-6, warmup_steps=20):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (num_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Warmup phase (step 1-20): lr increases from 0 → 1e-4
Decay phase (step 21-300): lr decreases from 1e-4 → 1e-6

Result:
- Late-stage trend improved: without scheduler loss was rising at step 240 (1.47),
  with scheduler loss was falling at step 240 (1.13)
- Oscillation persisted throughout training

Root cause of persistent oscillation: batch size=4 is too small.
Each batch contains only 4 random samples, causing high variance
in gradient direction regardless of lr schedule.

Real fix: gradient accumulation
- Accumulate gradients over N steps before updating
- Effective batch size = batch_size × accumulation_steps
- Memory cost stays the same as batch_size=4
- Gradient quality equivalent to batch_size=4×N

Key insight: warmup + cosine decay improves convergence trend,
but cannot fix high-variance gradients from small batch size.
These are two separate problems requiring separate solutions.

### Step 9: What gradient accumulation enables

With gradient accumulation, you can decouple memory from batch quality:

Current constraint:
batch_size=4, max_length=256 → memory ~15GB

With gradient accumulation (accumulation_steps=8):
batch_size=1, max_length=512, accumulation_steps=16
→ effective batch_size=16, longer sequences, same memory

## Key Findings

**1. MLX API differences from PyTorch**
- No ignore_index in cross_entropy → manual masking required
- No built-in clip_grad_norm → manual implementation required  
- trainable_parameters() returns unreliable counts in this version
- Lazy execution: mx.eval() must be called to trigger actual computation

**2. LoRA math**
output = W·x + (B·A·x) * scale

= original(x) + lora_b(lora_a(x)) * scale
Trainable params: 32 layers × (3072×8 + 8×9216) = 3,145,728

Total params: 3,821,080,000

Ratio: 0.082%
**3. Why prompt masking matters**
Without masking: model learns to predict instruction format tokens.
These are known inputs at inference time — wasted model capacity.
With masking: every gradient update improves response quality only.

**4. Freeze verification**
Do not rely on trainable_parameters() count alone.
Always verify by comparing weights before and after training.

## What to add next
- Learning rate warmup + cosine decay scheduler
- Validation loss tracking
- Save and load LoRA adapter weights
- Longer training run (1000+ steps) to see convergence

