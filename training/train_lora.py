import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from datasets import load_dataset
from mlx.utils import tree_flatten, tree_map
import random


import math


# ============================================================
# 1. LoRA Layer
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, scale=0.1):
        super().__init__()
        self.original = original_layer
        in_dim = original_layer.weight.shape[1]
        out_dim = original_layer.weight.shape[0]
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.scale = scale
        self.lora_b.weight = mx.zeros_like(self.lora_b.weight)

    def __call__(self, x):
        return self.original(x) + self.lora_b(self.lora_a(x)) * self.scale

# ============================================================
# 2. Apply LoRA
# ============================================================
def apply_lora(model, rank=8):
    for layer in model.model.layers:
        layer.self_attn.qkv_proj = LoRALinear(
            layer.self_attn.qkv_proj, rank=rank
        )
    return model

# ============================================================
# 3. Config
# ============================================================
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
LR = 1e-4
BATCH_SIZE = 4
MAX_LENGTH = 256
NUM_STEPS = 300
LORA_RANK = 8

# ============================================================
# 4. Load model
# ============================================================
print("Loading model...")
model, tokenizer = load(MODEL_NAME)

print("Applying LoRA...")
model = apply_lora(model, rank=LORA_RANK)

model.freeze()
for layer in model.model.layers:
    layer.self_attn.qkv_proj.lora_a.unfreeze()
    layer.self_attn.qkv_proj.lora_b.unfreeze()

# ============================================================
# 5. Dataset
# ============================================================
print("Loading dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

def tokenize(example):
    instruction_text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    response_text = example['output']
    instruction_tokens = tokenizer.encode(instruction_text)
    response_tokens = tokenizer.encode(response_text)
    tokens = (instruction_tokens + response_tokens)[:MAX_LENGTH]
    labels = ([-100] * len(instruction_tokens) + response_tokens)[:MAX_LENGTH]
    pad_len = MAX_LENGTH - len(tokens)
    tokens = tokens + [tokenizer.pad_token_id] * pad_len
    labels = labels + [-100] * pad_len
    return tokens, labels

print("Tokenizing...")
all_tokens, all_labels = zip(*[tokenize(ex) for ex in dataset])

def get_batch(step):
    indices = random.sample(range(len(all_tokens)), BATCH_SIZE)
    x = mx.array([all_tokens[i] for i in indices])
    y = mx.array([all_labels[i] for i in indices])
    return x, y

# ============================================================
# 6. Loss and grad clipping and lr scheduler 
# ============================================================
def loss_fn(model, x, labels):
    logits = model(x)
    B, T, V = logits.shape
    shifted_logits = logits[:, :-1, :].reshape(B * (T-1), V)
    shifted_labels = labels[:, 1:].reshape(B * (T-1))
    loss = nn.losses.cross_entropy(shifted_logits, shifted_labels)
    mask = (shifted_labels != -100).astype(mx.float32)
    return (loss * mask).sum() / mask.sum()

def clip_grad_norm(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_norm = mx.sqrt(sum(
        mx.sum(g * g)
        for _, g in leaves
        if isinstance(g, mx.array)
    ))
    scale = mx.minimum(max_norm / (total_norm + 1e-6), 1.0)
    return tree_map(
        lambda g: g * scale if isinstance(g, mx.array) else g,
        grads
    )
def get_lr(step, num_steps, max_lr=1e-4, min_lr=1e-6, warmup_steps=20):
    # Warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / (num_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ============================================================
# 7. Training loop
# ============================================================
print("Starting training...")
optimizer = optim.Adam(learning_rate=LR)
loss_and_grad = nn.value_and_grad(model, loss_fn)

# Record before training
orig_before = mx.array(
    model.model.layers[0].self_attn.qkv_proj.original.weight
)
lora_b_before = mx.array(
    model.model.layers[0].self_attn.qkv_proj.lora_b.weight
)
print("lora_b before (should be ~0):")
print(lora_b_before[:2, :4])

for step in range(NUM_STEPS):
    lr = get_lr(step, NUM_STEPS)
    optimizer.learning_rate = lr
    
    x, y = get_batch(step)
    loss, grads = loss_and_grad(model, x, y)
    grads = clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    if (step + 1) % 10 == 0:
        print(f"Step {step+1}: loss = {loss.item():.4f}, lr = {lr:.2e}")

# Record after training
orig_after = mx.array(
    model.model.layers[0].self_attn.qkv_proj.original.weight
)
lora_b_after = mx.array(
    model.model.layers[0].self_attn.qkv_proj.lora_b.weight
)

print("\nOriginal weight changed (should be False if freeze works):")
print(not mx.allclose(orig_before, orig_after).item())
print("\nlora_b after (should be non-zero):")
print(lora_b_after[:2, :4])

print("Done!")
