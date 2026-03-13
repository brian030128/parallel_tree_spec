# HQQ Quantization in Our Project

## Overview

HQQ (Half-Quadratic Quantization) compresses model weights to lower bit-widths (2, 3, 4, 8 bits) while preserving model quality. In our project, we quantize the **draft model** to make beam search faster and more memory-efficient, then measure how quantization affects token acceptance rates.

## How HQQ Works

HQQ replaces `nn.Linear` layers with `HQQLinear` layers that store:
- **Quantized weights** (int2/int3/int4/int8)
- **Scaling factors** and **zero points** per group
- Forward pass: dequantize → matmul (fused with GemLite backend)

## Key Configuration Parameters

```python
from hqq.core.quantize import BaseQuantizeConfig

config = BaseQuantizeConfig(
    nbits=4,       # Number of bits: 1, 2, 3, 4, 8
    group_size=64, # Quantization group size (weight.numel() must be divisible)
    axis=1,        # Axis for grouping: 0=better quality, 1=faster inference
)
```

### Our Test Configurations

| nbits | group_size | Expected VRAM (8B model) | Quality |
|-------|-----------|--------------------------|---------|
| 4 | 64 | ~4-5 GB | Good |
| 3 | 64 | ~3-4 GB | Moderate |
| 2 | 64 | ~2-3 GB | Lower |

## Per-Layer Quantization Config

We quantize ALL attention and MLP linear layers uniformly. Our `make_quant_config` uses HQQ's short "linear_tag" convention (e.g. `self_attn.q_proj`), which the stock `AutoHQQHFModel` maps to all matching layers across all transformer blocks.

Note: subspec_v2 uses a custom `AutoHQQHFModel` subclass that overrides `name_to_linear_tag` to use fully-qualified names (e.g. `model.layers.0.self_attn.q_proj`). Both approaches work; the short-tag style is simpler for uniform quantization.

```python
def make_quant_config(model, nbits=4, group_size=64, axis=1):
    """Generate per-layer HQQ quantization config."""
    quant_config = {}
    # Short tags — applied to all layers by AutoHQQHFModel
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        quant_config[f"self_attn.{proj}"] = BaseQuantizeConfig(
            nbits=nbits, group_size=group_size, axis=axis
        )
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        quant_config[f"mlp.{proj}"] = BaseQuantizeConfig(
            nbits=nbits, group_size=group_size, axis=axis
        )
    return quant_config
```

Note: Embedding and LM head layers are NOT quantized (they remain in full precision).

## Quantization + Inference Backend Setup

```python
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import HQQLinear, HQQBackend
from hqq.utils.patching import prepare_for_inference

# Step 1: Quantize model weights
AutoHQQHFModel.quantize_model(
    model,
    quant_config=quant_config,      # per-layer config dict
    compute_dtype=torch.float16,     # activation dtype
    device='cuda'
)

# Step 2: Set base backend
HQQLinear.set_backend(HQQBackend.PYTORCH)

# Step 3: Patch with GemLite optimized kernels
prepare_for_inference(model, backend="gemlite")

# Step 4: IMPORTANT — reset eval mode
# prepare_for_inference replaces HQQLinear modules with GemLiteLinearTriton
# modules that default to training=True. Must call model.eval() AFTER patching.
model.eval()
```

### GemLite Backend

GemLite provides fused dequantization + GEMM kernels:
- `A16Wn`: FP16 activations × Wn (quantized weights) — default
- `A8Wn_dynamic`: FP8 activations × Wn — optional via `SUBSPEC_GEMLITE_ACTIVATIONS=fp8`

Requirements for GemLite:
- `nbits` in {1, 2, 4} (note: 3-bit not supported by GemLite, falls back to PyTorch)
- `axis=1` (column-wise quantization)
- `compute_dtype=float16`

After patching, linear layers become `GemLiteLinearTriton` instances. Verify with:
```python
for name, mod in model.named_modules():
    if 'q_proj' in name:
        print(type(mod).__name__)  # Should print: GemLiteLinearTriton
        break
```

### GemLite Environment Knobs (subspec_v2)

subspec_v2's `HqqQuantizer` supports these env vars for GemLite tuning:
- `GEMLITE_AUTOTUNE` — autotune mode
- `GEMLITE_CONFIG` — path to a config JSON
- `SUBSPEC_GEMLITE_PACKING_BITWIDTH` — packing width
- `SUBSPEC_GEMLITE_KERNEL_CACHING` — enable/disable kernel caching
- `SUBSPEC_GEMLITE_ACTIVATIONS=fp8` — use FP8 dynamic activation quantization

## Performance: GPU-Fast but CPU-Bound

### Profiling Results (RTX 6000 Ada, Llama-3.1-8B, single-token decode)

**CUDA kernel time** (what the GPU actually spends):

| Component | Target (bf16) | Draft (HQQ 4b/g64) |
|-----------|--------------|---------------------|
| Linear layers | 17.7ms (cuBLAS gemvx, 225 calls) | 6.0ms (Triton gemv_revsplitK, 224 calls) |
| Total CUDA | 19.9ms | 9.8ms |

GemLite's Triton kernels are ~3x faster than bf16 cuBLAS for the linear layers on GPU.

**Wall-clock time** (what you actually measure):

| | Target (bf16) | Draft (HQQ 4b/g64) |
|---|---|---|
| Median decode | 23.7ms | 46.6ms |

The draft appears **2x slower** despite faster GPU kernels. The bottleneck is **CPU-side dispatch overhead**:

- `gemlite::forward_functional`: 27.5ms self CPU time (224 calls × ~123μs each)
- `aten::zeros`: 6ms — GemLite allocates a fresh output tensor per call
- Total CPU time: **56ms** vs ~10ms CUDA time

Each GemLite Triton kernel launch involves Python-side argument packing, JIT cache lookup, and output tensor allocation. For 224 calls per forward pass, this dominates.

### Why cuBLAS Doesn't Have This Problem

cuBLAS GEMV launches are thin C++ calls with ~2-5μs overhead each. Triton kernels go through Python → triton runtime → CUDA launch, costing ~120μs+ each.

### The Fix: torch.compile / CUDA Graphs

The subspec_v2 config uses `compile_mode: max-autotune-no-cudagraphs` to eliminate per-call Python overhead. `torch.compile` traces the full forward pass and fuses kernel launches, removing the Python dispatch bottleneck.

Without compilation, the HQQ draft model is **slower** than bf16 for single-token decode despite faster GPU kernels. This is a critical detail for any benchmark or production use.

## HQQ + FlashInfer: Orthogonal Composition

HQQ and FlashInfer operate on different parts of the computation and don't interact directly:

```
Input Token Embedding (full precision)
    ↓
For each Transformer Layer:
  ├─ Q = HQQLinear(hidden)       ← quantized weight, FP16 activation output
  ├─ K = HQQLinear(hidden)       ← quantized weight, FP16 activation output
  ├─ V = HQQLinear(hidden)       ← quantized weight, FP16 activation output
  │
  ├─ FlashInfer Attention:
  │   ├─ Append K,V to paged cache  (FP16 values written to cache)
  │   ├─ Run attention kernel        (operates on FP16 Q and cached K,V)
  │   └─ Output: attention(Q, cached_KV)
  │
  ├─ O = HQQLinear(attn_output)  ← quantized weight
  │
  └─ MLP:
      ├─ gate = HQQLinear(x)     ← quantized weight
      ├─ up = HQQLinear(x)       ← quantized weight
      └─ down = HQQLinear(gated) ← quantized weight
    ↓
Next Layer
```

**Key insight**: HQQ quantizes the linear projection **weights**. The activations (including Q, K, V tensors) remain in full precision (FP16). FlashInfer operates on these full-precision activations, so it's completely unaware of quantization.

### dtype Flow

- Model loaded as bf16
- FlashInfer patches applied (replaces LlamaAttention with FiLlamaAttention)
- HQQ quantization casts non-linear layers (norms, embed, lm_head) to fp16 via `_patch_other`
- GemLite layers produce fp16 activations
- KV cache must be fp16 (matching activation dtype, not the original bf16)

## Installation

```bash
pip install hqq
pip install gemlite  # or: pip install git+https://github.com/mobiusml/gemlite/
```

Or from the local HQQ repo:
```bash
cd /path/to/hqq && pip install .
```

## Memory Savings

For Llama-3.1-8B (7B parameters in linear layers):

| Precision | Model Size | Approx VRAM |
|-----------|-----------|-------------|
| FP16 | 14 GB | ~16 GB (with overhead) |
| 4-bit HQQ | ~3.5 GB | ~5 GB |
| 3-bit HQQ | ~2.6 GB | ~4 GB |
| 2-bit HQQ | ~1.75 GB | ~3 GB |

## Known Limitations

1. **3-bit not supported by GemLite**: Falls back to PyTorch dequantize + matmul (slower)
2. **axis=0** gives better quality but is slower (not optimized by GemLite)
3. **training=True after patching**: `prepare_for_inference` creates new `GemLiteLinearTriton` modules that default to `training=True`. Must call `model.eval()` after patching.
4. **CPU-bound without torch.compile**: Triton kernel dispatch overhead (~123μs/call × 224 calls = 27ms) makes uncompiled HQQ models slower than bf16 for single-token decode. torch.compile is required for actual speedup.
5. **Accuracy**: Lower bits → more quantization error → lower acceptance rate (this is what we measure)

## Reference Files

- HQQ library: `/home/brain_l/flashtree/base/hqq/`
- HQQ README: `/home/brain_l/flashtree/base/hqq/Readme.md`
- Quantizer wrapper: `subspec_v2/specdecodes/helpers/quantizers/hqq/__init__.py`
- Custom AutoHQQHFModel: `subspec_v2/specdecodes/helpers/quantizers/hqq/hf/base.py`
- Recipe example: `subspec_v2/specdecodes/helpers/recipes/subspec/hqq_4bit_postspec.py`
- GemLite source: `.venv/lib/python3.12/site-packages/gemlite/core.py`
- Benchmark script: `scripts/bench_decode_latency.py`
