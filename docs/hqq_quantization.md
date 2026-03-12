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

We quantize ALL attention and MLP linear layers uniformly:

```python
def make_quant_config(model, nbits=4, group_size=64, axis=1):
    """Generate per-layer HQQ quantization config."""
    quant_config = {}
    config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=axis)

    for i in range(len(model.model.layers)):
        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            quant_config[f"model.layers.{i}.self_attn.{proj}"] = config
        # MLP projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            quant_config[f"model.layers.{i}.mlp.{proj}"] = config

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
```

### GemLite Backend

GemLite provides fused dequantization + GEMM kernels:
- `A16Wn`: FP16 activations × Wn (quantized weights) — default
- `A8Wn_dynamic`: FP8 activations × Wn — optional via `SUBSPEC_GEMLITE_ACTIVATIONS=fp8`

Requirements for GemLite:
- `nbits` in {1, 2, 4} (note: 3-bit not supported by GemLite, falls back to PyTorch)
- `axis=1` (column-wise quantization)
- `compute_dtype=float16`

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
3. **CUDA graphs**: HQQ linear layers are compatible with CUDA graph capture
4. **Accuracy**: Lower bits → more quantization error → lower acceptance rate (this is what we measure)

## Reference Files

- HQQ library: `/home/brain_l/flashtree/base/hqq/`
- HQQ README: `/home/brain_l/flashtree/base/hqq/Readme.md`
- Quantizer wrapper: `subspec_v2/specdecodes/helpers/quantizers/hqq/__init__.py`
- Recipe example: `subspec_v2/specdecodes/helpers/recipes/quant/hqq_4bit.py`
