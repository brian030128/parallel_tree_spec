# Speculative Decoding: Draft Phase and Verify Phase

## Overview

Speculative decoding accelerates LLM inference by using a fast "draft" model to propose candidate tokens, then using the full "target" model to verify them in parallel. When the draft model's predictions match, multiple tokens are accepted per target model forward pass, achieving speedups of 2-3x.

## Two Phases

### 1. Draft Phase (Beam Search)

The draft model proposes candidate tokens by running **beam search** — maintaining K top-scoring hypotheses simultaneously.

**Algorithm** (beam width K=6, depth=10):

```
PREFILL:
  Feed full prompt through draft model → get logits for last position
  Select top-K tokens → K initial beams, each with cum_log_prob

DECODE LOOP (9 steps):
  For each step:
    1. Feed last token of each beam through draft model as a batch [K, 1]
    2. Compute scores: cum_log_prob[beam] + log_prob[beam][token] → [K, vocab]
    3. Flatten to [K × vocab], select top-K (parent_beam, new_token) pairs
    4. Beams can fork (2 children from same parent) or die (not in top-K)
    5. Build tree: new tokens become children of their parent beam's node

OUTPUT: Tree structure with ~K×depth candidate tokens, organized as a trie
```

**Why beam search instead of greedy?**
- Explores multiple token paths simultaneously
- The tree of candidates provides more opportunities for verification matches
- The "best" beam may not always start with the greedily-best first token

**Key efficiency concern**: The draft model runs K forward passes per step (batched). Quantization (HQQ) reduces the draft model's memory footprint and speeds up these forward passes.

### 2. Verify Phase (Target Model)

The full-precision target model verifies the draft tree in a **single forward pass**.

**Algorithm:**

```
PREPARE TREE INPUTS:
  1. Flatten tree nodes into a token sequence: tree_input_ids [N tokens]
  2. Build tree attention mask: each node attends to its ancestors + prompt prefix
     (this is NOT a causal mask — it's a tree-shaped mask)
  3. Compute position_ids: each node's position = prompt_length + node_depth

TARGET FORWARD:
  Feed all N tree tokens through target model in one batched forward pass
  Use tree attention mask so each node only sees its correct context
  Output: logits for each tree node [N, vocab_size]

EXACT VERIFICATION:
  Starting from root:
    1. Sample target token from logits at current node (argmax for greedy)
    2. Check if target token matches any child in the draft tree
    3. If MATCH: accept that child, move to it, repeat
    4. If NO MATCH: stop, emit "bonus token" from current position

  Return: number of accepted tokens (accept_len)
```

**Why tree attention?**
Without tree attention, the target model would need one forward pass per beam path (K passes). With tree attention, shared prefix computation is amortized, and all paths are verified in a single forward pass.

## Acceptance Rate and Depth

The key metric is **accepted tokens per depth**. For each depth d in the draft tree:
- How often does the target model agree with the draft's prediction at that depth?
- Acceptance typically decreases with depth because errors compound

Higher quantization (fewer bits) → more approximation → lower acceptance rates. Our experiment measures this tradeoff.

## Data Flow

```
                 ┌──────────────────┐
  Prompt ──────→ │   Draft Model    │──→ Draft Tree (beam search)
                 │ (quantized, fast)│        │
                 └──────────────────┘        │
                                             ▼
                 ┌──────────────────┐    ┌──────────┐
                 │  Target Model    │──→ │ Verify   │──→ accepted_tokens
                 │ (full precision) │    │ (exact)  │    per-depth stats
                 └──────────────────┘    └──────────┘
```

## Verification Methods

Our project uses **exact verification** with greedy matching (`do_sample=False`):
- At each tree node, target model's argmax token is computed
- If it matches a child in the draft tree, that child is accepted
- Otherwise, verification stops and a bonus token is emitted

The subspec_v2 codebase also supports:
- **Lossy verification**: can accept non-matching tokens under certain conditions (entropy/probability thresholds)
- **Traversal verification**: probabilistic acceptance using cumulative draft probabilities

We use exact verification for clean measurement of quantization impact.

## Reference Implementation

The draft phase beam search is adapted from `subspec_v2/specdecodes/models/draft_models/be_classic_sd_fi.py`.
The verification is adapted from `subspec_v2/specdecodes/models/utils/tree_verify.py` (exact method).
