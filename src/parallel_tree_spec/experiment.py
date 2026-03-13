"""
Experiment runner for HQQ-quantized draft model evaluation.

Loads target and draft models, runs beam search + verification on multiple
prompts, and collects metrics across quantization configs.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .beam_search import BeamSearchConfig, beam_search
from .flashinfer.attention_wrapper import BeFlashinferWrapper
from .flashinfer.cache_manager import KvCachePool, RequestKvCache, copy_kv_pages
from .flashinfer.monkey_patch import apply_flashinfer_kernel_to_llama
from .metrics import QuantConfigResult, SingleRunMetrics, SweepResults
from .quantization import make_quant_config, quantize_model
from .tree import Tree
from .verification import verify_draft_tree


logger = logging.getLogger(__name__)


class BeamSearchExperiment:
    """Runs beam search draft + target verify experiments."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        beam_width: int = 6,
        max_depth: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        page_len: int = 16,
        max_pages: int = 4096,
        share_kv: bool = False,
        temperature: float = 1.0,
        use_cuda_graph: bool = False,
        warmup_iters: int = 1,
    ):
        self.model_name = model_name
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.device = torch.device(device)
        self.dtype = dtype
        self.page_len = page_len
        self.max_pages = max_pages
        self.share_kv = share_kv
        self.temperature = temperature
        self.use_cuda_graph = use_cuda_graph
        self.warmup_iters = warmup_iters

        self.tokenizer = None
        self.target_model = None
        self.target_kv_pool = None
        self.target_wrapper = None

        self.draft_model = None
        self.draft_kv_pool = None
        self.draft_wrapper = None
        self.draft_cuda_runner = None

    def load_tokenizer(self):
        """Load tokenizer."""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_target_model(self):
        """Load full-precision target model with FlashInfer patches."""
        self.load_tokenizer()

        logger.info(f"Loading target model: {self.model_name} ({self.dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        model.eval()
        apply_flashinfer_kernel_to_llama(model)

        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.target_kv_pool = KvCachePool(
            max_pages=self.max_pages,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            page_len=self.page_len,
            dtype=self.dtype,
            device=self.device,
        )

        self.target_wrapper = BeFlashinferWrapper(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            page_len=self.page_len,
        )

        self.target_model = model
        logger.info("Target model loaded")

    def load_draft_model(self, nbits: int = 4, group_size: int = 64):
        """Load and quantize draft model with FlashInfer patches.

        Args:
            nbits: Quantization bits (2, 3, or 4)
            group_size: Quantization group size
        """
        self.load_tokenizer()

        logger.info(f"Loading draft model: {self.model_name} (HQQ {nbits}b/g{group_size})")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        model.eval()

        # Apply FlashInfer patches BEFORE quantization
        apply_flashinfer_kernel_to_llama(model)

        # Quantize
        quant_config = make_quant_config(model, nbits=nbits, group_size=group_size)
        quantize_model(model, quant_config, compute_dtype=torch.float16, device=str(self.device))

        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.draft_kv_pool = KvCachePool(
            max_pages=self.max_pages,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            page_len=self.page_len,
            dtype=torch.float16,
            device=self.device,
        )

        self.draft_wrapper = BeFlashinferWrapper(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            page_len=self.page_len,
        )

        self.draft_model = model
        logger.info(f"Draft model loaded (HQQ {nbits}b/g{group_size})")

    def load_draft_model_unquantized(self):
        """Load unquantized bf16 draft model (same as target, for baseline)."""
        self.load_tokenizer()

        logger.info(f"Loading draft model (unquantized bf16): {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        model.eval()
        apply_flashinfer_kernel_to_llama(model)

        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.draft_kv_pool = KvCachePool(
            max_pages=self.max_pages,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            page_len=self.page_len,
            dtype=torch.bfloat16,
            device=self.device,
        )

        self.draft_wrapper = BeFlashinferWrapper(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            page_len=self.page_len,
        )

        self.draft_model = model
        logger.info("Draft model loaded (unquantized baseline)")

    def unload_draft_model(self):
        """Free draft model memory."""
        if self.draft_model is not None:
            del self.draft_model
            self.draft_model = None
        if self.draft_kv_pool is not None:
            del self.draft_kv_pool
            self.draft_kv_pool = None
        if self.draft_wrapper is not None:
            del self.draft_wrapper
            self.draft_wrapper = None
        if self.draft_cuda_runner is not None:
            del self.draft_cuda_runner
            self.draft_cuda_runner = None
        torch.cuda.empty_cache()

    def run_single(self, prompt_ids: torch.Tensor) -> SingleRunMetrics:
        """Run one draft+verify cycle on a single prompt.

        Args:
            prompt_ids: [1, seq_len] prompt token IDs

        Returns:
            SingleRunMetrics for this run
        """
        assert self.target_model is not None, "Target model not loaded"
        assert self.draft_model is not None, "Draft model not loaded"

        # Reset KV pools
        self.draft_kv_pool.reset()
        self.target_kv_pool.reset()

        # Create request KV caches
        draft_kv_cache = RequestKvCache(
            kvCachePool=self.draft_kv_pool,
            page_len=self.page_len,
            seq_init_len=0,
        )
        target_kv_cache = RequestKvCache(
            kvCachePool=self.target_kv_pool,
            page_len=self.page_len,
            seq_init_len=0,
        )

        # --- Target prefill ---
        target_kv_cache.increment(prompt_ids.shape[1])
        from .flashinfer.cache_manager import getKvCacheBatchPosition

        target_batch_pos = getKvCacheBatchPosition(
            request_kv_caches=[target_kv_cache],
            mode="tree",
            device=self.device,
            treeTokens=prompt_ids.shape[1],
        )
        self.target_wrapper.prepareAttention(
            "prefill",
            target_batch_pos,
            self.page_len,
            "NONE",
            self.target_kv_pool.cache_data[0].dtype,
        )
        position_ids = torch.arange(
            prompt_ids.shape[1], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        target_outputs = self.target_model(
            prompt_ids,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            kvCachePool=self.target_kv_pool,
            batch_position=target_batch_pos,
            mode="prefill",
            flashinferWrapper=self.target_wrapper,
        )

        # --- Target single-token decode benchmark ---
        test_token_id = target_outputs.logits[0, -1, :].argmax().item()
        test_input = torch.tensor([[test_token_id]], dtype=torch.long, device=self.device)
        test_pos = torch.tensor([[prompt_ids.shape[1]]], dtype=torch.long, device=self.device)

        target_kv_cache.increment(1)
        target_decode_pos = getKvCacheBatchPosition(
            request_kv_caches=[target_kv_cache],
            mode="tree",
            device=self.device,
            treeTokens=1,
        )
        self.target_wrapper.prepareAttention(
            "decode", target_decode_pos, self.page_len, "NONE",
            self.target_kv_pool.cache_data[0].dtype,
        )

        torch.cuda.synchronize()
        t_target_decode_start = time.perf_counter()
        self.target_model(
            test_input,
            position_ids=test_pos,
            past_key_values=None,
            use_cache=False,
            kvCachePool=self.target_kv_pool,
            batch_position=target_decode_pos,
            mode="decode",
            flashinferWrapper=self.target_wrapper,
        )
        torch.cuda.synchronize()
        target_decode_time = time.perf_counter() - t_target_decode_start

        # Undo the KV increment so verification sees the original state
        target_kv_cache.decrement(1)

        # --- Draft: beam search ---
        beam_config = BeamSearchConfig(
            topk_len=self.beam_width,
            max_depth=self.max_depth,
            temperature=self.temperature,
            use_cascade=True,
            use_cuda_graph=self.use_cuda_graph,
        )

        prefilled_logits = None
        if self.share_kv:
            # Copy target's KV cache to draft pool (with dtype cast)
            draft_kv_cache = copy_kv_pages(
                src_pool=self.target_kv_pool,
                src_request=target_kv_cache,
                dst_pool=self.draft_kv_pool,
            )
            # Use target's prefill logits so draft skips its own prefill
            prefilled_logits = target_outputs.logits

        tree, step_times, cuda_runner = beam_search(
            model=self.draft_model,
            request_kv_cache=draft_kv_cache,
            input_ids=prompt_ids,
            config=beam_config,
            flashinfer_wrapper=self.draft_wrapper,
            prefilled_logits=prefilled_logits,
            cuda_graph_runner=self.draft_cuda_runner,
        )
        self.draft_cuda_runner = cuda_runner

        # --- Verify: target model scores the tree ---
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        verify_result = verify_draft_tree(
            target_model=self.target_model,
            flashinfer_wrapper=self.target_wrapper,
            tree=tree,
            request_kv_cache=target_kv_cache,
            position_offset=prompt_ids.shape[1] - 1,
            device=self.device,
            eos_token_id=eos_token_id,
        )

        # Collect per-depth acceptance info
        per_depth = {}
        for d in range(1, tree.get_depth() + 1):
            per_depth[d] = verify_result.accept_len >= d

        # Release KV caches
        draft_kv_cache.release()
        target_kv_cache.release()

        return SingleRunMetrics(
            accept_len=verify_result.accept_len,
            total_len=verify_result.total_len,
            draft_step_times=step_times,
            verify_time=verify_result.verify_time,
            tree_size=tree.current_size,
            tree_depth=tree.get_depth(),
            target_decode_time=target_decode_time,
            per_depth_accepted=per_depth,
        )

    def run_sweep(
        self,
        prompts: List[str],
        quant_configs: List[Tuple[int, int]],
    ) -> SweepResults:
        """Run full sweep across quantization configs and prompts.

        Args:
            prompts: List of prompt strings
            quant_configs: List of (nbits, group_size) tuples

        Returns:
            SweepResults with aggregated metrics
        """
        self.load_tokenizer()
        self.load_target_model()

        results = SweepResults(
            model_name=self.model_name,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
        )

        for nbits, group_size in quant_configs:
            logger.info(f"\n{'='*60}")
            if nbits == 0:
                logger.info("Config: unquantized bf16 (baseline)")
            else:
                logger.info(f"Config: {nbits}b / g{group_size}")
            logger.info(f"{'='*60}")

            if nbits == 0:
                self.load_draft_model_unquantized()
            else:
                self.load_draft_model(nbits=nbits, group_size=group_size)
            config_result = QuantConfigResult(nbits=nbits, group_size=group_size)

            # Warm-up: run draft+verify cycles to JIT-compile CUDA kernels
            if self.warmup_iters > 0:
                warmup_ids = self.tokenizer.encode(prompts[0], return_tensors="pt").to(self.device)
                for wi in range(self.warmup_iters):
                    logger.info(f"  Warm-up {wi+1}/{self.warmup_iters}...")
                    self.run_single(warmup_ids)
                torch.cuda.synchronize()
                logger.info("  Warm-up complete")

            for i, prompt in enumerate(prompts):
                logger.info(f"  Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                try:
                    metrics = self.run_single(prompt_ids)
                    config_result.runs.append(metrics)
                    logger.info(
                        f"    Accept: {metrics.accept_len}/{metrics.total_len}, "
                        f"Tree: {metrics.tree_size} nodes, "
                        f"Draft: {sum(metrics.draft_step_times)*1000:.1f}ms, "
                        f"Verify: {metrics.verify_time*1000:.1f}ms, "
                        f"Target: {metrics.target_decode_time*1000:.1f}ms"
                    )
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    import traceback
                    traceback.print_exc()

            results.configs.append(config_result)
            self.unload_draft_model()

        return results


# ---------------------------------------------------------------------------
# Default prompts for evaluation
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "The theory of general relativity, proposed by Albert Einstein in 1915,",
    "In computer science, a hash table is a data structure that",
    "The process of photosynthesis in plants involves the conversion of",
    "Machine learning algorithms can be broadly categorized into three types:",
    "The French Revolution, which began in 1789, was a period of",
    "In organic chemistry, a benzene ring is a cyclic compound consisting of",
    "The TCP/IP protocol suite is the foundation of internet communication,",
    "Quantum computing leverages the principles of quantum mechanics to",
    "The human immune system consists of two main subsystems:",
    "In economics, the concept of supply and demand describes how",
    "The Great Wall of China, one of the most impressive architectural feats,",
    "Neural networks are computational models inspired by biological",
    "The Pythagorean theorem states that in a right-angled triangle,",
    "Climate change refers to long-term shifts in temperatures and weather",
    "In philosophy, epistemology is the branch that studies the nature of",
    "The periodic table of elements organizes all known chemical elements",
    "Blockchain technology provides a decentralized and transparent way to",
    "The Renaissance was a cultural movement that began in Italy during",
    "In distributed systems, the CAP theorem states that it is impossible to",
    "The human brain contains approximately 86 billion neurons that",
]
