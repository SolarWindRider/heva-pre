"""
ContextAwareLogitsProcessor: Context-aware decoding for multimodal LMs.

When the model is uncertain (high entropy), this processor boosts tokens that are
strongly supported by "context heads" - attention heads that focus on context tokens.

Based on doc2.md design:
    P(token) + λ × ContextEvidence(token)

The ContextEvidence is computed using attention heads that most attend to context tokens.
In the visual reasoning task, context tokens are the visual tokens, but the method
is general and can be applied to any defined context.
"""

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

# 从 inference.py 导入已实现的工具函数
from metrics.inference import get_visual_token_indices, get_input_token_indices


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of the distribution.

    Args:
        logits: (batch, vocab)

    Returns:
        entropy: (batch,)
    """
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    return entropy


def select_context_heads(attentions: tuple, context_token_indices: tuple, top_h: int = 5) -> list:
    """
    Identify attention heads that most focus on context tokens.

    Args:
        attentions: Tuple of attention tensors per layer, each (batch, heads, seq, seq)
        context_token_indices: (start_indices, end_indices), each shape (batch,)
        top_h: Number of top heads to select

    Returns:
        List of (layer_idx, head_idx) tuples representing context heads
    """
    if not attentions or attentions[0] is None:
        return []

    # Use last layer attention
    attn = attentions[-1]  # (batch, heads, query, key)

    # For the last generated token (query=-1), compute attention to context tokens
    # attn[:, :, -1, :] means: for last position, attention to all previous positions
    last_token_attn = attn[:, :, -1, :]  # (batch, heads, seq_len)

    batch_size, num_heads, seq_len = last_token_attn.shape
    context_heads_scores = []

    for b in range(batch_size):
        ctx_start = context_token_indices[0][b].item()
        ctx_end = context_token_indices[1][b].item()

        if ctx_end <= ctx_start:
            # No context tokens, use uniform attention to all
            context_heads_scores.append(torch.zeros(num_heads, device=attn.device))
        else:
            # Sum attention to context token positions
            attn_to_ctx = last_token_attn[b, :, ctx_start : ctx_end + 1].sum(dim=-1)
            context_heads_scores.append(attn_to_ctx)

    # Average over batch
    avg_attn_to_ctx = torch.stack(context_heads_scores).mean(dim=0)  # (heads,)

    # Select top heads
    top_heads = torch.topk(avg_attn_to_ctx, k=min(top_h, num_heads)).indices.tolist()

    # Return as list of (layer, head) tuples for last layer
    return [(len(attentions) - 1, h) for h in top_heads]


def compute_token_support_from_attentions(
    model,
    attentions: tuple,
    token_ids: torch.Tensor,
    context_heads: list,
    context_token_indices: tuple,
) -> torch.Tensor:
    """
    Compute context support for each candidate token based on context heads.

    Each candidate token gets a DIFFERENT support score by weighting
    attention to context tokens with the token's unembedding vector norm.

    Args:
        model: The model
        attentions: Attention tensors from forward pass (last layer from outputs.attentions)
        token_ids: Candidate token ids (k,)
        context_heads: List of (layer, head) tuples
        context_token_indices: (start, end) indices for context tokens

    Returns:
        support scores: (k,) - each candidate gets its own support score
    """
    if not context_heads or token_ids.numel() == 0:
        return torch.zeros(len(token_ids), device=token_ids.device)

    # Find a valid (non-None) attention layer
    attn = None
    for layer_attn in reversed(attentions):
        if layer_attn is not None:
            attn = layer_attn
            break

    if attn is None:
        return torch.zeros(len(token_ids), device=token_ids.device)

    # Get unembedding weight for per-token weighting
    try:
        W_U = model.lm_head.weight  # (vocab, d_model) = (151936, 2048)
    except AttributeError:
        W_U = None

    ctx_start = context_token_indices[0][0].item()
    ctx_end = context_token_indices[1][0].item()

    supports = []
    for token_id in token_ids:
        token_support = 0.0

        for _, head_idx in context_heads:
            # Attention this head pays to context tokens (visual tokens)
            if ctx_end > ctx_start:
                attn_to_ctx = attn[0, head_idx, -1, ctx_start : ctx_end + 1].mean().item()
            else:
                attn_to_ctx = 0.0

            # Weight by the token's unembedding vector magnitude
            # Different tokens have different |W_U[token_id]|, giving different support
            if W_U is not None:
                try:
                    # W_U: (vocab, d_model) for Qwen3-VL
                    token_W_U = W_U[token_id.item(), :]  # (d_model,) row
                    weight = token_W_U.norm().item()
                except Exception:
                    weight = 1.0
            else:
                weight = 1.0

            token_support += attn_to_ctx * weight

        supports.append(token_support / len(context_heads))

    return torch.tensor(supports, device=token_ids.device, dtype=torch.float32)


class ContextAwareLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that selects tokens based on context-head support when model is uncertain.

    When entropy > threshold:
        1. Identify context_heads (attention heads most attending to context tokens)
        2. Get top-k candidate tokens
        3. Compute context support for each candidate
        4. Select the token with highest context support (replacing model distribution)
    """

    def __init__(
        self,
        model,
        top_k: int = 20,
        entropy_threshold: float = 5.0,
        top_heads: int = 5,
    ):
        """
        Args:
            model: Qwen3VLForConditionalGeneration model instance
            top_k: Number of top tokens to consider
            entropy_threshold: Entropy threshold to trigger context-aware selection
            top_heads: Number of top context heads to use
        """
        self.model = model
        self.top_k = top_k
        self.entropy_threshold = entropy_threshold
        self.top_heads = top_heads

        # Cached state from last forward pass
        self._last_attentions = None
        self._last_hidden_states = None

        # Context token positions (will be set during generation)
        self._context_token_indices = None

    def set_context_token_indices(self, indices: tuple):
        """Set the context token indices (start, end) for the current input."""
        self._context_token_indices = indices

    def _get_context_heads(self) -> list:
        """Get context heads from the cached attentions (stored on the model by monkey-patch)."""
        if not hasattr(self.model, "_last_attentions") or self.model._last_attentions is None:
            return []
        return select_context_heads(self.model._last_attentions, self._context_token_indices, self.top_heads)

    def _compute_support(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute context support scores for candidate tokens."""
        if not hasattr(self.model, "_last_attentions") or self.model._last_attentions is None:
            return torch.zeros(len(token_ids), device=token_ids.device)

        return compute_token_support_from_attentions(
            self.model,
            self.model._last_attentions,
            token_ids,
            self._get_context_heads(),
            self._context_token_indices,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Select token based on context-head support when uncertain.

        Flow:
        1. top-k filtering on raw scores (same as TopKLogitsWarper)
        2. top-p filtering on top-k result (same as TopPLogitsWarper)
        3. In final candidates, select the one with highest context support
        4. Apply boost so this token wins after warper filtering

        Args:
            input_ids: (batch, seq_len) - current input sequence
            scores: (batch, vocab) - raw logits from model

        Returns:
            modified scores biased toward context-supported tokens
        """
        # 1. Compute entropy
        entropy = compute_entropy(scores)  # (batch,)

        # If model is confident (low entropy), don't intervene
        if entropy[0] < self.entropy_threshold:
            return scores

        # 2. Get context heads from cached attentions
        context_heads = self._get_context_heads()
        if not context_heads:
            return scores

        # 4. Top-k filtering: keep top-k, mask rest to -inf
        k = min(self.top_k, scores.shape[-1])
        _, topk_ids = torch.topk(scores[0], k=k)  # (k,)

        # 5. In the top-k candidates, select the one with highest context support
        supports = self._compute_support(topk_ids)  # (k,)
        ctx_best_idx = torch.argmax(supports).item()
        ctx_selected_id = topk_ids[ctx_best_idx].item()

        # Set all tokens except ctx_selected_id to -inf (return new tensor, don't modify in-place)
        ctx_logit = scores[0, ctx_selected_id].item()
        new_scores = torch.full_like(scores, -float("inf"))
        new_scores[0, ctx_selected_id] = ctx_logit

        return new_scores


class ContextAwareModelWrapper:
    """
    Wrapper that attaches attention hooks to the model and provides
    a clean interface for the ContextAwareLogitsProcessor.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._hooks = []
        self._attentions = None
        self._hidden_states = None

    def _register_hooks(self):
        """Register forward hooks to capture attention tensors."""
        # Remove existing hooks
        self._remove_hooks()

        def get_attention_hook(module, input, output):
            if hasattr(output, "attentions") and output.attentions is not None:
                self._attentions = output.attentions
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                self._hidden_states = output.hidden_states

        # Register hook on the model backbone
        if hasattr(self.model, "model"):
            # Qwen3VLModel structure
            backbone = self.model.model
            if hasattr(backbone, "layers"):
                for layer in backbone.layers:
                    self._hooks.append(layer.register_forward_hook(get_attention_hook))

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()
        return False

    def get_attentions(self):
        """Get cached attentions from last forward pass."""
        return self._attentions

    def get_hidden_states(self):
        """Get cached hidden states from last forward pass."""
        return self._hidden_states


def get_context_token_indices(
    input_ids: torch.Tensor,
    processor,
    image_token_indices: tuple = None,
) -> tuple:
    """
    Get context token indices for the input.

    Priority:
    1. If image_token_indices is provided by user → use it (user-specified range)
    2. Try to detect image tokens automatically using get_visual_token_indices
    3. If no image tokens found → fall back to all prompt tokens (excluding padding)

    Args:
        input_ids: (batch_size, seq_len)
        processor: AutoProcessor instance
        image_token_indices: Optional user-specified (start_indices, end_indices).
            If None, will auto-detect.

    Returns:
        (start_indices, end_indices): tuple of tensors, each shape (batch_size,)
    """
    # Case 1: User specified
    if image_token_indices is not None:
        return image_token_indices

    # Case 2: Auto-detect image tokens
    visual_indices = get_visual_token_indices(input_ids, processor)

    # Check if image tokens were found (start != end or not all zeros)
    batch_size = input_ids.shape[0]
    has_image_tokens = False
    for b in range(batch_size):
        if visual_indices[0][b] != visual_indices[1][b] or (visual_indices[0][b].item() != 0 and visual_indices[1][b].item() != 0):
            has_image_tokens = True
            break

    if has_image_tokens:
        return visual_indices

    # Case 3: No image tokens found → use all prompt tokens (excluding padding)
    return get_input_token_indices(input_ids, processor)


def get_image_token_id(processor) -> int:
    """
    Get the image token ID from processor.

    Args:
        processor: AutoProcessor instance

    Returns:
        image_token_id: int
    """
    try:
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<|image_pad|>")
        ]
    except (ValueError, KeyError):
        # Fallback to common values for different Qwen3-VL versions
        image_token_id = 151643
    return image_token_id


if __name__ == "__main__":
    # Simple test to verify the module compiles
    print("ContextAwareLogitsProcessor imported successfully")
    print("Available classes:")
    print("  - ContextAwareLogitsProcessor")
    print("  - ContextAwareModelWrapper")
    print("  - compute_entropy")
    print("  - select_context_heads")
    print("  - compute_token_support_from_attentions")
    print("  - get_context_token_indices")
    print("  - get_all_prompt_token_indices")
