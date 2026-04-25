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
    Compute context support for each candidate token using per-head z · W_O · W_U.

    For each context head and each candidate token:
        contribution = z[head] · W_O[head] · W_U[token]

    The support for a token is the sum of contributions from all context heads.
    This gives each token a DIFFERENT support value based on how each head's
    z vector contributes to that token's logit.

    Args:
        model: The model
        attentions: Attention tensors (not used anymore, z comes from model._last_z)
        token_ids: Candidate token ids (k,)
        context_heads: List of (layer, head) tuples (layer is ignored, only head matters)
        context_token_indices: (start, end) indices for context tokens

    Returns:
        support scores: (k,) - each candidate gets its own support score
    """
    if not context_heads or token_ids.numel() == 0:
        return torch.zeros(len(token_ids), device=token_ids.device)

    # Get cached z from last layer's attention module (before W_O reshape)
    # Shape: (batch, seq, heads, d_head) — because eager_attention_forward does transpose(1,2) before returning
    try:
        last_layer = model.model.language_model.layers[-1]
        z = getattr(last_layer.self_attn, "_last_z", None)
        if z is None:
            return torch.zeros(len(token_ids), device=token_ids.device)
    except AttributeError:
        return torch.zeros(len(token_ids), device=token_ids.device)

    # z shape: (batch, seq, heads, d_head) = (1, 1, 16, 128)
    num_heads = z.shape[2]  # heads dimension is at index 2
    head_dim = z.shape[3]   # d_head dimension is at index 3
    d_model = last_layer.self_attn.o_proj.weight.shape[0]

    # Get W_O (output projection) and W_U (unembedding)
    try:
        W_O = last_layer.self_attn.o_proj.weight
        # W_O shape: (d_model, n_heads * head_dim)
        # For 2B: (2048, 2048), for 4B: (2560, 4096)
        n_heads_cfg = model.model.language_model.config.num_attention_heads
        d_model = W_O.shape[0]
        head_dim = W_O.shape[1] // n_heads_cfg  # use output dim to compute head_dim
        W_O = W_O.view(n_heads_cfg, head_dim, d_model)  # (n_heads, head_dim, d_model)
    except AttributeError:
        W_O = None

    try:
        W_U = model.lm_head.weight  # (vocab, d_model)
    except AttributeError:
        W_U = None

    if W_O is None or W_U is None:
        return torch.zeros(len(token_ids), device=token_ids.device)

    supports = []
    for token_id in token_ids:
        token_support = 0.0

        for _, head_idx in context_heads:
            # z for last position, this head: (d_head,)
            # z shape: (batch, seq, heads, d_head) = (1, 1, 16, 128)
            head_z = z[0, -1, head_idx, :]  # (d_head,)

            # W_O for this head: (head_dim, d_model)
            head_W_O = W_O[head_idx, :, :]  # (head_dim, d_model)

            # head_output = z @ W_O[head]: (d_model,)
            head_output = head_z @ head_W_O  # (d_model,)

            # contribution = head_output · W_U[token]: scalar
            token_W_U = W_U[token_id.item(), :]  # (d_model,)
            contribution = (head_output * token_W_U).sum().item()

            token_support += contribution

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
        2. In final candidates, select top half with highest context support
        3. Return new scores tensor with kept tokens, others set to -inf

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

        # Check if context range is valid (visual tokens exist)
        ctx_start = self._context_token_indices[0][0].item()
        ctx_end = self._context_token_indices[1][0].item()
        if ctx_end <= ctx_start:
            # No valid context tokens detected, don't干预
            return scores

        # 4. Top-k filtering: keep top-k, mask rest to -inf
        k = min(self.top_k, scores.shape[-1])
        _, topk_ids = torch.topk(scores[0], k=k)  # (k,)

        # 5. In the top-k candidates, select top half with highest context support
        supports = self._compute_support(topk_ids)  # (k,)
        keep_count = max(1, k // 2)  # 保留 topk 的一半

        # Get the support value at the boundary
        sorted_supports, _ = torch.sort(supports, descending=True)
        threshold = sorted_supports[keep_count - 1].item()  # support 阈值

        # Build a mask for topk positions: True if support >= threshold
        keep_mask = (supports >= threshold)  # (k,)

        # Create a mask for all vocabulary: True = set to -inf
        drop_mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        drop_mask[topk_ids] = False  # topk tokens: don't drop
        drop_mask[topk_ids[~keep_mask]] = True  # bottom half of topk: drop

        # Set dropped tokens to -inf (in-place)
        scores[0, drop_mask] = -float("inf")

        return scores


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
        self._last_z = None

    def _register_hooks(self):
        """Register forward hooks to capture attention tensors and z."""
        # Remove existing hooks
        self._remove_hooks()

        def get_attention_hook(module, input, output):
            if hasattr(output, "attentions") and output.attentions is not None:
                self._attentions = output.attentions
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                self._hidden_states = output.hidden_states

        def capture_z_hook(module, input, output):
            # o_proj input is z (before W_O projection)
            # input[0]: (batch, heads, seq, d_head) or similar
            self._last_z = input[0].clone()

        # Register hook on the model backbone for attention weights
        if hasattr(self.model, "model"):
            backbone = self.model.model
            if hasattr(backbone, "language_model"):
                # Qwen3VLModel -> language_model (Qwen3VLTextModel)
                text_model = backbone.language_model
                if hasattr(text_model, "layers"):
                    for layer in text_model.layers:
                        self._hooks.append(layer.register_forward_hook(get_attention_hook))

                    # Also register z capture on last layer's o_proj
                    last_layer = text_model.layers[-1]
                    self._hooks.append(
                        last_layer.self_attn.o_proj.register_forward_hook(capture_z_hook)
                    )

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

    def get_last_z(self):
        """Get cached z from last layer's attention output (before W_O)."""
        return self._last_z

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
