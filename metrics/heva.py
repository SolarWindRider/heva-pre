import os
from typing import TYPE_CHECKING, Optional, Union
import torch
from torch import nn
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerationConfig,
    GenerateNonBeamOutput,
)
from transformers.utils import logging
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


# Variable names used to hold the cache at generation time
ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]

GENERATION_MODES_MAPPING = {
    GenerationMode.SAMPLE: "_sample",
    GenerationMode.GREEDY_SEARCH: "_sample",
    GenerationMode.BEAM_SEARCH: "_beam_search",
    GenerationMode.BEAM_SAMPLE: "_beam_search",
    GenerationMode.ASSISTED_GENERATION: "_assisted_decoding",
    # Deprecated methods
    GenerationMode.DOLA_GENERATION: "transformers-community/dola",
    GenerationMode.CONTRASTIVE_SEARCH: "transformers-community/contrastive-search",
    GenerationMode.GROUP_BEAM_SEARCH: "transformers-community/group-beam-search",
    GenerationMode.CONSTRAINED_BEAM_SEARCH: "transformers-community/constrained-beam-search",
}


def _sample_with_vattn_and_entropy(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(
        cur_len, input_ids.device, model_kwargs
    )

    # Monkey-patch attention to capture z (before reshape) for ALL layers.
    # The patch captures model_all_z_ref which is set to self._all_layers_z below.
    # This mirrors transformer_lens's hook_z: z per layer per token.
    model_all_z_ref = [{}]  # defined outside if block so it's always in scope
    if not hasattr(self, "_z_patched") or not self._z_patched:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        original_forward = Qwen3VLTextAttention.forward

        def patched_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_values is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            # Capture z before reshape: (batch, seq, heads, d_head)
            # model_all_z_ref[0] is the model's _all_layers_z dict
            model_all_z_ref[0][self.layer_idx] = attn_output.clone()

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        Qwen3VLTextAttention.forward = patched_forward
        self._z_patched = True

    # _all_layers_z: dict mapping layer_idx -> z tensor (batch, seq, heads, d_head)
    # The patch writes to this via model_all_z_ref[0] closure
    self._all_layers_z = {}
    model_all_z_ref[0] = self._all_layers_z

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if (
                generation_config.compile_config is not None
                and generation_config.compile_config.fullgraph
            ):
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(
            input_ids, generation_config, **model_kwargs
        )
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )

        # Cache attentions for ContextAwareLogitsProcessor (reads model._last_attentions)
        self._last_attentions = outputs.attentions

        # Collect all layers' z for DLA trace (mirrors transformer_lens hook_z)
        # The patch writes to self._all_layers_z (via model_all_z_ref[0] closure)
        num_layers = self.model.language_model.config.num_hidden_layers
        all_z_list = []
        for layer_idx in range(num_layers):
            z = self._all_layers_z.get(layer_idx)
            if z is not None:
                all_z_list.append(z)
            else:
                # Should not happen in normal flow; pad with zeros using first available
                first_z = next((v for v in self._all_layers_z.values()), None)
                if first_z is not None:
                    all_z_list.append(torch.zeros_like(first_z))
                else:
                    all_z_list.append(None)

        if any(z is not None for z in all_z_list):
            # Stack (num_layers, batch, seq, heads, d_head)
            non_none = [z for z in all_z_list if z is not None]
            if non_none:
                max_shape = non_none[0].shape
                padded = [
                    (
                        z
                        if z is not None
                        else torch.zeros(max_shape, device=next_token_logits.device)
                    )
                    for z in all_z_list
                ]
                all_z = torch.stack(padded, dim=0)
                self.gen_zs.append(all_z)  # keep on NPU during generation
            else:
                self.gen_zs.append(None)
        else:
            self.gen_zs.append(None)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # [NEW] Attention-guided token selection via DLA trace
        # Mirrors the reference code: compute backward causal path for each candidate,
        # verify if its attention focuses on critical_indices, then resample from survivors.
        if (
            getattr(self, "use_attention_guidance", False)
            and self.gen_zs
            and len(self.gen_zs) > 0
        ):
            critical_indices = getattr(self, "critical_indices", [])
            if critical_indices and len(critical_indices) > 0:
                top_k_vocab = getattr(self, "attn_guidance_top_k", 10)
                top_k_attn = getattr(self, "attn_guidance_topk_attn", 5)
                dla_entropy_threshold = getattr(self, "dla_entropy_threshold", None)

                # Only run DLA path computation for high-entropy tokens
                all_zs_current = self.gen_zs[-1]
                if all_zs_current is None:
                    has_attn_guidance_override = False
                elif dla_entropy_threshold is not None:
                    step_entropy = get_entropy(next_token_logits[0]).item()
                    if step_entropy < dla_entropy_threshold:
                        has_attn_guidance_override = False
                    else:
                        _, top_indices = torch.topk(
                            next_token_scores,
                            k=min(top_k_vocab, next_token_scores.shape[-1]),
                        )
                        top_indices = top_indices[0]
                        valid_candidates = []
                        valid_logits = []
                        for i in range(len(top_indices)):
                            tok_id = top_indices[i].item()
                            path, _ = compute_dla_path_for_token(
                                all_zs_current, self, tok_id, b=0
                            )
                            is_valid = verify_attention_focus_on_path(
                                outputs.attentions,
                                path,
                                critical_indices,
                                b=0,
                                top_k_attn=top_k_attn,
                            )
                            if is_valid:
                                valid_candidates.append(tok_id)
                                valid_logits.append(next_token_scores[0, tok_id].item())
                        if valid_candidates:
                            valid_logits_tensor = torch.tensor(
                                valid_logits, device=next_token_scores.device
                            )
                            normalized_probs = F.softmax(valid_logits_tensor, dim=-1)
                            next_tokens = torch.multinomial(
                                normalized_probs, num_samples=1
                            ).squeeze(1)
                            has_attn_guidance_override = True
                        else:
                            has_attn_guidance_override = False
                else:
                    _, top_indices = torch.topk(
                        next_token_scores,
                        k=min(top_k_vocab, next_token_scores.shape[-1]),
                    )
                    top_indices = top_indices[0]
                    valid_candidates = []
                    valid_logits = []
                    for i in range(len(top_indices)):
                        tok_id = top_indices[i].item()
                        path, _ = compute_dla_path_for_token(
                            all_zs_current, self, tok_id, b=0
                        )
                        is_valid = verify_attention_focus_on_path(
                            outputs.attentions,
                            path,
                            critical_indices,
                            b=0,
                            top_k_attn=top_k_attn,
                        )
                        if is_valid:
                            valid_candidates.append(tok_id)
                            valid_logits.append(next_token_scores[0, tok_id].item())
                    if valid_candidates:
                        valid_logits_tensor = torch.tensor(
                            valid_logits, device=next_token_scores.device
                        )
                        normalized_probs = F.softmax(valid_logits_tensor, dim=-1)
                        next_tokens = torch.multinomial(
                            normalized_probs, num_samples=1
                        ).squeeze(1)
                        has_attn_guidance_override = True
                    else:
                        has_attn_guidance_override = False
            else:
                has_attn_guidance_override = False
        else:
            has_attn_guidance_override = False

        gen_entropy = get_entropy(next_token_logits)
        gen_vattn = get_vattn(
            outputs.attentions, visual_token_indices=self.visual_token_indices
        )
        attn_acc_input, attn_acc_visual = get_attn_acc(
            outputs.attentions,
            visual_token_indices=self.visual_token_indices,
            inputs_token_indices=self.inputs_token_indices,
            topk=cur_len,
        )
        self.gen_entropy.append(gen_entropy)
        self.gen_vattn.append(gen_vattn)
        self.attn_acc_input.append(attn_acc_input)
        self.attn_acc_visual.append(attn_acc_visual)
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            # if output_attentions:
            #     decoder_attentions += (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #     if self.config.is_encoder_decoder:
            #         cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if has_attn_guidance_override:
            pass  # next_tokens already set in attention-guided block above
        elif do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

        # Clear _all_layers_z for the next generation step (z was already collected into gen_zs)
        self._all_layers_z.clear()

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def get_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (Batch_size,)
    return entropy


def get_vattn(attentions, visual_token_indices):
    """
    计算当前生成token对视觉token的平均注意力。

    Args:
        attentions: attention 列表
        visual_token_indices: (start_indices, end_indices), each shape (batch_size,)

    Returns:
        每batch的视觉注意力均值: (batch_size,)
    """
    attn = attentions[-1][:, :, -1, :]  # (batch_size, num_heads, seq_len)
    attn = torch.mean(attn, dim=1)  # (batch_size, seq_len)

    # visual_token_indices 现在是 (batch_size,) tensors
    visual_start = visual_token_indices[0]  # (batch_size,)
    visual_end = visual_token_indices[1]  # (batch_size,)

    # 对每个 batch 元素分别计算
    batch_size = attn.shape[0]
    results = []
    for b in range(batch_size):
        v_start = visual_start[b].item()
        v_end = visual_end[b].item()
        results.append(attn[b, v_start : v_end + 1].mean())

    return torch.stack(results)  # (batch_size,)


def get_attn_acc(attentions, visual_token_indices, inputs_token_indices, topk):
    """
    计算当前生成token对input tokens的attention中，top-k注意力里有多少比例指向视觉token和input token。

    Args:
        attentions: 模型输出的attention列表，最后一层为 attentions[-1]
        visual_token_indices: (start_indices, end_indices), each shape (batch_size,)
        inputs_token_indices: (start_indices, end_indices), each shape (batch_size,)
        topk: 取attention值最高的top-k个token

    Returns:
        attn_acc_input: 每batch的input token占比: (batch_size,)
        attn_acc_visual: 每batch的visual token占比: (batch_size,)
    """
    attn = attentions[-1][:, :, -1, :]  # (batch_size, num_heads, seq_len)
    attn = torch.mean(attn, dim=1)  # (batch_size, seq_len)

    _, topk_indices = torch.topk(attn, k=topk, dim=-1)  # (batch_size, topk)

    visual_start, visual_end = visual_token_indices  # each (batch_size,)
    input_start, input_end = inputs_token_indices  # each (batch_size,)

    batch_size = attn.shape[0]
    attn_acc_visual_list = []
    attn_acc_input_list = []

    for b in range(batch_size):
        seq_len = attn.shape[-1]
        v_s, v_e = visual_start[b].item(), visual_end[b].item()
        i_s, i_e = input_start[b].item(), input_end[b].item()

        # 构建当前batch的mask
        visual_mask = torch.zeros(seq_len, device=attn.device)
        visual_mask[v_s : v_e + 1] = 1.0

        input_mask = torch.zeros(seq_len, device=attn.device)
        input_mask[i_s : i_e + 1] = 1.0

        # 统计top-k中视觉token和input token的比例
        topk_is_visual = visual_mask[topk_indices[b]]  # (topk,)
        topk_is_input = input_mask[topk_indices[b]]  # (topk,)

        attn_acc_visual_list.append(topk_is_visual.sum().item() / topk)
        attn_acc_input_list.append(topk_is_input.sum().item() / topk)

    return torch.tensor(attn_acc_input_list), torch.tensor(attn_acc_visual_list)


def compute_dla_path_for_token(all_zs, model, token_id, b=0):
    """
    计算目标 token 的 DLA 因果路径（从后往前逐层反向溯源）。

    等价于 transformer_lens 的 get_backward_causal_path_for_token：
    对每个生成 token，从 W_U[token_id] 开始，通过 z[layer] @ W_O[layer] @ W_V[layer]
    逐层反向计算哪个 head 贡献最大。

    Args:
        all_zs: list of (num_layers, batch, seq, heads, d_head), one per generated token
                或单个 tensor (num_layers, batch, seq, heads, d_head)
        model: Qwen3VLForConditionalGeneration
        token_id: 目标 token 的 ID
        b: batch index (默认 0)

    Returns:
        Dict[int, Dict]: layer_idx -> {"head": head_idx, "score": contribution_score}
        以及 (num_layers, batch, heads, d_head) 的所有层 z 堆叠
    """
    # all_zs: list of tensors or single tensor
    if isinstance(all_zs, list):
        if not all_zs:
            return {}, None
        # 取最后一个 token 的 z（当前生成位置）
        all_zs = torch.stack(
            all_zs, dim=1
        )  # (num_layers, gen_tokens, batch, seq, heads, d_head)
        # 只取最后一个 token 的 z
        last_zs = all_zs[:, -1, b, -1, :, :]  # (num_layers, heads, d_head)
    else:
        last_zs = all_zs[:, b, -1, :, :]  # (num_layers, heads, d_head)

    num_layers = last_zs.shape[0]
    n_heads = last_zs.shape[1]
    head_dim = last_zs.shape[2]

    # 获取 W_U 目标向量 (d_model,) 或 (vocab, d_model)
    W_U = model.lm_head.weight  # (vocab, d_model)
    if W_U.shape[0] == model.model.language_model.config.hidden_size:
        target_vector = W_U[:, token_id].detach()
    else:
        target_vector = W_U[token_id, :].detach()

    path = {}

    # 从后往前（从最后一层到第一层）
    for layer_idx in reversed(range(num_layers)):
        z = last_zs[layer_idx]  # (heads, d_head)
        W_O = _get_layer_W_O(model, layer_idx)  # (heads, d_head, d_model)

        # head_outputs = z @ W_O: (heads, d_model)
        head_outputs = torch.einsum("hd,hdm->hm", z, W_O)
        # head_contributions = head_outputs @ target_vector: (heads,)
        head_contributions = head_outputs @ target_vector

        max_head = torch.argmax(head_contributions).item()
        max_score = head_contributions[max_head].item()

        path[layer_idx] = {"head": max_head, "score": max_score}

        # 更新 target_vector: 追溯到上一层
        # 在 GQA 结构下 W_V 无法按 Q head 分解（z 是 Q×KV 混合输出，W_V 只作用于 KV 维度）。
        # 改用 head_outputs 的均值来近似：取所有 head 的 W_O 输出均值作为残差传递的代理。
        # 这样保持 d_model 维度，同时保留 max_head 的主导方向信息。
        target_vector = head_outputs.mean(dim=0)  # (d_model,)
        # 归一化防止数值爆炸
        target_vector = target_vector / (target_vector.norm() + 1e-6)

    return {k: path[k] for k in sorted(path.keys())}, last_zs


def _get_layer_W_O(model, layer_idx):
    """获取指定层的 W_O 权重 (heads, d_head, d_model)。"""
    try:
        layer = model.model.language_model.layers[layer_idx]
        W_O = layer.self_attn.o_proj.weight  # (d_model, n_heads * head_dim)
        n_heads = model.model.language_model.config.num_attention_heads
        head_dim = W_O.shape[1] // n_heads  # use output_dim to compute head_dim
        return W_O.view(n_heads, head_dim, -1)  # (n_heads, d_head, d_model)
    except Exception:
        return None


def verify_attention_focus_on_path(
    attentions, head_path, critical_indices, b=0, top_k_attn=5
):
    """
    检查 head_path 中各层的 head 是否将注意力聚焦在 critical_indices 上。

    等价于 transformer_lens 的 verify_attention_focus。

    Args:
        attentions: attention tuple, attentions[-1] = (batch, heads, seq, seq)
        head_path: Dict[layer_idx, {"head": head_idx, "score": ...}]
        critical_indices: List[int] of token indices that are "critical" (e.g., numbers, operators)
        b: batch index
        top_k_attn: 取每个 head 的 top-k 注意力位置来检查

    Returns:
        bool: 如果 >= 30% 的层关注了 critical_indices 则返回 True
    """
    if not critical_indices or head_path is None:
        return False

    # attentions[-1]: (batch, heads, query_len, key_len)
    attn = attentions[-1][b]  # (heads, query, key)
    # 取最后一个 query（当前生成 token）的注意力
    attn_last = attn[:, -1, :]  # (heads, key_len)

    seq_len = attn_last.shape[-1]
    hit_count = 0
    total = len(head_path)

    for layer_idx, info in head_path.items():
        head = info["head"]
        attn_pattern = attn_last[head]  # (key_len,)

        k = min(top_k_attn, seq_len)
        _, top_indices = torch.topk(attn_pattern, k=k)
        top_list = top_indices.tolist()

        if any(idx in top_list for idx in critical_indices):
            hit_count += 1

    ratio = hit_count / total if total > 0 else 0
    return ratio >= 0.3
