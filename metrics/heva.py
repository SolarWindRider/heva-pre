import os
from typing import TYPE_CHECKING, Optional, Union
import torch
from torch import nn
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerationConfig, GenerateNonBeamOutput
from transformers.utils import logging
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
import torch.nn.functional as F

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
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
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
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        gen_entropy = get_entropy(next_token_logits)
        gen_vattn = get_vattn(outputs.attentions, visual_token_indices=self.visual_token_indices)
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
                decoder_hidden_states += (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

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
    visual_end = visual_token_indices[1]    # (batch_size,)

    # 对每个 batch 元素分别计算
    batch_size = attn.shape[0]
    results = []
    for b in range(batch_size):
        v_start = visual_start[b].item()
        v_end = visual_end[b].item()
        results.append(attn[b, v_start:v_end + 1].mean())

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
    input_start, input_end = inputs_token_indices    # each (batch_size,)

    batch_size = attn.shape[0]
    attn_acc_visual_list = []
    attn_acc_input_list = []

    for b in range(batch_size):
        seq_len = attn.shape[-1]
        v_s, v_e = visual_start[b].item(), visual_end[b].item()
        i_s, i_e = input_start[b].item(), input_end[b].item()

        # 构建当前batch的mask
        visual_mask = torch.zeros(seq_len, device=attn.device)
        visual_mask[v_s:v_e + 1] = 1.0

        input_mask = torch.zeros(seq_len, device=attn.device)
        input_mask[i_s:i_e + 1] = 1.0

        # 统计top-k中视觉token和input token的比例
        topk_is_visual = visual_mask[topk_indices[b]]  # (topk,)
        topk_is_input = input_mask[topk_indices[b]]    # (topk,)

        attn_acc_visual_list.append(topk_is_visual.sum().item() / topk)
        attn_acc_input_list.append(topk_is_input.sum().item() / topk)

    return torch.tensor(attn_acc_input_list), torch.tensor(attn_acc_visual_list)
