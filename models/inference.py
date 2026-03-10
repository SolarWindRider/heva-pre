"""
HEVA 模型推理模块
支持 Qwen3-VL 推理并捕获 attention 和 logits
支持 NVIDIA GPU 和华为 NPU
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import sys

# 立即刷新输出的打印函数
def log_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

# 检测设备类型
def get_device():
    """自动检测并返回可用设备"""
    # 优先使用 NPU (华为昇腾)
    try:
        import torch_npu
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return "npu"
    except ImportError:
        pass

    # 其次使用 CUDA
    if torch.cuda.is_available():
        return "cuda"

    # 最后使用 CPU
    return "cpu"


class Qwen3VLInference:
    """Qwen3-VL 推理类"""

    def __init__(self, model_path: str, device: str = None, num_gpus: int = 1, heva_device: str = None):
        """
        Args:
            model_path: 模型路径
            device: 设备，默认为自动检测
            num_gpus: 推理使用的加速卡数量，默认为1
            heva_device: HEVA 计算设备，默认为 None (与 model 相同)
        """
        self.device = device or get_device()
        self.model_path = model_path
        self.num_gpus = num_gpus
        # HEVA 计算设备：如果指定，则把 HEVA 计算转移到另一张卡
        self.heva_device = heva_device if heva_device else self.device

        log_print(f"Loading model from {model_path}...")
        log_print(f"Using device: {self.device}")
        log_print(f"HEVA will be computed on: {self.heva_device}")
        log_print(f"Number of Accelerators: {num_gpus}")

        # 根据设备类型和推理加速卡数量设置 device_map
        if num_gpus > 1:
            # 多卡模式：使用 device_map="auto" 自动分布到多个GPU
            device_map = "auto"
        elif self.device == "npu":
            device_map = "npu"
        elif self.device == "cuda":
            device_map = "cuda"
        else:
            device_map = "cpu"

        # 设置使用 eager attention 以支持 attention 输出
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = "eager"

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            config=config
        )

        # 先移动到设备，再确保 bf16 精度
        if self.device != "cpu" and device_map != "auto":
            self.model = self.model.to(self.device)

        # 确保模型使用 bf16 精度
        self.model = self.model.to(dtype=torch.bfloat16)

        # 确认使用 eager attention
        self.model.config._attn_implementation = "eager"

        # 设置为评估模式
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        log_print(f"Model loaded on {self.device}")

        # 异步 HEVA 计算的线程池
        self._heva_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_heva_futures = {}

    def compute_heva_async(self, result_dict: dict, sample_idx: int):
        """
        异步计算 HEVA (在后台线程中计算)

        Args:
            result_dict: 推理结果字典，会被修改
            sample_idx: 样本索引，用于追踪

        Returns:
            Future 对象
        """
        def _compute():
            try:
                full_logits = result_dict.get('logits')
                full_attentions = result_dict.get('attentions')
                visual_token_indices = result_dict.get('visual_token_indices')
                prompt_length = result_dict.get('prompt_length')
                generated_ids = result_dict.get('generated_ids')

                if full_logits is None or full_attentions is None:
                    return 0.0

                # 检查tensor实际设备，判断是否需要转移
                target_device = result_dict.get('heva_device', 'cpu')

                # 如果设备已经相同，就不需要转移
                if str(full_logits.device) != str(target_device):
                    # 需要转移数据到HEVA设备
                    full_logits = full_logits.to(target_device)
                    # 只保留最后一层attention，大幅减少显存使用
                    full_attentions = tuple([full_attentions[-1].to(target_device)])
                    visual_token_indices = visual_token_indices.to(target_device)
                    generated_ids = generated_ids.to(target_device)

                # 导入 HEVA 配置
                import config

                # 导入 HEVA 计算函数
                from metrics.heva import compute_heva_from_result

                # 计算自定义 alpha 值的 HEVA
                heva_dict = {}
                for alpha in config.ALPHA_VALUES:
                    heva_result = compute_heva_from_result({
                        "logits": full_logits,
                        "attentions": full_attentions,
                        "visual_token_indices": visual_token_indices,
                        "prompt_length": prompt_length,
                        "input_ids": generated_ids,
                    }, alpha=alpha)
                    heva_dict[f'heva_{int(alpha*100)}'] = float(heva_result['heva'])

                return heva_dict
            except Exception as e:
                print(f"ERROR: Async HEVA computation failed: {e}")
                return {'heva_20': 0.0}

        future = self._heva_executor.submit(_compute)
        self._pending_heva_futures[sample_idx] = future
        return future

    def get_heva_result(self, sample_idx: int):
        """
        获取异步 HEVA 计算结果

        Args:
            sample_idx: 样本索引

        Returns:
            HEVA 字典或 0.0
        """
        if sample_idx in self._pending_heva_futures:
            future = self._pending_heva_futures.pop(sample_idx)
            return future.result()
        return {'heva_20': 0.0}

    def shutdown_heva_executor(self):
        """关闭 HEVA 计算线程池"""
        self._heva_executor.shutdown(wait=True)

    def get_visual_token_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        获取视觉 token 的索引

        在 Qwen3-VL 中:
        - image tokens 使用特殊的 token_id 表示
        - 需要根据 token_id 范围来确定视觉 token 位置

        Returns:
            视觉 token 索引的 tensor
        """
        # Qwen3-VL 的 image token ID 范围
        # 查看 tokenizer 中的特殊 token
        # 这里我们通过检查 input_ids 中是否有大量连续的 image token 来判断

        # 获取 image token 的 token_id
        # 在 Qwen3-VL 中，图像 token 通常在特定范围内
        # 这里使用一个更通用的方法：通过 processor 获取 image token

        # 尝试从 processor 获取 image token id
        try:
            # Qwen3-VL 使用的 image token
            image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index('<|image_pad|>')
            ]
        except (ValueError, KeyError):
            # 如果找不到，使用默认值
            # Qwen3-VL-2B 通常使用 151643 作为 image pad
            image_token_id = 151643

        # 找出所有 image token 的位置
        visual_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

        return visual_indices

    def get_gen_token_indices(self, input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
        """
        获取生成 token 的索引

        Args:
            input_ids: 输入的 token IDs
            prompt_length: prompt 的长度（不包括图像 token）

        Returns:
            生成 token 索引
        """
        # 生成 token 从 prompt_length 开始
        gen_indices = torch.arange(prompt_length, len(input_ids))
        return gen_indices

    def generate_with_attention(
        self,
        image: Any,
        question: str,
        options: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        async_heva: bool = False,
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        带 attention 的生成

        Args:
            image: PIL Image
            question: 问题文本
            options: 选项文本
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: top-p 采样
            top_k: top-k 采样
            do_sample: 是否采样
            (image_size 使用模型默认值)

        Returns:
            {
                "generated_text": str,
                "generated_ids": torch.Tensor,
                "logits": torch.Tensor,  # (seq_len, vocab_size)
                "attentions": Tuple,    # (num_layers, batch, num_heads, seq_len, seq_len)
                "input_ids": torch.Tensor,
                "visual_token_indices": torch.Tensor,
                "gen_token_indices": torch.Tensor,
                "prompt": str,  # 完整的prompt
            }
        """
        # 构建prompt，参考用户提供的格式
        option_str = f"option: {options}\n" if options else ""
        full_question = question + option_str + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'

        # 准备消息 (包含system message)
        messages = [
            {"role": "system", "content": "You are good at step by step reasoning."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": full_question}
                ]
            }
        ]

        # 应用 chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 处理输入
        # 先不使用size参数，让processor自动处理图像
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # 移动到设备
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)
        # 获取 image_grid_thw 用于视觉编码
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        # 获取视觉 token 索引
        # 在 Qwen3-VL 中，图像 token 在 input_ids 中是连续的
        visual_token_indices = self.get_visual_token_indices(input_ids[0])

        # 计算 prompt 长度
        # 找到最后一个非图像 token 的位置，然后找到 user message 结束位置
        # Qwen3-VL 使用 <|im_end|> 作为消息结束符
        if len(visual_token_indices) > 0:
            # 找到图像 token 结束后的位置
            img_end = int(visual_token_indices[-1].item()) + 1
            # 从这里继续找 <|im_end|> 标记
            input_ids_list = input_ids[0].tolist()
            try:
                # 找到 <|im_end|> (151645) 的位置
                prompt_length = input_ids_list.index(151645, img_end) + 1
            except ValueError:
                # 如果找不到，使用图像 token 结束位置
                prompt_length = img_end
        else:
            prompt_length = len(input_ids[0])

        # 生成 (先生成文本，使用SDPA加速，不输出attention)
        with torch.no_grad():
            # 临时切换到SDPA加速生成
            original_attn = self.model.config._attn_implementation
            self.model.config._attn_implementation = "sdpa"

            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_attentions=False,  # 先生成文本，不输出attention节省显存
                output_scores=True,     # 输出 logits
            )

            # 恢复eager attention（用于后续HEVA计算）
            self.model.config._attn_implementation = original_attn

        # 提取结果
        generated_ids = output.sequences[0]
        gen_ids = generated_ids[prompt_length:]

        generated_text = self.processor.batch_decode(
            gen_ids.unsqueeze(0) if gen_ids.dim() == 1 else gen_ids,
            skip_special_tokens=False
        )[0]
        generated_text = generated_text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

        # 提取 logits (只需要用于返回)
        logits = None
        if output.scores is not None and len(output.scores) > 0:
            logits = torch.cat([output.scores[i] for i in range(len(output.scores))], dim=0)

        # 设置 attentions 为 None (因为不在 generate 时输出)
        attentions = None

        # ========== 先生成文本，后计算HEVA ==========
        # 第一步：只生成文本（已完成）
        # 第二步：用完整序列做前向传播获取attention来计算HEVA
        full_attentions = None
        full_logits = None
        try:
            full_input_ids = generated_ids.unsqueeze(0)  # (1, full_seq_len)
            with torch.no_grad():
                full_outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=torch.ones_like(full_input_ids),
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_attentions=True,
                    return_dict=True,
                )
            full_attentions = full_outputs.attentions  # Tuple of (1, num_heads, full_seq_len, full_seq_len)
            # 只保留最后一层，减少显存占用
            full_attentions = (full_attentions[-1],)
            full_logits = full_outputs.logits.squeeze(0).detach()  # (full_seq_len, vocab_size)

        except Exception as e:
            import traceback
            print(f"ERROR: Failed to get full sequence attention: {e}")
            traceback.print_exc()

        # 计算 HEVA (使用完整序列的 logits 和 attention)
        heva_value = 0.0
        if full_logits is not None and full_attentions is not None:
            # 异步模式下，始终由后台线程处理HEVA计算和设备转移
            if async_heva:
                result_for_heva = {
                    'logits': full_logits,
                    'attentions': full_attentions,
                    'visual_token_indices': visual_token_indices,
                    'prompt_length': prompt_length,
                    'generated_ids': generated_ids,
                    'heva_device': self.heva_device,  # 传递目标设备
                }
                # 启动异步计算
                self.compute_heva_async(result_for_heva, sample_idx)
                # 立即返回，HEVA 值稍后获取
            else:
                # 同步模式
                heva_on_same_device = (str(full_logits.device) == self.heva_device)
                # 同步模式（原有逻辑）
                try:
                    # 如果 HEVA 使用不同设备，需要将数据转移到该设备
                    if not heva_on_same_device:
                        full_logits_heva = full_logits.to(self.heva_device)
                        # 只保留最后一层attention，大幅减少显存使用
                        full_attentions_heva = tuple([attentions[-1].to(self.heva_device)])
                        visual_token_indices_heva = visual_token_indices.to(self.heva_device)
                        generated_ids_heva = generated_ids.to(self.heva_device)
                        # 释放原始设备上的内存
                        del full_attentions, full_logits
                        if self.device != "cpu":
                            if self.device == "cuda":
                                torch.cuda.empty_cache()
                            else:
                                try:
                                    import torch_npu
                                    torch_npu.npu.empty_cache()
                                except:
                                    pass
                    else:
                        full_logits_heva = full_logits
                        full_attentions_heva = full_attentions
                        visual_token_indices_heva = visual_token_indices
                        generated_ids_heva = generated_ids

                    # 延迟导入避免循环依赖
                    import sys as _sys
                    if 'metrics.heva' in _sys.modules:
                        from metrics.heva import compute_heva_from_result
                    else:
                        from metrics.heva import compute_heva_from_result

                    # 使用完整序列的数据计算 HEVA
                    heva_result = compute_heva_from_result({
                        "logits": full_logits_heva,
                        "attentions": full_attentions_heva,
                        "visual_token_indices": visual_token_indices_heva,
                        "prompt_length": prompt_length,
                        "input_ids": generated_ids_heva,
                    }, alpha=0.2)
                    heva_value = heva_result['heva']

                    # HEVA 计算完成后释放内存
                    del full_logits_heva, full_attentions_heva
                    if self.heva_device != "cpu":
                        if self.heva_device == "cuda":
                            torch.cuda.empty_cache()
                        else:
                            try:
                                import torch_npu
                                torch_npu.npu.empty_cache()
                            except:
                                pass
                except Exception as e:
                    import traceback
                    print(f"ERROR: Failed to compute HEVA: {e}")
                    traceback.print_exc()
        else:
            print(f"WARNING: full_logits or full_attentions is None, skipping HEVA computation")

        return {
            "generated_text": generated_text,
            "generated_ids": generated_ids,
            "logits": logits,  # 已经是 (seq_len, vocab_size) 格式
            "attentions": attentions,
            "input_ids": input_ids[0],
            "generated_ids": generated_ids,  # 完整的生成序列
            "visual_token_indices": visual_token_indices,
            "prompt_length": prompt_length,
            "heva": heva_value,
            "prompt": text,  # 完整的prompt字符串
            "raw_question": full_question,  # 原始问题+选项+格式要求
        }

    def forward_with_attention(
        self,
        image: Any,
        question: str,
    ) -> Dict[str, Any]:
        """
        前向传播，获取中间表示

        Args:
            image: PIL Image
            question: 问题文本

        Returns:
            包含 logits 和 attentions 的字典
        """
        # 准备消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        # 获取视觉 token 索引
        visual_token_indices = self.get_visual_token_indices(input_ids[0])

        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
            )

        return {
            "logits": outputs.logits,  # (batch, seq_len, vocab_size)
            "attentions": outputs.attentions,  # Tuple of (batch, num_heads, seq_len, seq_len)
            "input_ids": input_ids[0],
            "visual_token_indices": visual_token_indices,
        }


# 全局模型缓存，避免每次都重新加载
_model_cache = {}


def load_model(model_path: str = None, device: str = None, num_gpus: int = 1, heva_device: str = None, reuse: bool = True) -> Qwen3VLInference:
    """加载模型的便捷函数，支持缓存复用和多卡"""
    import config
    if model_path is None:
        model_path = config.MODEL_DIR

    # 如果启用缓存且模型已加载，直接返回缓存的模型
    # 注意：如果 heva_device 改变，需要重新加载
    cache_key = f"{model_path}_{device}_{num_gpus}_{heva_device}"
    if reuse and cache_key in _model_cache:
        log_print(f"Reusing cached model: {model_path}")
        return _model_cache[cache_key]

    # 加载新模型
    model = Qwen3VLInference(model_path, device, num_gpus, heva_device)

    # 缓存模型
    if reuse:
        _model_cache[cache_key] = model
        log_print(f"Model cached: {model_path}")

    return model


if __name__ == "__main__":
    # 测试
    from data.loader import load_dataset

    dataset = load_dataset()
    sample = dataset[0]

    log_print("Loading model...")
    model = load_model()

    log_print("Running inference...")
    result = model.generate_with_attention(
        image=sample['image'],
        question=sample['question'] + "\n" + sample['options']
    )

    log_print(f"Generated text: {result['generated_text'][:200]}...")
    log_print(f"Visual token indices: {result['visual_token_indices']}")
    log_print(f"Prompt length: {result['prompt_length']}")
    log_print(f"Attention layers: {len(result['attentions'])}")
