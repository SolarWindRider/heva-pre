"""
HEVA 模型推理模块
支持 Qwen3-VL 推理并捕获 attention 和 logits
支持 NVIDIA GPU 和华为 NPU
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, Any, Tuple, Optional
import torch.nn.functional as F

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

    def __init__(self, model_path: str, device: str = None, num_gpus: int = 1):
        """
        Args:
            model_path: 模型路径
            device: 设备，默认为自动检测
            num_gpus: 使用的GPU数量，默认为1
        """
        self.device = device or get_device()
        self.model_path = model_path
        self.num_gpus = num_gpus

        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        print(f"Number of GPUs: {num_gpus}")

        # 根据设备类型和GPU数量设置 device_map
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

        print(f"Model loaded on {self.device}")

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
        image_size: int = 448,
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
            image_size: 图像处理大小

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

        # 生成
        with torch.no_grad():
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
            )

        # 提取结果
        # output.sequences 包含输入 + 输出
        # 使用 prompt_length 而不是 input_len 来切片，因为 prompt_length 是用户消息结束的位置
        generated_ids = output.sequences[0]
        gen_ids = generated_ids[prompt_length:]  # 从用户消息结束后开始取

        # 先不用 skip_special_tokens，获取完整输出
        generated_text = self.processor.batch_decode(
            gen_ids.unsqueeze(0) if gen_ids.dim() == 1 else gen_ids,
            skip_special_tokens=False
        )[0]
        # 只清理特殊token，保留markdown格式
        generated_text = generated_text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

        # 使用 forward_with_attention 获取 attention 和 logits
        # 将生成的 token 加入输入，重新前向传播获取 attention
        full_input_ids = generated_ids.unsqueeze(0)
        full_attention_mask = torch.ones_like(full_input_ids)

        # 获取 attention 和 logits
        attentions = None
        logits = None

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_attentions=True,
                    return_dict=True,
                )
            attentions = outputs.attentions
            logits = outputs.logits
        except Exception as e:
            print(f"Warning: Failed to get attentions: {e}")

        # 计算 HEVA
        heva_value = 0.0
        if logits is not None and attentions is not None:
            try:
                # 延迟导入避免循环依赖
                import sys
                if 'metrics.heva' in sys.modules:
                    from metrics.heva import compute_heva_from_result
                else:
                    from metrics.heva import compute_heva_from_result

                heva_result = compute_heva_from_result({
                    "logits": logits,
                    "attentions": attentions,
                    "visual_token_indices": visual_token_indices,
                    "prompt_length": prompt_length,
                    "input_ids": generated_ids,  # 使用完整的生成序列
                }, alpha=0.2)
                heva_value = heva_result['heva']
            except Exception as e:
                print(f"Warning: Failed to compute HEVA: {e}")

        return {
            "generated_text": generated_text,
            "generated_ids": generated_ids,
            "logits": logits[0] if logits is not None else None,
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


def load_model(model_path: str = None, device: str = None, num_gpus: int = 1, reuse: bool = True) -> Qwen3VLInference:
    """加载模型的便捷函数，支持缓存复用和多卡"""
    import config
    if model_path is None:
        model_path = config.MODEL_DIR

    # 如果启用缓存且模型已加载，直接返回缓存的模型
    cache_key = f"{model_path}_{device}_{num_gpus}"
    if reuse and cache_key in _model_cache:
        print(f"Reusing cached model: {model_path}")
        return _model_cache[cache_key]

    # 加载新模型
    model = Qwen3VLInference(model_path, device, num_gpus)

    # 缓存模型
    if reuse:
        _model_cache[cache_key] = model
        print(f"Model cached: {model_path}")

    return model


if __name__ == "__main__":
    # 测试
    from data.loader import load_dataset

    dataset = load_dataset()
    sample = dataset[0]

    print("Loading model...")
    model = load_model()

    print("Running inference...")
    result = model.generate_with_attention(
        image=sample['image'],
        question=sample['question'] + "\n" + sample['options']
    )

    print(f"Generated text: {result['generated_text'][:200]}...")
    print(f"Visual token indices: {result['visual_token_indices']}")
    print(f"Prompt length: {result['prompt_length']}")
    print(f"Attention layers: {len(result['attentions'])}")
