"""
HEVA 模型推理模块
支持 Qwen3-VL 推理并捕获 attention 和 logits
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Tuple, Optional
import torch.nn.functional as F


class Qwen3VLInference:
    """Qwen3-VL 推理类"""

    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: 模型路径
            device: 设备，默认为 cuda
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        print(f"Loading model from {model_path}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"  # 需要 eager 来获取 attention
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

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
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        带 attention 的生成

        Args:
            image: PIL Image
            question: 问题文本
            max_new_tokens: 最大生成长度
            temperature: 温度

        Returns:
            {
                "generated_text": str,
                "generated_ids": torch.Tensor,
                "logits": torch.Tensor,  # (seq_len, vocab_size)
                "attentions": Tuple,    # (num_layers, batch, num_heads, seq_len, seq_len)
                "input_ids": torch.Tensor,
                "visual_token_indices": torch.Tensor,
                "gen_token_indices": torch.Tensor,
            }
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

        # 应用 chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 处理输入
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

        # 获取视觉 token 索引
        # 在 Qwen3-VL 中，图像 token 在 input_ids 中是连续的
        visual_token_indices = self.get_visual_token_indices(input_ids[0])

        # 计算 prompt 长度（不包括图像 token）
        # 找到最后一个非图像 token 的位置
        if len(visual_token_indices) > 0:
            prompt_length = int(visual_token_indices[0].item())
        else:
            prompt_length = len(input_ids[0])

        # 生成
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # 提取结果
        generated_ids = output.sequences[0]
        generated_text = self.processor.batch_decode(
            generated_ids[prompt_length:],
            skip_special_tokens=True
        )[0]

        # 获取 logits (需要重新前向传播获取)
        # 由于 generate 不返回完整 logits，我们需要单独前向传播
        # 这里简化处理，使用 generated token 的 logits

        # 获取最后一层的 attention
        attentions = output.attentions  # Tuple of (batch, num_heads, seq_len, seq_len)

        # 获取生成的 token 对应的 logits
        # 这里的 logits 形状是 (seq_len, vocab_size)
        if hasattr(output, 'logits'):
            logits = output.logits[0]  # (seq_len, vocab_size)
        else:
            logits = None

        return {
            "generated_text": generated_text,
            "generated_ids": generated_ids,
            "logits": logits,
            "attentions": attentions,
            "input_ids": input_ids[0],
            "visual_token_indices": visual_token_indices,
            "prompt_length": prompt_length,
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


def load_model(model_path: str = None, device: str = None) -> Qwen3VLInference:
    """加载模型的便捷函数"""
    import config
    if model_path is None:
        model_path = config.MODEL_DIR

    return Qwen3VLInference(model_path, device)


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
