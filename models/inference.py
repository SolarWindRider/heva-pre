"""
HEVA 模型推理模块
支持 Qwen3-VL 推理并捕获 attention 和 logits
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
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
        # 强制使用 cuda/npu，因为模型在 GPU 上
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model_path = model_path

        print(f"Loading model from {model_path}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        # 设置使用 eager attention 以支持 attention 输出
        if hasattr(self.model, 'config'):
            self.model.config.attn_implementation = "eager"
        if hasattr(self.model, 'set_attn_implementation'):
            self.model.set_attn_implementation("eager")
        self.model = self.model.to(self.device)

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

        # 生成 - 先不使用 output_attentions，加快速度
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
            )

        # 提取结果
        # output.sequences 包含输入 + 输出
        # 输出部分从 len(input_ids) 开始
        generated_ids = output.sequences[0]
        input_len = len(input_ids[0])
        gen_ids = generated_ids[input_len:]  # 只取新生成的 token

        generated_text = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True
        )[0]

        # 使用 forward_with_attention 获取 attention 和 logits
        # 将生成的 token 加入输入，重新前向传播获取 attention
        full_input_ids = generated_ids.unsqueeze(0)
        full_attention_mask = torch.ones_like(full_input_ids)

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
            attentions = None
            logits = None

        return {
            "generated_text": generated_text,
            "generated_ids": generated_ids,
            "logits": logits[0] if logits is not None else None,
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
