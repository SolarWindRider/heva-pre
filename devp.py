from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from metrics.heva import _sample_with_vattn_and_entropy

Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy  # 替换原有的 _sample 方法，以便在生成过程中获取注意力权重和熵值


def get_visual_token_indices(input_ids: torch.Tensor, processor: AutoProcessor) -> torch.Tensor:
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
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<|image_pad|>")
        ]
    except (ValueError, KeyError):
        # 如果找不到，使用默认值
        # Qwen3-VL-2B 通常使用 151643 作为 image pad
        image_token_id = 151643

    # 找出所有 image token 的位置
    visual_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    visual_indices = visual_indices[0], visual_indices[-1]
    return visual_indices


model_name_or_path = "/home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Thinking"
processor = AutoProcessor.from_pretrained(model_name_or_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name_or_path, dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
)
print(model)

# question = """You are given a puzzle. The puzzle consists of a question part on the top and the choices part in the bottom. The question part on the top is a set of visual panels arranged in a 1 by 5 sequence, with the last piece missing. Choices part on the bottom contains 4 choices (marked by 1, 2, 3, or 4). Which choice (either 1, 2, 3, or 4) is the most appropriate answer to fill the missing part?"""
# full_question = question + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'

# messages = [
#     {"role": "system", "content": [{"type": "text", "text": "You are good at step by step reasoning."}]},
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "/home/ma-user/work/datas/MARVEL_AVR/Marvel/1.png",  # "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": full_question},
#         ],
#     },
# ]

# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt",
# )
# inputs.pop("token_type_ids", None)
# inputs = inputs.to(model.device)
# prompt_token_num = inputs["input_ids"].shape[1]

# # 获取视觉 token 索引
# visual_token_indices = get_visual_token_indices(inputs["input_ids"][0], processor=processor)
# model.visual_token_indices = visual_token_indices
# model.gen_entropy = []
# model.gen_vattn = []
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=17,
#     temperature=0.7,
#     top_p=0.9,
#     top_k=50,
#     do_sample=True,
#     return_dict_in_generate=True,
#     output_attentions=True,
#     # output_logits=True,
# )
# model.gen_entropy = torch.stack(model.gen_entropy, dim=1).detach().cpu()  # (gen_tokens_num, batch_size)
# model.gen_vattn = torch.stack(model.gen_vattn, dim=1).detach().cpu()  # (gen_tokens_num, batch_size, visual_token_num)

# print(outputs)
# # outputs.attentions (gen_tokens_num, model_layers_sum, [tensor_shape(batch_size, head_num, Query Length, Key Length)])
