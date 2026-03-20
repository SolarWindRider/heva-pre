from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_name_or_path = "/home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Thinking"
processor = AutoProcessor.from_pretrained(model_name_or_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name_or_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
)

question = """You are given a puzzle. The puzzle consists of a question part on the top and the choices part in the bottom. The question part on the top is a set of visual panels arranged in a 1 by 5 sequence, with the last piece missing. Choices part on the bottom contains 4 choices (marked by 1, 2, 3, or 4). Which choice (either 1, 2, 3, or 4) is the most appropriate answer to fill the missing part?"""
full_question = question + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are good at step by step reasoning."}]},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/ma-user/work/datas/MARVEL_AVR/Marvel/1.png",  # "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": full_question},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs.pop("token_type_ids", None)

outputs = model.generate(
    **inputs,
    max_new_tokens=17,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    return_dict_in_generate=True,
    output_attentions=True,
    output_logits=True,
)


print(outputs)
# outputs.attentions (gen_tokens_num, model_layers_sum, [tensor_shape(batch_size, head_num, Query Length, Key Length)])
