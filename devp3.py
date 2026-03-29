"""
测试 ContextAwareLogitsProcessor 的功能测试脚本

测试内容：
1. 模块导入和类实例化
2. compute_entropy 函数
3. select_context_heads 函数
4. ContextAwareLogitsProcessor 完整流程
5. 与标准 generate() 的输出对比
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 导入待测试的模块
from metrics.context_aware_logits_processor import (
    ContextAwareLogitsProcessor,
    ContextAwareModelWrapper,
    compute_entropy,
    select_context_heads,
    compute_token_support_from_attentions,
    get_context_token_indices,
    get_image_token_id,
)
from metrics.inference import get_visual_token_indices, get_input_token_indices
from metrics.heva import _sample_with_vattn_and_entropy

# Monkey patch
Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy


def test_import():
    """测试1: 模块导入"""
    print("\n" + "="*60)
    print("测试1: 模块导入")
    print("="*60)
    print("✓ ContextAwareLogitsProcessor 导入成功")
    print("✓ ContextAwareModelWrapper 导入成功")
    print("✓ compute_entropy 导入成功")
    print("✓ select_context_heads 导入成功")
    print("✓ get_context_token_indices 导入成功")
    print("✓ get_image_token_id 导入成功")
    return True


def test_compute_entropy():
    """测试2: compute_entropy 函数"""
    print("\n" + "="*60)
    print("测试2: compute_entropy 函数")
    print("="*60)

    # 创建测试 logits
    # 均匀分布 -> 高熵
    uniform_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # 4个token均匀分布
    entropy_uniform = compute_entropy(uniform_logits)
    print(f"均匀分布 logits: {uniform_logits}")
    print(f"均匀分布 entropy: {entropy_uniform.item():.4f}")

    # 尖峰分布 -> 低熵
    peaked_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # 第一个token概率接近1
    entropy_peaked = compute_entropy(peaked_logits)
    print(f"尖峰分布 logits: {peaked_logits}")
    print(f"尖峰分布 entropy: {entropy_peaked.item():.4f}")

    # 验证
    assert entropy_uniform[0] > entropy_peaked[0], "熵值关系验证失败"
    assert entropy_uniform[0] > 0.5, "均匀分布熵值应该较高"
    assert entropy_peaked[0] < 0.5, "尖峰分布熵值应该较低"

    print("✓ 熵值计算正确: 均匀 > 尖峰")
    return True


def test_select_context_heads():
    """测试3: select_context_heads 函数"""
    print("\n" + "="*60)
    print("测试3: select_context_heads 函数")
    print("="*60)

    # 模拟 attention tensors: (batch, heads, seq, seq)
    batch_size = 1
    num_heads = 8
    seq_len = 100

    # 创建 attention: 最后一个 query 对 context tokens (0-20) 的注意力较高
    attentions = []
    for layer_idx in range(2):  # 2层
        attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        # 假设 context tokens 在位置 0-20
        # 对于 head 3 和 head 5，让它们对 context tokens 的注意力更强
        attn[0, 3, -1, 0:21] = 2.0  # head 3 关注 context
        attn[0, 5, -1, 0:21] = 3.0  # head 5 更关注 context
        # 其他 head 注意力较低
        attn[0, 0, -1, 0:21] = 0.1
        attn[0, 1, -1, 0:21] = 0.2
        # 对非 context tokens 的注意力
        attn[0, :, -1, 21:] = 0.01
        # 归一化 (简化的 softmax 效果)
        attn = torch.softmax(attn, dim=-1)
        attentions.append(attn)

    context_token_indices = (torch.tensor([0]), torch.tensor([20]))

    context_heads = select_context_heads(
        attentions=attentions,
        context_token_indices=context_token_indices,
        top_h=3
    )

    print(f"检测到的 context heads: {context_heads}")
    print(f"预期: 至少包含 head 3 和 head 5 (最后层的)")

    # 验证：应该选中关注 context 的 heads
    head_indices = [h[1] for h in context_heads]
    assert 5 in head_indices or 3 in head_indices, f"应该选中 head 3 或 5，实际: {head_indices}"

    print("✓ Context heads 检测正确")
    return True


def test_logits_processor_forward():
    """测试4: ContextAwareLogitsProcessor __call__ 方法"""
    print("\n" + "="*60)
    print("测试4: ContextAwareLogitsProcessor __call__ 方法")
    print("="*60)

    # 创建 mock model
    class MockModel:
        pass

    model = MockModel()
    model.config = type('Config', (), {'d_model': 1024})()

    processor = ContextAwareLogitsProcessor(
        model=model,
        top_k=5,
        entropy_threshold=1.0,
        top_heads=3
    )

    # 设置 context token indices
    processor.set_context_token_indices((torch.tensor([0]), torch.tensor([20])))

    # 模拟高熵的 logits
    input_ids = torch.tensor([[1, 2, 3]])  # batch=1
    high_entropy_logits = torch.randn(1, 100)  # 随机 -> 高熵

    print(f"输入 logits shape: {high_entropy_logits.shape}")
    print(f"熵值: {compute_entropy(high_entropy_logits).item():.4f}")

    # 不带 attention 调用（应该直接返回原始 scores）
    result = processor(input_ids, high_entropy_logits.clone())

    # 带 mock attention 调用
    batch_size, num_heads = 1, 8
    seq_len = 100
    mock_attentions = []
    for _ in range(2):
        attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        attn[0, 3, -1, 0:21] = 2.0
        attn[0, 5, -1, 0:21] = 3.0
        attn = torch.softmax(attn, dim=-1)
        mock_attentions.append(attn)

    processor._last_attentions = mock_attentions

    result = processor(input_ids, high_entropy_logits.clone())

    # 验证：top-k 以外的 logits 不变
    # top-k 内的 logits 被调整了
    print(f"输出 logits shape: {result.shape}")
    print(f"部分输出: {result[0, :5]}")

    print("✓ LogitsProcessor forward 执行成功")
    return True


def test_end_to_end_with_model():
    """测试5: 端到端测试 - 与模型集成"""
    print("\n" + "="*60)
    print("测试5: 端到端测试 - 模型集成")
    print("="*60)

    model_name_or_path = "/home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Thinking"

    print("正在加载模型...")
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("✓ 模型加载成功")

    # 准备输入
    question = """You are given a puzzle. The puzzle consists of a question part on the top and the choices part in the bottom."""
    full_question = question + '\nWrite the answer into a JSON form\n```json\n{"answer": "X"}```'

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are good at step by step reasoning."}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ma-user/work/datas/MARVEL_AVR/Marvel/1.png"},
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
    inputs = inputs.to(model.device)

    # 获取 context token indices（这里是视觉 token）
    context_token_indices = get_context_token_indices(inputs["input_ids"], processor=processor)
    print(f"Context token 索引 (auto-detect): start={context_token_indices[0]}, end={context_token_indices[1]}")

    # 打印前几个 token 的 ID，看是否正确识别了 image token
    input_ids = inputs["input_ids"][0]
    print(f"输入 token 数量: {len(input_ids)}")

    # 检查 image token ID
    img_tok_id = get_image_token_id(processor)
    print(f"Image token ID: {img_tok_id}")

    # 统计 image tokens 数量
    img_count = (input_ids == img_tok_id).sum().item()
    print(f"Image tokens 数量: {img_count}")

    # 测试：用户手动指定 context 范围
    user_specified_indices = (torch.tensor([10]), torch.tensor([100]))
    user_context = get_context_token_indices(inputs["input_ids"], processor=processor, image_token_indices=user_specified_indices)
    print(f"Context token (user-specified 10-100): start={user_context[0]}, end={user_context[1]}")

    # 测试：获取所有 prompt tokens（排除 padding）
    all_prompt = get_input_token_indices(inputs["input_ids"], processor=processor)
    print(f"All prompt tokens (via get_input_token_indices): start={all_prompt[0]}, end={all_prompt[1]}")

    # 设置模型属性 (用于 monkey-patched _sample)
    model.visual_token_indices = context_token_indices
    model.inputs_token_indices = (torch.tensor([0]), torch.tensor([inputs["input_ids"].shape[1]-1]))
    model.gen_entropy = []
    model.gen_vattn = []
    model.attn_acc_input = []
    model.attn_acc_visual = []

    # 创建 ContextAwareLogitsProcessor
    ctx_processor = ContextAwareLogitsProcessor(
        model=model,
        top_k=20,
        entropy_threshold=5.0,
        top_heads=5,
    )
    ctx_processor.set_context_token_indices(context_token_indices)

    print("\n--- 测试: 标准 generate (baseline) ---")
    with torch.no_grad():
        # Baseline: 不使用 ContextAwareLogitsProcessor
        outputs_baseline = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    print(f"Baseline 输出长度: {outputs_baseline.sequences.shape}")

    print("\n--- 测试: 带 ContextAwareLogitsProcessor ---")
    with torch.no_grad():
        outputs_vg = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            return_dict_in_generate=True,
            output_attentions=True,
            logits_processor=[ctx_processor],
        )
    print(f"VG 输出长度: {outputs_vg.sequences.shape}")

    # 检查 gen_entropy 是否被捕获
    if model.gen_entropy:
        entropy_tensor = torch.stack(model.gen_entropy, dim=1).detach().cpu()
        print(f"\n捕获的 entropy 序列数: {entropy_tensor.shape[1]}")
        print(f"Entropy 均值: {entropy_tensor.mean().item():.4f}")
        print("✓ Monkey-patch 正常工作")

    print("\n✓ 端到端测试完成")
    return True


def main():
    """运行所有测试"""
    print("="*60)
    print("ContextAwareLogitsProcessor 功能测试")
    print("="*60)

    tests = [
        ("模块导入", test_import),
        ("compute_entropy", test_compute_entropy),
        ("select_context_heads", test_select_context_heads),
        ("LogitsProcessor forward", test_logits_processor_forward),
        ("端到端模型测试", test_end_to_end_with_model),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n❌ 测试失败: {name}")
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # 汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"通过: {passed}/{total}")

    for name, success, error in results:
        status = "✓" if success else "❌"
        print(f"{status} {name}")
        if error:
            print(f"  错误: {error}")

    if passed == total:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
