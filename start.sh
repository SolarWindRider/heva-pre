#!/bin/bash
set -e

# 默认参数
EXP_NAME="exp001"
DATASET="VisuRiddles"
NUM_SAMPLES=-1
SHUFFLE=false
BATCH_SIZE=1
SEED=42
TEMPERATURE=0.7
MAX_NEW_TOKENS=128
TOP_P=0.9
TOP_K=50
MODEL_PATH=""
NUM_GPUS=8
HEVA_DEVICE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -n|--num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -s|--shuffle)
            SHUFFLE="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -r|--seed)
            SEED="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -m|--max_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        -p|--top_p)
            TOP_P="$2"
            shift 2
            ;;
        -k|--top_k)
            TOP_K="$2"
            shift 2
            ;;
        -o|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -g|--num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --heva_device)
            HEVA_DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "HEVA 实验启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -e, --exp_name      实验名称 (默认: exp001)"
            echo "  -d, --dataset       数据集名称 (默认: VisuRiddles, 可用逗号分隔多个)"
            echo "  -n, --num_samples   样本数量 (默认: -1, 全部样本)"
            echo "  -s, --shuffle       是否打乱数据集 (默认: false)"
            echo "  -b, --batch_size    batch size (默认: 1)"
            echo "  -r, --seed         随机种子 (默认: 42)"
            echo "  -t, --temperature   温度 (默认: 0.7)"
            echo "  -m, --max_tokens    最大生成长度 (默认: 128)"
            echo "  -p, --top_p         top-p 采样 (默认: 0.9)"
            echo "  -k, --top_k         top-k 采样 (默认: 50)"
            echo "  -o, --model_path    模型路径 (默认: config.py 中的路径)"
            echo "  -g, --num_gpus     推理加速卡数量 (默认: 1)"
            echo "  --heva_device       HEVA计算设备 (默认: 与模型相同)"
            echo "  -h, --help          显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                          # 使用默认参数运行"
            echo "  $0 -d VisuRiddles -n 10    # 处理 VisuRiddles 前10个样本"
            echo "  $0 -e exp002 -d RAVEN -n 100 -t 0.5  # 自定义实验"
            echo "  $0 --heva_device npu:1     # 使用npu:1计算HEVA"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 打印配置
echo "=========================================="
echo "  HEVA 实验启动"
echo "=========================================="
echo "实验名称:    $EXP_NAME"
echo "数据集:      $DATASET"
echo "样本数量:    $NUM_SAMPLES"
echo "打乱数据:    $SHUFFLE"
echo "Batch size:  $BATCH_SIZE"
echo "随机种子:    $SEED"
echo "温度:        $TEMPERATURE"
echo "最大长度:    $MAX_NEW_TOKENS"
echo "top-p:       $TOP_P"
echo "top-k:       $TOP_K"
echo "推理加速卡数量:    $NUM_GPUS"
echo "HEVA设备:   $HEVA_DEVICE"
echo "输出目录:    results/$EXP_NAME/<dataset>"
echo "=========================================="
echo ""

# 分割数据集 (逗号分隔)
IFS=',' read -ra DATASETS <<< "$DATASET"

# 运行推理
for ds in "${DATASETS[@]}"; do
    ds=$(echo "$ds" | xargs)  # 去除空格
    echo "=========================================="
    echo "  Running dataset: $ds"
    echo "=========================================="

    PYTHON_CMD="python experiments/1_run_inference.py \
        --exp_name \"$EXP_NAME\" \
        --dataset \"$ds\" \
        --num_samples \"$NUM_SAMPLES\" \
        --shuffle \"$SHUFFLE\" \
        --batch_size \"$BATCH_SIZE\" \
        --seed \"$SEED\" \
        --temperature \"$TEMPERATURE\" \
        --max_new_tokens \"$MAX_NEW_TOKENS\" \
        --top_p \"$TOP_P\" \
        --top_k \"$TOP_K\" \
        --num_gpus \"$NUM_GPUS\""

    # 如果指定了模型路径，添加到命令中
    if [ -n "$MODEL_PATH" ]; then
        PYTHON_CMD="$PYTHON_CMD --model_path \"$MODEL_PATH\""
    fi

    # 如果指定了HEVA计算设备，添加到命令中
    if [ -n "$HEVA_DEVICE" ]; then
        PYTHON_CMD="$PYTHON_CMD --heva_device \"$HEVA_DEVICE\""
    fi

    eval $PYTHON_CMD

    echo ""
done

echo "实验完成! 结果保存在 results/$EXP_NAME/"
