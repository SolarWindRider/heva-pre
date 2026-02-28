#!/bin/bash
# 运行所有数据集的快速测试脚本
# 每个数据集只跑 1 个样本

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  快速测试: 每个数据集跑 100 个样本"
echo "=========================================="
echo ""

# 所有支持的数据集
DATASETS="VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA"

# 调用 start.sh
./start.sh --exp_name exp001 -d "$DATASETS" -n 100 -b 8 -s true -m 8192 -g 1

echo ""
echo "所有数据集快速测试完成!"
