#!/bin/bash
# Download Qwen3-VL official benchmarks for HEVA project
# Run from project root: bash download_benchmarks.sh

set -e

DATAS=/data2/lixiang/datas
mkdir -p $DATAS/AI2D $DATAS/RealWorldQA $DATAS/VQAv2 $DATAS/GQA $DATAS/MMMU

# Remove existing dirs to re-download cleanly
rm -rf $DATAS/AI2D $DATAS/VQAv2 $DATAS/GQA

source /data2/lixiang/miniconda3/etc/profile.d/conda.sh
conda activate r1

# Use hf-mirror.com to avoid proxy SSL issues
export HF_ENDPOINT=https://hf-mirror.com

echo "=== 1/5: AI2D ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='AI2-MLE/AI2D', repo_type='dataset', local_dir='/data2/lixiang/datas/AI2D', local_dir_use_symlinks=False)
"

echo "=== 2/5: RealWorldQA ==="
wget -q https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv -O $DATAS/RealWorldQA/data.tsv
echo "RealWorldQA images: download manually from OpenCompass"

echo "=== 3/5: VQAv2 ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='HuggingFaceM4/VQAv2', repo_type='dataset', local_dir='/data2/lixiang/datas/VQAv2', local_dir_use_symlinks=False)
"

echo "=== 4/5: GQA ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='GQA', repo_type='dataset', local_dir='/data2/lixiang/datas/GQA', local_dir_use_symlinks=False)
"

echo "=== 5/5: MMMU ==="
wget -q https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv -O $DATAS/MMMU/data.tsv
echo "MMMU images: download manually from OpenCompass"

echo ""
echo "=== Done ==="
echo "Verify with: python -c 'from data.loader import load_dataset; print(load_dataset(\"AI2D\"))'"
