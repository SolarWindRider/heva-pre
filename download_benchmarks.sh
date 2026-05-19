#!/bin/bash
set -e

DATAS=/w0rk5pace/aaworks/datas
mkdir -p $DATAS/AI2D $DATAS/RealWorldQA $DATAS/GQA $DATAS/MMMU

source /opt/conda/etc/profile.d/conda.sh
conda activate heva

echo "=== 1/4: AI2D (lmms-lab/ai2d, test only) ==="
python -c "
from datasets import load_dataset
load_dataset('lmms-lab/ai2d', split='test', cache_dir='/w0rk5pace/aaworks/datas/AI2D')
"

echo "=== 2/4: RealWorldQA (xai-org/RealworldQA, test only) ==="
python -c "
from datasets import load_dataset
load_dataset('xai-org/RealworldQA', split='test', cache_dir='/w0rk5pace/aaworks/datas/RealWorldQA')
"

echo "=== 3/4: GQA (lmms-lab/GQA) ==="
python -c "
from datasets import load_dataset
load_dataset('lmms-lab/GQA', 'testdev_balanced_images', cache_dir='/w0rk5pace/aaworks/datas/GQA')
load_dataset('lmms-lab/GQA', 'testdev_balanced_instructions', cache_dir='/w0rk5pace/aaworks/datas/GQA')
"

echo "=== 4/4: MMMU (MMMU/MMMU, all configs test split) ==="
python -c "
from datasets import load_dataset, get_dataset_config_names
configs = get_dataset_config_names('MMMU/MMMU')
print('Available configs:', configs)
for config in configs:
    print(f'Loading config: {config}')
    load_dataset('MMMU/MMMU', config, split='test', cache_dir=f'/w0rk5pace/aaworks/datas/MMMU/{config}')
"

echo ""
echo "=== Done ==="
echo "Verify with: du -sh /w0rk5pace/aaworks/datas/*/"
