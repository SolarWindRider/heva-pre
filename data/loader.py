"""
HEVA (High-Entropy Visual Attention) 数据加载模块
"""

import json
import os
from PIL import Image
from typing import Dict, List, Any


class VisuRiddlesDataset:
    """VisuRiddles 数据加载器"""

    def __init__(self, data_path: str, image_base_path: str):
        """
        Args:
            data_path: JSON数据文件路径
            image_base_path: 图像基础路径
        """
        self.data_path = data_path
        self.image_base_path = image_base_path

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回样本:
        {
            "id": str,
            "image": PIL.Image,
            "question": str,
            "answer": str,
            "options": str,
            "analysis": str
        }
        """
        item = self.data[idx]

        # 构建图像路径
        img_path = item['imgs'][0]  # 取第一张图
        full_img_path = os.path.join(self.image_base_path, img_path)

        # 加载图像
        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot load image {full_img_path}: {e}")
            # 返回一个空白图像作为fallback
            image = Image.new('RGB', (448, 448), color='white')

        return {
            "id": item['id'],
            "image": image,
            "question": item['question'],
            "answer": item['gold_answer'],
            "options": item.get('option', ''),
            "analysis": item.get('gold_analysis', '')
        }

    def get_sample_by_id(self, sample_id: str) -> Dict[str, Any]:
        """通过ID获取样本"""
        for item in self.data:
            if item['id'] == sample_id:
                idx = self.data.index(item)
                return self[idx]
        raise ValueError(f"Sample with id {sample_id} not found")


def load_dataset(data_path: str = None, image_base_path: str = None) -> VisuRiddlesDataset:
    """加载VisuRiddles数据集的便捷函数"""
    import config
    if data_path is None:
        data_path = config.DATA_FILE
    if image_base_path is None:
        image_base_path = config.IMAGE_DIR

    return VisuRiddlesDataset(data_path, image_base_path)


if __name__ == "__main__":
    # 测试数据加载
    dataset = load_dataset()
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample ID: {sample['id']}")
    print(f"Question: {sample['question'][:100]}...")
    print(f"Answer: {sample['answer']}")
    print(f"Image size: {sample['image'].size}")
