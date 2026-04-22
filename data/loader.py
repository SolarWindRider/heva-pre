"""
HEVA (High-Entropy Visual Attention) 数据加载模块
支持多个数据集: VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA
"""

import json
import os
from PIL import Image
from typing import Dict, List, Any


# 数据集根目录
DATA_ROOT = "../datas"


class AVGDataset:
    """多模态问答数据集加载器"""

    def __init__(self, data: List[Dict], image_base_path: str):
        """
        Args:
            data: 数据列表
            image_base_path: 图像基础路径
        """
        self.data = data
        self.image_base_path = image_base_path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
               返回样本:
               {
                   "id": str,
                   "image            "question":": PIL.Image,
        str,
                   "            "options":answer": str,
        str,
               }
        """
        item = self.data[idx]

        # 构建图像路径
        img_path = item["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_base_path, img_path)

        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Cannot load image {img_path}: {e}")
            image = Image.new("RGB", (448, 448), color="white")

        return {
            "id": str(item.get("id", idx)),
            "image": image,
            "image_path": img_path,
            "question": item["question"],
            "answer": str(item["answer"]),
            "options": item.get("option", ""),
            "issudoku": item.get("issudoku", False),
        }

    def get_sample_by_id(self, sample_id: str) -> Dict[str, Any]:
        """通过ID获取样本"""
        for i, item in enumerate(self.data):
            if str(item.get("id", i)) == sample_id:
                return self[i]
        raise ValueError(f"Sample with id {sample_id} not found")


def preprocess_multimodal_dataset(bench: str) -> List[Dict]:
    """
    加载多模态图文问答数据集

    Args:
        bench: 数据集名称

    Returns:
        数据列表
    """
    if bench == "VisuRiddles":
        dsjson = json.load(open(f"{DATA_ROOT}/VisuRiddles/test_dataset.json"))
        ds = []
        for example in dsjson:
            ds.append(
                {
                    "image_path": example["imgs"][0],
                    "question": example["question"],
                    "option": example["option"],
                    "answer": example["gold_answer"],
                    "id": example["id"],
                    "issudoku": True if "sudoku" in example.get("class", "") else False,
                }
            )
        return ds

    elif bench == "RAVEN":
        dsjson = json.load(open(f"{DATA_ROOT}/RAVEN/raven_test.json"))
        ds = []
        for example in dsjson:
            ds.append(
                {
                    "image_path": example["images"][0],
                    "question": "Which one of the options is the correct answer for the question?",
                    "option": "A, B, C, D, E, F, G, H.",
                    "answer": example["messages"][1]["content"],
                    "id": example["images"][0],
                    "issudoku": False,
                }
            )
        return ds

    elif bench == "MARVEL":
        ds = []
        for i in range(770):  # MARVEL总共只有770个样本
            idx = i + 1
            js = json.load(
                open(f"{DATA_ROOT}/MARVEL_AVR/Json_data/{idx}/{idx}_label.json")
            )
            image_path = f"{DATA_ROOT}/MARVEL_AVR/Json_data/{idx}/{idx}.png"
            ds.append(
                {
                    "image_path": image_path,
                    "question": js["avr_question"],
                    "option": "",  # MARVEL原本的question里面包含options
                    "answer": str(js["answer"]),
                    "id": idx,
                    "issudoku": False,
                }
            )
        return ds

    elif bench == "LogicVista":
        dsjson = json.load(open(f"{DATA_ROOT}/LogicVista/data/dataset.json"))
        ds = []
        for key in dsjson.keys():
            ds.append(
                {
                    "image_path": dsjson[key]["imagename"],
                    "question": dsjson[key]["question"],
                    "option": "",  # LogicVista 原本的question里面包含options
                    "answer": dsjson[key]["answer"],
                    "id": key,
                    "issudoku": False,
                }
            )
        return ds

    elif bench == "PuzzleVQA":
        base_image_dir = f"{DATA_ROOT}/LLM-PuzzleTest/PuzzleVQA/data"
        dsjson = []
        for each in os.listdir(base_image_dir):
            if ".json" not in each:
                continue
            with open(base_image_dir + "/" + each, "r", encoding="utf-8") as f:
                li = f.readlines()
            dsjson += li
        ds = []
        for example in dsjson:
            example = json.loads(example)
            ds.append(
                {
                    "image_path": example["image"],
                    "question": example["question"],
                    "option": str(example["options"])[1:-1] + ".",
                    "answer": example["answer"],
                    "id": example["image"].split("/")[-1],
                    "issudoku": False,
                }
            )
        return ds

    elif bench == "AlgoPuzzleVQA":
        base_image_dir = f"{DATA_ROOT}/LLM-PuzzleTest/AlgoPuzzleVQA/data"
        dsjson = []
        for each in os.listdir(base_image_dir):
            if ".json" not in each:
                continue
            with open(base_image_dir + "/" + each, "r", encoding="utf-8") as f:
                li = f.readlines()
            dsjson += li
        ds = []
        for example in dsjson:
            example = json.loads(example)
            ds.append(
                {
                    "image_path": example["image"],
                    "question": example["question"],
                    "option": str(example["options"])[1:-1] + ".",
                    "answer": example["answer"],
                    "id": example["image"].split("/")[-1],
                    "issudoku": False,
                }
            )
        return ds

    else:
        raise ValueError(f"Unknown dataset: {bench}")


def load_dataset(bench: str = "VisuRiddles") -> AVGDataset:
    """
    加载数据集的便捷函数

    Args:
        bench: 数据集名称

    Returns:
        数据集对象
    """
    data = preprocess_multimodal_dataset(bench)

    # 获取图像基础路径
    if bench == "VisuRiddles":
        base_path = f"{DATA_ROOT}/VisuRiddles"
    elif bench == "RAVEN":
        base_path = f"{DATA_ROOT}/RAVEN"
    elif bench == "MARVEL":
        base_path = f"{DATA_ROOT}/MARVEL_AVR"
    elif bench == "LogicVista":
        base_path = f"{DATA_ROOT}/LogicVista/data/images"
    elif bench == "PuzzleVQA":
        base_path = f"{DATA_ROOT}/LLM-PuzzleTest/PuzzleVQA/data"
    elif bench == "AlgoPuzzleVQA":
        base_path = f"{DATA_ROOT}/LLM-PuzzleTest/AlgoPuzzleVQA/data"
    else:
        base_path = DATA_ROOT

    return AVGDataset(data, base_path)


# 支持的数据集列表
SUPPORTED_DATASETS = [
    "VisuRiddles",
    "RAVEN",
    "MARVEL",
    "LogicVista",
    "PuzzleVQA",
    "AlgoPuzzleVQA",
]


if __name__ == "__main__":
    # 测试数据加载
    for bench in SUPPORTED_DATASETS:
        print(f"\n=== {bench} ===")
        try:
            dataset = load_dataset(bench)
            print(f"Dataset size: {len(dataset)}")
            sample = dataset[0]
            print(f"Sample ID: {sample['id']}")
            print(f"Question: {sample['question'][:80]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Image: {sample['image_path']}")
        except Exception as e:
            print(f"Error: {e}")
