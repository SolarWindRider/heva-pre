"""
HEVA (High-Entropy Visual Attention) 数据加载模块
支持多个数据集: VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA
"""

import json
import os
import csv
from PIL import Image
from typing import Dict, List, Any


# 数据集根目录
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datas"))


# ============================================
# Helper Functions
# ============================================

def int2letter(idx: int) -> str:
    """Convert integer 0-3 to letter A-D"""
    return chr(ord("A") + idx) if 0 <= idx <= 3 else str(idx)


def format_mcq_options(choices) -> str:
    """Format choices list/dict as 'A. xxx B. xxx C. xxx D. xxx' string"""
    if isinstance(choices, dict):
        parts = []
        for k in sorted(choices.keys()):
            parts.append(f"{k}. {choices[k]}")
        return " ".join(parts)
    elif isinstance(choices, list):
        parts = []
        for i, c in enumerate(choices):
            parts.append(f"{chr(ord('A') + i)}. {c}")
        return " ".join(parts)
    return str(choices)


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

        # 支持直接传入 PIL Image 或图像路径
        image = item.get("image")
        img_path = item.get("image_path", "")
        if image is None or not isinstance(image, Image.Image):
            # Load from path
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.image_base_path, img_path)
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
            "answer_format": item.get("answer_format", "mcq"),
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

    elif bench == "AI2D":
        image_base = f"{DATA_ROOT}/AI2D/images"
        ds = []
        with open(f"{DATA_ROOT}/AI2D/test.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for example in reader:
                ds.append(
                    {
                        "image_path": os.path.join(image_base, example["image"]),
                        "question": example["question"],
                        "option": format_mcq_options(example["options"]),
                        "answer": example["answer"].upper(),
                        "answer_format": "mcq",
                        "id": example["id"],
                        "issudoku": False,
                    }
                )
        return ds

    elif bench == "RealWorldQA":
        image_base = f"{DATA_ROOT}/RealWorldQA/images"
        ds = []
        with open(f"{DATA_ROOT}/RealWorldQA/data.tsv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for example in reader:
                ds.append(
                    {
                        "image_path": os.path.join(image_base, example.get("image", example.get("image_path", ""))),
                        "question": example["question"],
                        "option": f"A. {example['A']} B. {example['B']} C. {example['C']} D. {example['D']}",
                        "answer": example["answer"].upper(),
                        "answer_format": "mcq",
                        "id": example["index"],
                        "issudoku": False,
                    }
                )
        return ds

    elif bench == "VQAv2":
        dsjson = json.load(open(f"{DATA_ROOT}/VQAv2/vqav2_val.json"))
        image_base = f"{DATA_ROOT}/VQAv2/images"
        ds = []
        for example in dsjson:
            ds.append(
                {
                    "image_path": os.path.join(image_base, f"{example['image_id']}.jpg"),
                    "question": example["question"],
                    "option": "",
                    "answer": example["multiple_choice_answer"],
                    "answer_format": "open_vqa",
                    "id": str(example["question_id"]),
                    "issudoku": False,
                }
            )
        return ds

    elif bench == "GQA":
        dsjson = json.load(open(f"{DATA_ROOT}/GQA/test.json"))
        image_base = f"{DATA_ROOT}/GQA/images"
        ds = []
        for item_id, example in dsjson.items():
            ds.append(
                {
                    "image_path": os.path.join(image_base, f"{example['image_id']}.jpg"),
                    "question": example["question"],
                    "option": format_mcq_options(example.get("options", {})),
                    "answer": example["answer"].upper(),
                    "answer_format": "mcq",
                    "id": item_id,
                    "issudoku": False,
                }
            )
        return ds

    elif bench == "MMMU":
        import pyarrow.parquet as pq
        import io
        data_list = []
        for root, dirs, files in os.walk(f"{DATA_ROOT}/MMMU"):
            for f in files:
                if not f.endswith(".parquet") or "-test-" in f:
                    # Skip test splits - they have corrupted parquet files
                    continue
                path = os.path.join(root, f)
                try:
                    tbl = pq.read_table(path, use_threads=False)
                except Exception as e:
                    print(f"Warning: skip corrupted {path}: {e}")
                    continue
                d = tbl.to_pydict()
                for i in range(tbl.num_rows):
                    opts = d["options"][i]
                    # Extract first available image from image_1 to image_7
                    image = None
                    for img_key in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
                        img_data = d.get(img_key, [None])[i]
                        if img_data and isinstance(img_data, dict) and "bytes" in img_data:
                            try:
                                image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                                break
                            except:
                                continue
                    data_list.append(
                        {
                            "image": image,
                            "question": d["question"][i],
                            "option": format_mcq_options(opts),
                            "answer": str(d["answer"][i]).upper(),
                            "answer_format": "mcq",
                            "id": str(d["id"][i]),
                            "issudoku": False,
                        }
                    )
        return data_list

    elif bench == "MathVista":
        import pyarrow.parquet as pq
        import io
        tbl = pq.read_table("/data2/lixiang/datas/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet", use_threads=False)
        d = tbl.to_pydict()
        data_list = []
        for i in range(tbl.num_rows):
            choices = d["choices"][i]
            di = d["decoded_image"][i]
            image = None
            if isinstance(di, dict) and "bytes" in di:
                try:
                    image = Image.open(io.BytesIO(di["bytes"])).convert("RGB")
                except:
                    pass
            data_list.append(
                {
                    "image": image,
                    "question": d["question"][i],
                    "option": format_mcq_options(choices) if choices else "",
                    "answer": str(d["answer"][i]),
                    "answer_format": "mcq" if choices else "open_vqa",
                    "id": str(d["pid"][i]),
                    "issudoku": False,
                }
            )
        return data_list

    elif bench == "MathVision":
        from datasets import load_dataset
        import io
        ds = load_dataset("/data2/lixiang/datas/MathVision/data", split="test")
        data_list = []
        for i, item in enumerate(ds):
            choices = item.get("choices")
            opts = format_mcq_options(choices) if choices else ""
            di = item["decoded_image"]
            image = None
            if isinstance(di, dict) and "bytes" in di:
                try:
                    image = Image.open(io.BytesIO(di["bytes"])).convert("RGB")
                except:
                    pass
            data_list.append(
                {
                    "image": image,
                    "question": item["question"],
                    "option": opts,
                    "answer": str(item["answer"]),
                    "answer_format": "mcq" if choices else "open_vqa",
                    "id": str(item["id"]),
                    "issudoku": False,
                }
            )
        return data_list

    elif bench == "MMMU":
        import pyarrow.parquet as pq
        data_list = []
        for root, dirs, files in os.walk(DATA_ROOT):
            for f in files:
                if not f.endswith(".parquet"):
                    continue
                tbl = pq.read_table(os.path.join(root, f))
                d = tbl.to_pydict()
                for i in range(tbl.num_rows):
                    opts = d["options"][i]
                    data_list.append(
                        {
                            "question": d["question"][i],
                            "option": format_mcq_options(opts),
                            "answer": str(d["answer"][i]).upper(),
                            "answer_format": "mcq",
                            "id": str(d["id"][i]),
                            "issudoku": False,
                        }
                    )
        return data_list

    elif bench == "MMMU_Pro":
        import pyarrow.parquet as pq
        import io
        base_dir = "/data2/lixiang/datas/MMMU_Pro"
        data_list = []
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if not f.endswith(".parquet") or "lock" in f or "incomplete" in f:
                    continue
                path = os.path.join(root, f)
                try:
                    tbl = pq.read_table(path, use_threads=False)
                except Exception:
                    continue
                d = tbl.to_pydict()
                cols = set(d.keys())
                for i in range(tbl.num_rows):
                    if "question" in cols:
                        # standard format: has image_1 to image_7
                        opts = d.get("options")
                        opts_val = opts[i] if opts else None
                        image = None
                        for img_key in [f"image_{j}" for j in range(1, 8)]:
                            img_data = d.get(img_key, [None])[i]
                            if img_data and isinstance(img_data, dict) and "bytes" in img_data:
                                try:
                                    image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                                    break
                                except:
                                    continue
                        data_list.append(
                            {
                                "image": image,
                                "question": d["question"][i],
                                "option": format_mcq_options(opts_val) if opts_val else "",
                                "answer": str(d["answer"][i]).upper(),
                                "answer_format": "mcq",
                                "id": str(d.get("id", [i])[i]),
                                "issudoku": False,
                            }
                        )
                    elif "image" in cols:
                        # vision format: has decoded image bytes
                        img_data = d["image"][i]
                        image = None
                        if isinstance(img_data, dict) and "bytes" in img_data:
                            try:
                                image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                            except:
                                pass
                        opts = d.get("options")
                        opts_val = opts[i] if opts else None
                        data_list.append(
                            {
                                "image": image,
                                "question": d.get("question", [""])[i] if "question" in d else "",
                                "option": format_mcq_options(opts_val) if opts_val else "",
                                "answer": str(d["answer"][i]).upper(),
                                "answer_format": "mcq",
                                "id": str(d.get("id", [i])[i]),
                                "issudoku": False,
                            }
                        )
        return data_list

    elif bench == "qwen3_vl_math":
        # qwen3-vl-math not yet downloaded
        print("Warning: qwen3_vl_math has no local data (download needed)")
        return []

    elif bench == "zerobench":
        # zerobench images not yet downloaded (only metadata cache exists)
        print("Warning: zerobench has no local data (download needed)")
        return []

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
    elif bench == "AI2D":
        base_path = f"{DATA_ROOT}/AI2D"
    elif bench == "RealWorldQA":
        base_path = f"{DATA_ROOT}/RealWorldQA/images"
    elif bench == "VQAv2":
        base_path = f"{DATA_ROOT}/VQAv2/images"
    elif bench == "GQA":
        base_path = f"{DATA_ROOT}/GQA/images"
    elif bench == "MMMU":
        base_path = f"{DATA_ROOT}/MMMU"
    elif bench == "MMMU_Pro":
        base_path = f"{DATA_ROOT}/MMMU_Pro"
    elif bench in ("MathVista", "MathVision", "zerobench", "qwen3_vl_math"):
        base_path = DATA_ROOT
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
    "AI2D",
    "RealWorldQA",
    "VQAv2",
    "GQA",
    "MMMU",
    "MathVista",
    "MathVision",
    "MMMU_Pro",
    "zerobench",
    "qwen3_vl_math",
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
