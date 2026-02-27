"""
图像扰动函数模块

实现以下扰动:
1. shuffle_patches: 打补丁
2. gaussian_blur: 高斯模糊
3. zero_image: 零图像
"""

import numpy as np
from PIL import Image, ImageFilter
import random
from typing import Tuple


def shuffle_patches(image: Image.Image, patch_size: int = 32) -> Image.Image:
    """
    打乱图像补丁

    Args:
        image: 输入图像
        patch_size: 补丁大小

    Returns:
        打乱后的图像
    """
    img_array = np.array(image)

    # 获取图像尺寸
    h, w = img_array.shape[:2]

    # 计算可以分割的补丁数量
    n_h = h // patch_size
    n_w = w // patch_size

    # 分割图像
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = img_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)

    # 打乱补丁顺序
    random.shuffle(patches)

    # 重新组合
    result = np.zeros_like(img_array)
    idx = 0
    for i in range(n_h):
        for j in range(n_w):
            result[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[idx]
            idx += 1

    return Image.fromarray(result)


def gaussian_blur(image: Image.Image, radius: int = 10) -> Image.Image:
    """
    高斯模糊

    Args:
        image: 输入图像
        radius: 模糊半径

    Returns:
        模糊后的图像
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def zero_image(image: Image.Image) -> Image.Image:
    """
    零图像（全黑）

    Args:
        image: 输入图像

    Returns:
        全黑图像
    """
    return Image.new(image.mode, image.size, color='black')


def add_gaussian_noise(image: Image.Image, std: float = 30) -> Image.Image:
    """
    添加高斯噪声

    Args:
        image: 输入图像
        std: 噪声标准差

    Returns:
        添加噪声后的图像
    """
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, std, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def shuffle_pixels(image: Image.Image, seed: int = None) -> Image.Image:
    """
    打乱像素（更彻底的扰动）

    Args:
        image: 输入图像
        seed: 随机种子

    Returns:
        打乱后的图像
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_array = np.array(image)
    flat = img_array.flatten()
    np.random.shuffle(flat)
    shuffled = flat.reshape(img_array.shape)
    return Image.fromarray(shuffled)


def get_perturbation_functions():
    """
    获取所有扰动函数

    Returns:
        dict: 函数名到函数的映射
    """
    return {
        'shuffle_patches': lambda img: shuffle_patches(img, patch_size=32),
        'gaussian_blur': lambda img: gaussian_blur(img, radius=10),
        'zero_image': zero_image,
        'gaussian_noise': lambda img: add_gaussian_noise(img, std=30),
        'shuffle_pixels': shuffle_pixels,
    }


if __name__ == "__main__":
    # 测试
    from data.loader import load_dataset

    dataset = load_dataset()
    sample = dataset[0]
    image = sample['image']

    print("Original image size:", image.size)

    # 测试各种扰动
    perturbations = get_perturbation_functions()

    for name, func in perturbations.items():
        perturbed = func(image.copy())
        print(f"{name}: {perturbed.size}")
