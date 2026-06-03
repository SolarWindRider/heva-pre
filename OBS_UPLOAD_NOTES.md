# HEVA 实验环境注意事项

## 图片查看

由于 terminal 环境无法直接查看图片，生成的图片需要上传到 OBS 云端：

```bash
/w0rk5pace/aaworks/obsutil_linux/obsutil cp [图片路径] obs://lixiang01/
```

**重要**：每次生成图片后都要上传，用户才能查看。

## 环境

- **conda 环境**: `conda activate heva`
- **GPU 限制**: 禁止使用 `gpu0`，使用其他 GPU（如 gpu1, gpu2 等）

## 模型路径

- Instruct: `/w0rk5pace/aaworks/Downloads/Models/Qwen/Qwen3-VL-2B-Instruct`
- Thinking: `/w0rk5pace/aaworks/Downloads/Models/Qwen/Qwen3-VL-2B-Thinking`

## 数据路径

- `/w0rk5pace/aaworks/datas/`

## 实验结果

- `/w0rk5pace/aaworks/heva-pre/results/`