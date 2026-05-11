CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v2 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Instruct --dataset VisuRiddles   --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v2_VisuRiddles.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v2 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Instruct --dataset RAVEN         --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v2_RAVEN.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v2 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Instruct --dataset MARVEL        --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v2_MARVEL.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v2 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Instruct --dataset LogicVista    --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v2_LogicVista.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v2 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Instruct --dataset PuzzleVQA     --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v2_PuzzleVQA.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v2 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Instruct --dataset AlgoPuzzleVQA --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v2_AlgoPuzzleVQA.out 2>&1



CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v1 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Thinking --dataset VisuRiddles   --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v1_VisuRiddles.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v1 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Thinking --dataset RAVEN         --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v1_RAVEN.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v1 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Thinking --dataset MARVEL        --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v1_MARVEL.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v1 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Thinking --dataset LogicVista    --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v1_LogicVista.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v1 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Thinking --dataset PuzzleVQA     --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v1_PuzzleVQA.out 2>&1
CUDA_VISIBLE_DEVICES=6  python 3_run_inference_trace.py --exp_name exp02v1 --model_path /data/public/model/Qwen/Qwen3-VL-2B-Thinking --dataset AlgoPuzzleVQA --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27   > exp02v1_AlgoPuzzleVQA.out 2>&1



