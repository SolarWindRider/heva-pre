CUDA_VISIBLE_DEVICES=5 python 3_run_inference_trace.py --exp_name exp02v2 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA --batch_size 4 --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27 > exp02v2_all.out 2>&1

CUDA_VISIBLE_DEVICES=5 python 3_run_inference_trace.py --exp_name exp02v1 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA --batch_size 2  --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.27 > exp02v1_all.out 2>&1



