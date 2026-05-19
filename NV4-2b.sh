CUDA_VISIBLE_DEVICES=3 python 3_run_inference_trace.py --exp_name exp024 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA --batch_size 2 --use_attention_guidance true  --dla_entropy_threshold 1.27 --use_context_aware true --ctx_entropy_threshold 1.27 > exp026_all.out 2>&1

CUDA_VISIBLE_DEVICES=3 python 3_run_inference_trace.py --exp_name exp025 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA --batch_size 2 --use_attention_guidance true  --dla_entropy_threshold 1.27 --use_context_aware true --ctx_entropy_threshold 1.27 > exp025_all.out 2>&1


