CUDA_VISIBLE_DEVICES=4 python 3_run_inference_trace.py --exp_name exp028 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA,AI2D,RealWorldQA,MMMU,MMMU_Pro,MathVista,MathVision --batch_size 1 --resume true --use_attention_guidance true --dla_entropy_threshold 1.3 --use_context_aware false > exp028_all.out 2>&1

CUDA_VISIBLE_DEVICES=4 python 3_run_inference_trace.py --exp_name exp027 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA,AI2D,RealWorldQA,MMMU,MMMU_Pro,MathVista,MathVision --batch_size 1 --resume true --use_attention_guidance true --dla_entropy_threshold 1.3 --use_context_aware false > exp027_all.out 2>&1

