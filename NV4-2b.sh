CUDA_VISIBLE_DEVICES=3 python 3_run_inference_trace.py --exp_name exp026 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA,AI2D,RealWorldQA,VQAv2,GQA,MMMU,MathVista,MathVision --batch_size 2 --use_attention_guidance true  --dla_entropy_threshold 1.3 --use_context_aware true --ctx_entropy_threshold 1.3 > exp026_all.out 2>&1

CUDA_VISIBLE_DEVICES=3 python 3_run_inference_trace.py --exp_name exp025 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA,AI2D,RealWorldQA,VQAv2,GQA,MMMU,MathVista,MathVision --batch_size 2 --use_attention_guidance true  --dla_entropy_threshold 1.3 --use_context_aware true --ctx_entropy_threshold 1.3 > exp025_all.out 2>&1


