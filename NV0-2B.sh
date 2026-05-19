CUDA_VISIBLE_DEVICES=7 python 3_run_inference_trace.py --exp_name exp022 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset AI2D,RealWorldQA,VQAv2,GQA,MMMU,MathVista,MathVision --batch_size 4 --use_attention_guidance false --use_context_aware false > exp022_all.out 2>&1

CUDA_VISIBLE_DEVICES=7 python 3_run_inference_trace.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset AI2D,RealWorldQA,VQAv2,GQA,MMMU,MathVista,MathVision --batch_size 4 --use_attention_guidance false --use_context_aware false > exp021_all.out 2>&1

# CUDA_VISIBLE_DEVICES=7 python 3_run_inference_trace.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset VisuRiddles,RAVEN,MARVEL,LogicVista,PuzzleVQA,AlgoPuzzleVQA,AI2D,RealWorldQA,VQAv2,GQA,MMMU,MathVista,MathVision --batch_size 4 --use_attention_guidance false --use_context_aware false > exp021_all.out 2>&1
