# ASCEND_RT_VISIBLE_DEVICES=2 nohup python ./experiments/1_run_inference.py --exp_name exp001 --dataset MARVEL --alpha_values 0.2 --batch_size 1 --max_new_tokens 12288 --num_gpus 1  > MARVEL_print2.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3  python 1_run_inference.py --exp_name exp001 --dataset MARVEL --max_new_tokens 12000
ASCEND_RT_VISIBLE_DEVICES=0 nohup python 1_run_inference.py --exp_name exp003 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Thinking --dataset VisuRiddles     > exp003_VisuRiddles.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=1 nohup python 1_run_inference.py --exp_name exp003 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Thinking --dataset RAVEN > exp003_RAVEN.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=2 nohup python 1_run_inference.py --exp_name exp003 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Thinking --dataset MARVEL > exp003_MARVEL.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=3 nohup python 1_run_inference.py --exp_name exp003 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Thinking --dataset LogicVista > exp003_LogicVista.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=4 nohup python 1_run_inference.py --exp_name exp003 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Thinking --dataset PuzzleVQA > exp003_PuzzleVQA.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=5 nohup python 1_run_inference.py --exp_name exp003 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Thinking --dataset AlgoPuzzleVQA > exp003_AlgoPuzzleVQA.out 2>&1 &
