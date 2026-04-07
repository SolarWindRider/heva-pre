# ASCEND_RT_VISIBLE_DEVICES=2 nohup python ./experiments/1_run_inference.py --exp_name exp001 --dataset MARVEL --alpha_values 0.2 --batch_size 1 --max_new_tokens 12288 --num_gpus 1  > MARVEL_print2.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3  python 1_run_inference.py --exp_name exp001 --dataset MARVEL --max_new_tokens 12000
ASCEND_RT_VISIBLE_DEVICES=0,1 nohup python 2_run_inference_heva_force.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-8B-Thinking --dataset VisuRiddles     > exp021_VisuRiddles.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=2,3 nohup python 2_run_inference_heva_force.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-8B-Thinking --dataset RAVEN > exp021_RAVEN.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=4,5 nohup python 2_run_inference_heva_force.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-8B-Thinking --dataset MARVEL > exp021_MARVEL.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=6,7 nohup python 2_run_inference_heva_force.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-8B-Thinking --dataset LogicVista > exp021_LogicVista.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=4 nohup python 2_run_inference_heva_force.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset PuzzleVQA > exp021_PuzzleVQA.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=5 nohup python 2_run_inference_heva_force.py --exp_name exp021 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset AlgoPuzzleVQA > exp021_AlgoPuzzleVQA.out 2>&1 &

# ASCEND_RT_VISIBLE_DEVICES=6 nohup python 1_run_inference.py --exp_name exp006 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Instruct --dataset VisuRiddles     > exp006_VisuRiddles.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=7 nohup python 1_run_inference.py --exp_name exp006 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Instruct --dataset RAVEN > exp006_RAVEN.out 2>&1 &

# ASCEND_RT_VISIBLE_DEVICES=6 nohup python 1_run_inference.py --exp_name exp006 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Instruct --dataset MARVEL     > exp006_MARVEL.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=7 nohup python 1_run_inference.py --exp_name exp006 --model_path ../Downloads/Models/Qwen/Qwen3-VL-4B-Instruct --dataset LogicVista > exp006_LogicVista.out 2>&1 &


# ASCEND_RT_VISIBLE_DEVICES=0 nohup python 1_run_inference.py --exp_name exp007 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset VisuRiddles     > exp007_VisuRiddles.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=1 nohup python 1_run_inference.py --exp_name exp007 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset RAVEN > exp007_RAVEN.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=2 nohup python 1_run_inference.py --exp_name exp007 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset MARVEL > exp007_MARVEL.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=3 nohup python 1_run_inference.py --exp_name exp007 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset LogicVista > exp007_LogicVista.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=4 nohup python 1_run_inference.py --exp_name exp007 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset PuzzleVQA > exp007_PuzzleVQA.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=5 nohup python 1_run_inference.py --exp_name exp007 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking --dataset AlgoPuzzleVQA > exp007_AlgoPuzzleVQA.out 2>&1 &

# ASCEND_RT_VISIBLE_DEVICES=0 nohup python 2_run_inference_heva_force.py --exp_name exp010 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset VisuRiddles     > exp010_VisuRiddles.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=1 nohup python 2_run_inference_heva_force.py --exp_name exp010 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset RAVEN > exp010_RAVEN.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=2 nohup python 2_run_inference_heva_force.py --exp_name exp010 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset MARVEL > exp010_MARVEL.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=3 nohup python 2_run_inference_heva_force.py --exp_name exp010 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset LogicVista > exp010_LogicVista.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=4 nohup python 2_run_inference_heva_force.py --exp_name exp010 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset PuzzleVQA > exp010_PuzzleVQA.out 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=5 nohup python 2_run_inference_heva_force.py --exp_name exp010 --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct --dataset AlgoPuzzleVQA > exp010_AlgoPuzzleVQA.out 2>&1 &
