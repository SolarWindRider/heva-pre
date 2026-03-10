# 调用 start.sh
# 前三个实验使用卡6作为HEVA计算卡 (VisuRiddles, RAVEN, MARVEL)
# 后三个实验使用卡7作为HEVA计算卡 (LogicVista, PuzzleVQA, AlgoPuzzleVQA)

# nohup bash ./start.sh --exp_name exp001 -d VisuRiddles   -a 0.2 -n 300 -b 1 -s true -m 12288 -g 8  > VisuRiddles-RAVEN-MARVEL.out &


# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 nohup bash ./start.sh --exp_name exp001 -d VisuRiddles   -a 0.2 -n 300 -b 1 -s true -m 12288 -g 1  > VisuRiddles-RAVEN-MARVEL.out &
# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup bash ./start.sh --exp_name exp001 -d RAVEN         -a 0.2 -n 300 -b 1 -s true -m 12288 -g 1  > LogicVista-PuzzleVQA-AlgoPuzzleVQA.out &
ASCEND_RT_VISIBLE_DEVICES=2,3 nohup bash ./start.sh --exp_name exp001 -d MARVEL        -a 0.2 -n 300 -b 1 -s true -m 12288 -g 2 --heva_device npu:1 > MARVEL.out &
# ASCEND_RT_VISIBLE_DEVICES=4,7 nohup bash ./start.sh --exp_name exp001 -d LogicVista    -a 0.2 -n 300 -b 1 -s true -m 12288 -g 1 --heva_device npu:1 > LogicVista.out &
# ASCEND_RT_VISIBLE_DEVICES=5,7 nohup bash ./start.sh --exp_name exp001 -d PuzzleVQA     -a 0.2 -n 300 -b 1 -s true -m 12288 -g 1 --heva_device npu:1 > PuzzleVQA.out &
# ASCEND_RT_VISIBLE_DEVICES=6,7 nohup bash ./start.sh --exp_name exp001 -d AlgoPuzzleVQA -a 0.2 -n 300 -b 1 -s true -m 12288 -g 1 --heva_device npu:1 > AlgoPuzzleVQA.out &




