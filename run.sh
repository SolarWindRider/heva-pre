# 调用 start.sh
# 前三个实验使用卡6作为HEVA计算卡 (VisuRiddles, RAVEN, MARVEL)
# 后三个实验使用卡7作为HEVA计算卡 (LogicVista, PuzzleVQA, AlgoPuzzleVQA)

ASCEND_RT_VISIBLE_DEVICES=0,3 nohup bash ./start.sh --exp_name exp001 -d VisuRiddles   -n 300 -b 1 -s true -m 2048 -g 1 --heva_device npu:1 > VisuRiddles.out &
ASCEND_RT_VISIBLE_DEVICES=1,3 nohup bash ./start.sh --exp_name exp001 -d RAVEN         -n 300 -b 1 -s true -m 2048 -g 1 --heva_device npu:1 > RAVEN.out &
ASCEND_RT_VISIBLE_DEVICES=2,3 nohup bash ./start.sh --exp_name exp001 -d MARVEL        -n 300 -b 1 -s true -m 2048 -g 1 --heva_device npu:1 > MARVEL.out &
ASCEND_RT_VISIBLE_DEVICES=4,7 nohup bash ./start.sh --exp_name exp001 -d LogicVista    -n 300 -b 1 -s true -m 2048 -g 1 --heva_device npu:1 > LogicVista.out &
ASCEND_RT_VISIBLE_DEVICES=5,7 nohup bash ./start.sh --exp_name exp001 -d PuzzleVQA     -n 300 -b 1 -s true -m 2048 -g 1 --heva_device npu:1 > PuzzleVQA.out &
ASCEND_RT_VISIBLE_DEVICES=6,7 nohup bash ./start.sh --exp_name exp001 -d AlgoPuzzleVQA -n 300 -b 1 -s true -m 2048 -g 1 --heva_device npu:1 > AlgoPuzzleVQA.out &




