# 调用 start.sh
ASCEND_RT_VISIBLE_DEVICES=0 nohup bash ./start.sh --exp_name exp001 -d VisuRiddles   -n 300 -b 1 -s true -m 24576 -g 1 > VisuRiddles.out &
ASCEND_RT_VISIBLE_DEVICES=1 nohup bash ./start.sh --exp_name exp001 -d RAVEN         -n 300 -b 1 -s true -m 24576 -g 1 > RAVEN.out &
ASCEND_RT_VISIBLE_DEVICES=2 nohup bash ./start.sh --exp_name exp001 -d MARVEL        -n 300 -b 1 -s true -m 24576 -g 1 > MARVEL.out &
ASCEND_RT_VISIBLE_DEVICES=3 nohup bash ./start.sh --exp_name exp001 -d LogicVista    -n 300 -b 1 -s true -m 24576 -g 1 > LogicVista.out &
ASCEND_RT_VISIBLE_DEVICES=4 nohup bash ./start.sh --exp_name exp001 -d PuzzleVQA     -n 300 -b 1 -s true -m 24576 -g 1 > PuzzleVQA.out &
ASCEND_RT_VISIBLE_DEVICES=5 nohup bash ./start.sh --exp_name exp001 -d AlgoPuzzleVQA -n 300 -b 1 -s true -m 24576 -g 1 > AlgoPuzzleVQA.out &




