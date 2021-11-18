#!/bin/sh

# for i in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230;
for i in 240 250 260 270 280 290 300;
do
	echo "epoch: $i" >> Result.txt
	/home/twang/anaconda3/envs/mindspore/bin/python -u /home/twang/BDCI2021/ResNet50/docs/docs/sample_code/resnet/test.py --checkpoint_path=/home/twang/BDCI2021/distribute_train/train_resnet_distribute-${i}_673.ckpt >> Result.txt
done