#!/bin/sh

for i in 5 10 15 20 25 30 35 40 45;
do
	echo "epoch: ${i}" >> result_transfer.txt
	/home/twang/anaconda3/envs/mindspore/bin/python /home/twang/BDCI2021/ResNet50/docs/docs/sample_code/resnet/transfer.py --checkpoint_path=/home/twang/BDCI2021/checkpoint/ckpt_transfer/train_resnet_transfer-${i}_2694.ckpt >> result_transfer.txt
done
for i in 5 10;
do
	echo "epoch: $((i+45))" >> result_transfer.txt
	# /home/twang/anaconda3/envs/mindspore/bin/python /home/twang/BDCI2021/ResNet50/docs/docs/sample_code/resnet/test.py --checkpoint_path=/home/twang/BDCI2021/checkpoint/ckpt_transfer/train_resnet_transfer_1-${i}_2694.ckpt >> result_transfer.txt
	/home/twang/anaconda3/envs/mindspore/bin/python /home/twang/BDCI2021/ResNet50/docs/docs/sample_code/resnet/transfer.py --checkpoint_path=/home/twang/BDCI2021/checkpoint/ckpt_transfer/train_resnet_transfer_1-${i}_2694.ckpt >> result_transfer.txt
done