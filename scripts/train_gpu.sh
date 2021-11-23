#!/bin/sh
# train
cd ../resnet50_transfer/scripts
bash run_train_gpu.sh
cd ../../resnet50_transfer_mixup_ls/scripts
bash run_train_gpu.sh
cd ../../shufflenetv2_transfer/scripts
bash run_train_gpu.sh
cd ../../efficientnetb0/scripts
bash run_train_gpu.sh
cd ../../efficientnetb0_transfer/scripts
bash run_train_gpu.sh
