# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""cifar_resnet50
This sample code is applicable to Ascend.
"""
import os
import random
import argparse
import logging
import mindspore
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.py_transforms as P
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspore.nn import Momentum, Adam, AdamWeightDecay
from mindspore import Model, context
from mindspore import common
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import load_checkpoint, load_param_into_net,Tensor,Parameter
import mindspore_hub as mshub
import numpy as np
from src.shufflenetv2_transfer import ShuffleNetV2


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE_NUM = 1
EPOCH_PER_CKPT = 3
MAX_CKPT = 5
# common.set_seed(1)
random.seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=DEVICE_NUM > 1, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=DEVICE_NUM, help='Device num.')
parser.add_argument('--epoch_size', type=int, default=50, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=2388, help='Num classes.')
parser.add_argument('--device_target', type=str, default='GPU', help='Device choice Ascend or GPU')

# # 训练
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--checkpoint_path', type=str, default='', help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default='', help='Dataset path.')
args_opt = parser.parse_args()
data_home = args_opt.dataset_path

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
# context.set_context(mode=False, device_target=args_opt.device_target)


def create_dataset(repeat_num=1, training=True):
    
    assert os.path.exists(data_home), "the dataset path is invalid!"
    if args_opt.run_distribute:
        rank_id = 0
        rank_size = DEVICE_NUM
        cifar_ds = ds.ImageFolderDataset(data_home, decode=True, shuffle=True, num_shards=rank_size, shard_id=rank_id)
    else:
        cifar_ds = ds.ImageFolderDataset(data_home, decode=True, shuffle=True)
 
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0
    random_horizontal_op = C.RandomHorizontalFlip()
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW() 

    cifar_ds = cifar_ds.map(operations=[random_horizontal_op, resize_op, rescale_op, normalize_op, changeswap_op], input_columns=["image"], output_columns=["image"])
    def apply_func(data):
        data = data.batch(args_opt.batch_size,True)
        return data
    cifar_ds = cifar_ds.apply(apply_func)
    return cifar_ds

if __name__ == '__main__':
    # in this way by judging the mark of args, users will decide which function to use
    if not args_opt.do_eval and args_opt.run_distribute:
        context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.AUTO_PARALLEL,
                                          all_reduce_fusion_config=[140])
        init()

    epoch_size = args_opt.epoch_size


    #model = "mindspore/ascend/1.1/shufflenetv2_v1.1"
    # initialize the number of classes based on the pre-trained model
    net = ShuffleNetV2()
    para_dict=load_checkpoint('/home/twang/BDCI2021/retail-goods-identification/shuffleNetv2/shufflenetv2_GPU_v111_imagenet2012_offical_cv_bs256_acc69.ckpt')
    mod_weight=Parameter(Tensor(np.ones((2388, 1024)),mindspore.float32) ,name='moments.classifier.0.weight', requires_grad=True)
    para_dict['classifier.0.weight']=mod_weight
    load_param_into_net(net,para_dict)
    net.set_train(True)

    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

    # as for train, users could use model.train
    if args_opt.do_train:
        dataset = create_dataset()
        batch_num = dataset.get_dataset_size()

        config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * EPOCH_PER_CKPT, keep_checkpoint_max=MAX_CKPT)
        ckpoint_cb = ModelCheckpoint(prefix="train_transfer_shuffle2", directory="../ckpt", config=config_ck)

        loss_cb = LossMonitor()
        model.train(epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])
