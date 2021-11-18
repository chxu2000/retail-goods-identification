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
from mindspore import load_checkpoint, load_param_into_net
from resnet import resnet50
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P


DEVICE_NUM = 1
EPOCH_PER_CKPT = 1
# common.set_seed(1)
random.seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=DEVICE_NUM, help='Device num.')
parser.add_argument('--epoch_size', type=int, default=100, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=2388, help='Num classes.')
parser.add_argument('--device_target', type=str, default='GPU', help='Device choice Ascend or GPU')

# 训练
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default='/home/twang/BDCI2021/train', help='Dataset path.')
# init("nccl")

args_opt = parser.parse_args()
data_home = args_opt.dataset_path
context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=False, reduction='mean', smooth_factor=0.1, num_classes=2388):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss

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
    onehot_op = C2.OneHot(num_classes=2388)

    cifar_ds = cifar_ds.map(operations=[random_horizontal_op, resize_op, rescale_op, normalize_op, changeswap_op], input_columns=["image"], output_columns=["image"])
    # cifar_ds = cifar_ds.map(operations=onehot_op, input_columns=["label"], output_columns=["label"])
    # Declare an apply_func function which returns a Dataset object
    def apply_func(data):
        data = data.batch(args_opt.batch_size,True)
        # mixup_op = C.MixUpBatch(alpha=0.5)
        # data = data.map(operations=mixup_op, input_columns=["image", "label"], output_columns=["image", "label"])
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
    net = resnet50(args_opt.batch_size, args_opt.num_classes)
    # ls = SoftmaxCrossEntropyWithLogits(sparse=False, reduction="mean")
    ls = CrossEntropySmooth(sparse=True, reduction="mean")


    opt = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

    # as for train, users could use model.train
    if args_opt.do_train:
        dataset = create_dataset()
        batch_num = dataset.get_dataset_size()

        config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * EPOCH_PER_CKPT, keep_checkpoint_max=30)
        ckpoint_cb = ModelCheckpoint(prefix="train_resnet-mixup", directory="/home/twang/train_mixup_2021-11-7", config=config_ck)

        # param_dict = load_checkpoint(args_opt.checkpoint_path)
        # load_param_into_net(net, param_dict)

        loss_cb = LossMonitor()
        model.train(epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])
