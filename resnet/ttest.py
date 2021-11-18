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
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspore.nn import Momentum, Adam, AdamWeightDecay
from mindspore import Model, context
from mindspore import common
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import load_checkpoint, load_param_into_net
from resnet import resnet50

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--epoch_size', type=int, default=100, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=2388, help='Num classes.')
parser.add_argument('--device_target', type=str, default='GPU', help='Device choice Ascend or GPU')
parser.add_argument('--do_train', type=bool, default=False, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=True, help='Do eval or not.')
# parser.add_argument('--checkpoint_path', type=str, default='/home/xiangsheng/XUCHENHAO/MindSpore/checkpoint/ckpt_original/train_resnet_original-45_2694.ckpt', help='CheckPoint file path.')
# parser.add_argument('--checkpoint_path', type=str, default='/home/xiangsheng/XUCHENHAO/MindSpore/checkpoint/train_resnet_label_smoothing/train_resnet_label_smoothing_4-100_2694.ckpt', help='CheckPoint file path.')
parser.add_argument('--checkpoint_path', type=str, default='/data/home/twang/BDCI2021/checkpoint/ckpt_transfer/train_resnet_transfer-2_2694.ckpt', help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default='/home/twang/BDCI2021/test', help='Dataset path.')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args_opt = parser.parse_args()
data_home = args_opt.dataset_path
context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

def create_dataset(repeat_num=1, training=True):
    """

    create data for next use such as training or inferring

    """
    assert os.path.exists(data_home), "the dataset path is invalid!"
    if args_opt.run_distribute:
        rank_id = 0
        rank_size = 4
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
    # test
    # normalize_op = C.Normalize((0.4726, 0.4258, 0.3550), (0.2742, 0.2608, 0.2536))
    # train
    # normalize_op = C.Normalize((0.4717, 0.4250, 0.3539), (0.2741, 0.2604, 0.2532))
    # imagenet
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    cifar_ds = cifar_ds.map(operations=[random_horizontal_op, resize_op,rescale_op,normalize_op,changeswap_op], input_columns=["image"], output_columns=["image"])

    # Declare an apply_func function which returns a Dataset object
    def apply_func(data):
        data = data.batch(args_opt.batch_size,True)
        return data
    cifar_ds = cifar_ds.apply(apply_func)
    return cifar_ds

if __name__ == '__main__':
    # in this way by judging the mark of args, users will decide which function to use
    
    epoch_size = args_opt.epoch_size
    net = resnet50(args_opt.batch_size, args_opt.num_classes)
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})
    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
    eval_dataset = create_dataset(training=False)
    res = model.eval(eval_dataset)
    print("result: ", res)
    logging.info("result: ", res)
