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
import numpy as np
import logging
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspore.nn import Momentum, Adam, AdamWeightDecay
from mindspore import Model, context, Tensor
from mindspore import common
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import load_checkpoint, load_param_into_net

import mindspore_hub as mshub
from mindspore import Parameter, Tensor, numpy as np


from resnet import resnet50
import cv2
from mindspore.dataset.transforms.c_transforms import Compose
from PIL import Image

data_home = "/home/twang/BDCI2021/test"
parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--epoch_size', type=int, default=100, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=2388, help='Num classes.')
parser.add_argument('--device_target', type=str, default='GPU', help='Device choice Ascend or GPU')
parser.add_argument('--do_train', type=bool, default=False, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=True, help='Do eval or not.')
parser.add_argument('--checkpoint_path', type=str, default='/home/twang/BDCI2021/checkpoint/ckpt_transfer/train_resnet_transfer_1-1_2694.ckpt', help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default=data_home, help='Dataset path.')
# init("nccl")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args_opt = parser.parse_args()
data_home = args_opt.dataset_path
context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

def TTA(model):

    def create_label():
        label = []
        folders = os.listdir(data_home)
        folders.sort()
        idx = 0
        for folder in folders:
            files = os.listdir(os.path.join(data_home, folder))
            label = label + [idx for i in range(len(files))]
            for lln in files:
                print(lln)
            idx = idx + 1 
        label = np.array(label)
        return label

    random_vertical_op = C.RandomVerticalFlip(prob=0.5)
    random_horizontal_op = C.RandomHorizontalFlip(prob=0.5)
    random_rotation_op = C.RandomRotation(degrees = 15)
    rescale_op = C.Rescale(rescale = 1.0/255.0, shift = 0.0)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()

    images_dataset = []
    folders = os.listdir(data_home)
    folders.sort()
    for folder in folders:
        file_name = os.path.join(data_home,folder)
        for file in os.listdir(file_name):
            img = Image.open(os.path.join(file_name,file)).convert("RGB")
            res = img.resize((224,224))
            images_dataset.append(np.array(res , dtype = np.float32))

    def compose1(x):
        x = random_vertical_op(x)
        x = changeswap_op(normalize_op(rescale_op(x)))
        return x

    def compose2(x):
        x = random_horizontal_op(x)
        x = changeswap_op(normalize_op(rescale_op(x)))
        return x

    def compose3(x):
        x = random_rotation_op(x)
        x = changeswap_op(normalize_op(rescale_op(x)))
        return x


    def slice(images, batch_num):
        res1 = []
        res2 = []
        res3 = []
        l = len(images)
        stride = int(l / batch_num)
        if (stride > 0):
            for i in range(0, l, stride):
                result1 = model.predict(Tensor(np.array([compose1(img) for img in images[i:i+stride]])))
                result2 = model.predict(Tensor(np.array([compose2(img) for img in images[i:i+stride]])))
                result3 = model.predict(Tensor(np.array([compose3(img) for img in images[i:i+stride]])))
                res1 = res1 + list(result1)
                res2 = res2 + list(result2)
                res3 = res3 + list(result3)
        if (stride*batch_num<l):
            result1 = model.predict(Tensor(np.array([compose1(img) for img in images[stride*batch_num:l]])))
            result2 = model.predict(Tensor(np.array([compose2(img) for img in images[stride*batch_num:l]])))
            result3 = model.predict(Tensor(np.array([compose3(img) for img in images[stride*batch_num:l]])))
            res1 = res1 + list(result1)
            res2 = res2 + list(result2)
            res3 = res3 + list(result3)
        return ((np.array(res1) + np.array(res2) + np.array(res3)) / 3.0)

    result = np.argmax(slice(images_dataset, 32), axis = 1)
    label = create_label()
    count = 0

    print(label)
    print(result)

    for i in range(len(label)):
        if result[i] == label[i]:
            count = count + 1
        else:
            pass

    acc = count/len(label)  
    print("result: ", acc)

def create_dataset(repeat_num=1, training=True):
    assert os.path.exists(data_home), "the dataset path is invalid!"
    if args_opt.run_distribute:
        rank_id = 0
        rank_size = 4
        cifar_ds = ds.ImageFolderDataset(data_home, decode=True, shuffle=True, num_shards=rank_size, shard_id=rank_id)
    else:
        # cifar_ds = ds.ImageFolderDataset(data_home, decode=True, shuffle=True)
        cifar_ds = ds.ImageFolderDataset(data_home, decode=True, shuffle=False)

    resize_height = 224
    resize_width = 224
    
    rescale = 1.0 / 255.0
    shift = 0.0
    random_horizontal_op = C.RandomHorizontalFlip()
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift)
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

#############################
    # model_name = "mindspore/ascend/1.2/resnet50thorcp_v1.2_imagenet2012"
    # net = mshub.load(model_name, class_num=2388, force_reload=False)
    # ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)

    # param_dict = load_checkpoint("/home/twang/BDCI2021/checkpoint/ckpt_transfer/train_resnet_transfer-40_2694.ckpt")
    # modWeight = Parameter(Tensor(np.ones((2388, 2048)), mstype.float32), name="moments.end_point.weight", requires_grad=True)
    # modBias = Parameter(Tensor(np.ones((2388)), mstype.float32), name="moments.end_point.bias", requires_grad=True)
    # param_dict['end_point.weight']=modWeight
    # param_dict['end_point.bias']=modBias
    # load_param_into_net(net, param_dict)
    # net.set_train(True)

    # model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})
#############################






    # 原版测试方法
    eval_dataset = create_dataset(training=False)
    res = model.eval(eval_dataset)
    print("result: ", res)
    logging.info("result: ", res)

    # TTA测试方法
    # TTA(model)
