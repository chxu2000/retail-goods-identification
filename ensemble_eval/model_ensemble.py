import os
import random
import argparse
import logging
import mindspore
import numpy as np
from mindspore import dtype as mstype
import mindspore_hub as mshub
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.py_transforms as P
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspore.nn import Momentum, Adam, AdamWeightDecay, SGD
from mindspore import Model, context
from mindspore import common, Parameter
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import load_checkpoint, load_param_into_net
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from PIL import Image
from tqdm import tqdm

from src.resnet import resnet50

from src.config_efficientnet import dataset_config
from src.efficientnet import efficientnet_b0
from src.loss import LabelSmoothingCrossEntropy

from src.config_shufflenetv2 import config_gpu as cfg_shufflenetv2
from src.dataset import create_dataset as create_dataset_snv2
from src.shufflenetv2 import ShuffleNetV2
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--device_target', type=str, default='GPU', help='Device choice Ascend or GPU')
parser.add_argument('--train_dataset_path', type=str, default='/home/twang/BDCI2021/train', help='Dataset path.')
parser.add_argument('--test_dataset_path', type=str, default='/home/twang/BDCI2021/test', help='Dataset path.')
args_opt = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(1)
# context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

model_num = 5
batch_size = 128
epoch_size = 100
num_classes = 2388
dataset_type = 'bdci'
cfg_efb0 = dataset_config[dataset_type].cfg

checkpoint_path = ['' for _ in range(model_num)]
best_checkpoint_path = ['' for _ in range(model_num)]
net = [0 for _ in range(model_num)]
ls = [0 for _ in range(model_num)]
opt = [0 for _ in range(model_num)]
model = [0 for _ in range(model_num)]
eval_dataset = [0 for _ in range(model_num)]
param_dict = [0 for _ in range(model_num)]
res = [0 for _ in range(model_num)]
best_acc = [0 for _ in range(model_num)]
correct = [0 for _ in range(model_num)]
total = [0 for _ in range(model_num)]
acc = [0 for _ in range(model_num)]


def softmax(y_pred):
	max = np.max(y_pred)
	softmax = np.exp(y_pred-max) / sum(np.exp(y_pred-max))
	return softmax


def create_dataset(repeat_num=1, training=True):
	if training:
		assert os.path.exists(args_opt.train_dataset_path), "the dataset path is invalid!"
		cifar_ds = ds.ImageFolderDataset(args_opt.train_dataset_path, decode=True, shuffle=False)
	else:
		assert os.path.exists(args_opt.test_dataset_path), "the dataset path is invalid!"
		cifar_ds = ds.ImageFolderDataset(args_opt.test_dataset_path, decode=True, shuffle=False)

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
    
    # Declare an apply_func function which returns a Dataset object
	def apply_func(data):
		data = data.batch(batch_size, True)
		return data
	cifar_ds = cifar_ds.apply(apply_func)
	return cifar_ds

# checkpoint path
prefix = ['' for _ in range(model_num)]
prefix[0] = '../resnet50_transfer/ckpt/'
prefix[1] = '../efficientnetb0/ckpt/'
prefix[2] = '../shufflenetv2_transfer/ckpt/'
prefix[3] = '../efficientnetb0_transfer/ckpt/'
prefix[4] = '../resnet50_transfer_mixup_ls/ckpt/'

except_model = []

print('========== Choosing Model ==========')

for i in range(model_num):
	for ckpt in os.listdir(prefix[i]):
		if not ckpt.endswith('.ckpt'):
			continue
		checkpoint_path[i] = prefix[i] + ckpt
		print('========== ', checkpoint_path[i], ' ==========')
		if i == 0 or i == 4:
			net[i] = resnet50(num_classes)
		elif i == 1 or i == 3:
			net[i] = efficientnet_b0(num_classes=2388,
									cfg=dataset_config[dataset_type],
									drop_rate=cfg_efb0.drop,
									drop_connect_rate=cfg_efb0.drop_connect,
									global_pool=cfg_efb0.gp,
									bn_tf=cfg_efb0.bn_tf,
									)
		elif i == 2:
			net[i] = ShuffleNetV2()
		net[i].set_train(False)
		ls[i] = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
		opt[i] = Momentum(filter(lambda x: x.requires_grad, net[i].get_parameters()), 0.01, 0.9)
		model[i] = Model(net[i], loss_fn=ls[i], optimizer=opt[i], metrics={'acc', 'predict_matrix'})
		eval_dataset[i] = create_dataset(training=False)
		
		param_dict[i] = load_checkpoint(checkpoint_path[i])
		load_param_into_net(net[i], param_dict[i])
		tmp_res = model[i].eval(eval_dataset[i])
		print('acc: ', tmp_res['acc'])
		if float(tmp_res['acc']) > best_acc[i]:
			best_checkpoint_path[i] = checkpoint_path[i]
			best_acc[i] = float(tmp_res['acc'])
			res[i] = tmp_res

print('========== Single Model Accuracy ==========')
for i in range(model_num):
	print('acc', i, ': ', res[i]['acc'])

# get normalization factor
print('========== Calc Normalization Factor ==========')

train_res = [0 for _ in range(model_num)]

for i in range(model_num):
	if i == 0 or i == 4:
		net[i] = resnet50(num_classes)
	elif i == 1 or i == 3:
		net[i] = efficientnet_b0(num_classes=2388,
								cfg=dataset_config[dataset_type],
								drop_rate=cfg_efb0.drop,
								drop_connect_rate=cfg_efb0.drop_connect,
								global_pool=cfg_efb0.gp,
								bn_tf=cfg_efb0.bn_tf,
								)
	elif i == 2:
		net[i] = ShuffleNetV2()
	net[i].set_train(False)
	ls[i] = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
	opt[i] = Momentum(filter(lambda x: x.requires_grad, net[i].get_parameters()), 0.01, 0.9)
	model[i] = Model(net[i], loss_fn=ls[i], optimizer=opt[i], metrics={'acc', 'predict_matrix'})
	eval_dataset[i] = create_dataset(training=True)
	
	param_dict[i] = load_checkpoint(checkpoint_path[i])
	load_param_into_net(net[i], param_dict[i])
	
	train_res[i] = model[i].eval(eval_dataset[i])['predict_matrix']
	correct[i] = [0 for _ in range(2388)]
	total[i] = [0 for _ in range(2388)]
	acc[i] = [0 for _ in range(2388)]

	for j in tqdm(range(len(train_res[i]))):
		total[i][train_res[i][j][1]] += 1
		if (np.argmax(train_res[i][j][0]) == train_res[i][j][1]):
			correct[i][train_res[i][j][1]] += 1
	for j in range(2388):
		if (total[i][j] > 0):
			acc[i][j] = correct[i][j] / total[i][j]
		else:
			acc[i][j] = 0

print('========== Calcing Ensemble Accuracy ==========')

predict_matrix = [0 for _ in range(model_num)]
for i in range(model_num):
	predict_matrix[i] = res[i]['predict_matrix']
total_num = len(predict_matrix[0])
acc_list = []
total_acc = [0 for _ in range(2388)]
for i in range(2388):
	for j in range(model_num):
		if j not in except_model:
			total_acc[i] += acc[j][i]

correct_num = 0

for i in tqdm(range(total_num)):
	ans = predict_matrix[0][i][1]
	prob_array = [0 for _ in range(model_num)]
	for j in range(model_num):
		if j not in except_model:
			prob_array[j] = softmax(predict_matrix[j][i][0])

	# 概率平均
	# print(prob_array_1)
	# if (np.max(prob_array_1) < 0.995 and np.max(prob_array_2) < 0.995):
	# 	result = 21
	# else:
	# 	prob_array = (p / 10) * prob_array_1 + (1 - p / 10) * prob_array_2
	# 	result = np.where(prob_array == np.max(prob_array))
	# prob_array_t1 = prob_array_1 + prob_array_2 + prob_array_3 + prob_array_4 + prob_array_5
	# result_t1 = np.argmax(prob_array_t1)
	# if (result_t1 == ans):
	# 	correct_num_1 = correct_num_1 + 1

	# 归一化参数
	prob_array_t = [0 for _ in range(2388)]
	for j in range(2388):
		if (total_acc[j] > 0):
			for k in range(model_num):
				if k not in except_model:
					prob_array_t[j] += acc[k][j] * prob_array[k][j]
			prob_array_t[j] /= total_acc[j]
		else:
			for k in range(model_num):
				if k not in except_model:
					prob_array_t[j] += prob_array[k][j]
			prob_array_t[j] /= total_num - len(except_model)
	result_t = np.argmax(prob_array_t)
	if (result_t == ans):
		correct_num = correct_num + 1

print('========== Model Ensemble Accuracy ==========', correct_num / total_num)