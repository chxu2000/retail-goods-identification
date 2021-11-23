# MindSpore框架实现零售商品识别

本模型集成了5个高性能cv网络，在rp2k数据集上达到了97.6%的准确率。

## 运行方式

```bash
cd model_zoo/scripts
bash download_pretrain_ckpt.sh
bash train_gpu.sh
bash eval_gpu.sh
```

## 源码框架

.

├── ckpt_transfer						//mshub预训练ckpt
├── efficientnetb0						//无预训练EfficientNet-B0网络
│  ├── ckpt							//保存ckpt的路径		
│  ├── scripts							
│  │  ├── run_train_ascend.sh			// Ascend单卡训练shell脚本
│  │  └── run_train_gpu.sh			// GPU单卡训练shell脚本
│  ├── src							
│  │  ├── config.py					//超参配置
│  │  ├── dataset.py					//数据集处理
│  │  ├── efficientnet.py				//网络结构
│  │  ├── loss.py					//损失函数
│  │  ├── transform.py				//数据增强
│  │  └── transform_utils.py			
│  └── train.py						//训练脚本
├── efficientnetb0_transfer				//带预训练EfficientNet-B0网络
│  ├── ckpt							//保存ckpt的路径	
│  ├── scripts
│  │  ├── run_train_ascend.sh			// Ascend单卡训练shell脚本
│  │  └── run_train_gpu.sh			// GPU单卡训练shell脚本
│  ├── src
│  │  ├── config.py					//超参配置
│  │  ├── dataset.py					//数据集处理
│  │  ├── efficientnetb0.py			//网络结构
│  │  ├── efficientnet.py		
│  │  ├── loss.py					//损失函数
│  │  ├── transform.py				//数据增强
│  │  └── transform_utils.py
│  └── train.py						//训练脚本
├── ensemble_eval						//模型集成
│  ├── model_ensemble.py				//模型集成脚本
│  ├── scripts
│  │   ├── run_ensemble_eval_ascend.sh
│  │   └── run_ensemble_eval_gpu.sh
│  └── src
│    ├── config_efficientnet.py		//超参配置
│    ├── config_shufflenetv2.py
│    ├── crossentropy.py				//交叉熵损失函数
│    ├── CrossEntropySmooth.py
│    ├── dataset.py					//数据集处理
│    ├── efficientnet.py				//efficientnet网络
│    ├── loss.py					//损失函数
│    ├── lr_generator.py				//学习率生成
│    ├── resnet.py					//resnet网络
│    └── shufflenetv2.py				//shufflenet网络
├── modified_repo_files					//模型集成库函数修改
│  ├── __init__.py
│  └── predict_matrix.py
├── README.md						//集成模型具体说明
├── resnet50_transfer					//不带mixup/label smooth的预训练resNet-50
│  ├── ckpt
│  ├── scripts
│  │  ├── run_train_ascend.sh			//Ascend单卡训练shell脚本
│  │  └── run_train_gpu.sh			//GPU单卡训练shell脚本
│  ├── src
│  │  └── resnet_transfer.py			//网络结构
│  └── train.py						//训练脚本
├── resnet50_transfer_mixup_ls		//带有mixup/label smooth的预训练resNet-50
│  ├── ckpt						//保存ckpt
│  ├── scripts			
│  │  ├── run_train_ascend.sh		//Ascend单卡训练shell脚本
│  │  └── run_train_gpu.sh		//GPU单卡训练shell脚本
│  ├── src
│  │  └── resnet_transfer.py		//网络结构
│  └── train.py					//训练脚本
├── scripts
│  ├── eval_ascend.sh				//Ascend 评估shell脚本
│  └── eval_gpu.sh				//GPU评估shell脚本
│  └── train_ascend.sh				//Ascend单卡训练shell脚本
│  └── train_gpu.sh				//GPU单卡训练shell脚本
│  └── download_pretrain_ckpt.sh	//ckpt下载shell脚本
└── shufflenetv2_transfer			//带预训练的ShuffleNet-v2
  ├── ckpt						//保存ckpt
  ├── scripts
  │  ├── run_train_ascend.sh		//Ascend单卡训练shell脚本
  │  └── run_train_gpu.sh		//GPU单卡训练shell脚本
  ├── src
  │  └── shufflenetv2_transfer.py	//网络结构
  └── train.py					//训练脚本

 

 