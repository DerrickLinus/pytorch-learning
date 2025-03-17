# PyTorch Learning

这是一个 PyTorch 入门学习项目，用于记录学习 PyTorch 深度学习框架的从0到1过程。

## 参考资料
【小土堆】PyTorch深度学习快速入门教程

https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=3dd871ff46120819eca6181d0acc32dd

## 数据集

hymenoptera_data：蚂蚁、蜜蜂分类练手数据集  
CIFAR10：广泛用于计算机视觉和深度学习领域的经典数据集

hymenoptera_data链接: https://pan.baidu.com/s/19HF4CNejn2n5WlV9ijXXJw?pwd=3k3h 提取码: 3k3h 

## 环境配置

我的环境：
- Python 3.10
- PyTorch
- GPU
- CUDA Version:12.6

环境配置教程：

【小土堆Pytorch环境配置】
https://www.bilibili.com/video/BV1S5411X7FY/?spm_id_from=333.337.search-card.all.click&vd_source=3dd871ff46120819eca6181d0acc32dd

## 学习目录
- `read_data.py`：Pytorch加载数据
- `rename_dataset.py`：获取数据集标签，创建label文件夹
- `tensorboard.py`：TensorBoard可视化工具的使用
- `transforms.py`：Transforms的使用
- `UsefulTransforms.py`：常见的Transforms
- `dataset_torchvision.py`：torchvision中的数据集使用
- `dataloader.py`：学习 PyTorch 数据加载器的使用
- `nn_module.py`：神经网络基本骨架nn.module的使用
- `nn_conv_base.py`：基本卷积操作
- `nn_conv2d.py`：卷积层
- `nn_maxpool.py`：最大池化层
- `nn_relu.py`：非线性激活
- `nn_linear.py`：线性层
- `nn_seq.py`：神经网络搭建小实战和Sequential的使用
- `nn_loss.py`：损失函数的定义及公式
- `nn_loss_network.py`：损失函数在神经网路中的使用
- `nn_optim.py`：优化器
- `model_pretrained.py`：现有网络模型的使用及修改
- `model_save.py`、`model_load.py`：网络模型的保存与读取
- `train.py`：完整的模型训练套路
- `train_gpu_1.py`：利用GPU训练（一）
- `train_gpu_2.py`：利用GPU训练（二）
- `train_gpu_improve.py`：模型改进与参数优化
- `test.py`：完整的模型验证套路