# FedGNNRec

一个基于图神经网络的隐私保护联邦推荐框架

<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]
 
## 目录

- [上手指南](#上手指南)
  - [环境配置](#环境配置)
  - [安装步骤](#安装步骤)
- [文件目录](#文件目录)
- [架构](#架构)
- [运行](#运行)

### 上手指南

###### 环境配置

1. Ubuntu       18.04
2. python       3.7.11
3. numpy        1.25.2
4. pytorch      1.7.1
5. dgl          2.1.0
6. pycryptodome 3.12.0

###### **安装步骤**
根据需求配置环境，若有问题，则：
* pycryptodome可能与crypto冲突，若出现此情况请卸载其中一个
* 请根据报错请自行安装所需包或调配环境
* 可在colab平台上运行，需要安装dgl和pycryptodom相关库


### 文件目录
```
filetree 
├── README.md
├── ml_100k.mat
├── requirements.txt
├── dataset
│  ├── db
│  │  ├── douban.mat
│  │  ├── ...
│  │  └── douban_5.mat
│  ├── fl
│  │  ├── flixster.mat
│  │  ├── ...
│  │  └── flixste_5.mat
│  ├── ml
│  │  ├── ml_100k_2.mat
│  │  ├── ...
│  │  └── ml_100k_5.mat
├── client.py
├── tp_server.py
├── encrypt.py
├── main.py
├── model.py
├── utils.py
└── server.py
```


### 架构 
* server:服务器，主要属性包括用户表user_list、物品表item_list、物品嵌入表item_embedding和全局模型model等，重要方法有聚合算法aggregator()和训练方法train()
* client:用户设备，主要属性包括用户IDself_id,本地图graph和本地模型model等，重要方法有本地差分隐私LDP()、伪随机抽样pseudo_sample_item()和训练方法train()
* tp_server:第三方服务器，主要属性包括用户邻接表nei_list和用户嵌入表user_embedding，主要方法有匹配邻居算法matching()、分发嵌入expanding()和梯度更新update_embedding()
```

```


### 运行
1. 下载ML-100K、Douban、Flixster数据集并进行处理，或者直接使用文件中提供的已处理的数据集
2. 选取超参数和模型
3. 运行 main.py 文件

注：
* 由于代码编写问题，更换模型和测试部分算法的超参数需要在代码中修改函数参数或自行启用(禁用)某些代码行
* 请保证Linux服务器的内存和CPU性能满足硬件要求







