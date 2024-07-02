import torch
import numpy as np
import pickle
from sklearn import metrics
import math
import argparse
import faulthandler
import warnings
import sys
from client import *
from server import *
from encrypt import *
from utils import *
from tp_server import tp_server

faulthandler.enable()
warnings.filterwarnings('ignore')
# 超参数
parser = argparse.ArgumentParser(description="args for FedGNNRec")
parser.add_argument('--embed_size', type=int, default=16)
parser.add_argument('--lr', type=float, default = 1)
parser.add_argument('--data', default='ml_100k')
parser.add_argument('--user_batch', type=int, default=64)
parser.add_argument('--clip', type=float, default = 0.1)
parser.add_argument('--laplace_lambda', type=float, default = 0.1)
parser.add_argument('--pseudo_sample', type = int, default = 500)
parser.add_argument('--valid_step', type = int, default = 20)
args = parser.parse_args()

embed_size = args.embed_size
user_batch = args.user_batch
lr = args.lr

# 手动输入数据集并读取数据
path_dataset =  'ml_100K.mat'
M = load_matlab_file(path_dataset, 'M')
train_data = load_matlab_file(path_dataset, 'Otraining')
valid_data = load_matlab_file(path_dataset, 'Ovalid')
test_data = load_matlab_file(path_dataset, 'Otest')
print('There are %i interactions logs.'%np.sum(np.array(np.array(M,dtype='bool'),dtype='int32')))

# 用户和项目表
user_id_list = []
for u in range(train_data.shape[0]):
    user_id_list.append(u)
item_id_list = []
for u in range(train_data.shape[1]):
    item_id_list.append(u)

# 预处理数据，生成用户交互历史等
print('---Preprocess---')
test_data = generate_test_data(test_data,M)
valid_data = generate_test_data(valid_data,M)
history = generate_history(train_data, M)
interaction = generate_interaction(train_data, M)

# 加密
# 若进行快速测试可取消加密
generate_key()
print('---Encrypt---')
local_cipher_itemid = []
for i in tqdm(history):
    messages = []
    for j in i:
        # messages.append(j)
        messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
    local_cipher_itemid.append(messages)

# 假设客户端向第三方服务器上传
print('---Mapping---')
tp_server = tp_server(embed_size, user_id_list, item_id_list, local_cipher_itemid)

# 客户端
user_list = []
for u in range(train_data.shape[0]):
    ratings = interaction[u]
    items = []
    rating = []
    for i in range(len(ratings)):
        item, rate = ratings[i]
        items.append(item)
        rating.append(rate)
    user_list.append(client(u, items, rating, list(tp_server.nei_list[u]), embed_size, args.clip, args.laplace_lambda, args.pseudo_sample, tp_server))


# 服务器
server = server(user_list, user_batch, user_id_list, item_id_list, embed_size, lr)
count = 0

# 训练 测试 评估
# 五次测试无优化后训练结束
rmse_best = 20
while 1:
    for i in range(args.valid_step):
        print(i)
        server.train()
    print('valid')
    mae, rmse = loss(server, valid_data)
    print('valid mae: {}, valid rmse:{}'.format(mae, rmse))
    if rmse < rmse_best:
        rmse_best = rmse
        count = 0
        mae_test, rmse_test = loss(server, test_data)
    else:
        count += 1
    if count > 5:
        print('not improved for 5 epochs, stop trianing')
        break
print('final test mae: {}, test rmse: {}'.format(mae_test, rmse_test))