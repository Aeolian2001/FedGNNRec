import torch
import os
import numpy as np
import torch.nn as nn
import dgl
from random import sample
from multiprocessing import Pool, Manager
from model import model
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')

class server():
    def __init__(self, user_list, user_batch, users, items, embed_size, lr):
        self.user_list_with_coldstart = user_list
        self.user_list = self.generate_user_list(self.user_list_with_coldstart)
        self.batch_size = user_batch
        self.user_embedding = torch.randn(len(users), embed_size).share_memory_()
        self.item_embedding = torch.randn(len(items), embed_size).share_memory_()
        self.model = model(embed_size, 1)
        self.lr = lr
        self.distribute(self.user_list_with_coldstart)

    def generate_user_list(self, user_list_with_coldstart):
        ls = []
        for user in user_list_with_coldstart:
            if len(user.items) > 0:
                ls.append(user)
        return ls
    # 聚合函数
    # 考虑用户交互数目的影响，加权计算用户聚合的梯度
    def aggregator(self, parameter_list):
        flag = False
        number = 0
        gradient_item = torch.zeros_like(self.item_embedding)
        gradient_user = torch.zeros_like(self.user_embedding)
        for parameter in parameter_list:
            [model_grad, item_grad, user_grad, returned_items, returned_users] = parameter
            #
            num = len(returned_items)
            number += num
            if not flag:
                flag = True
                gradient_model = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i] * num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i] * num
        gradient_item /= number
        gradient_user /= number
        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number
        return gradient_model, gradient_item, gradient_user
    # 分发模型
    def distribute(self, users):
        for user in users:
            user.update_local_GNN(self.model)

    def distribute_one(self, user):
        user.update_local_GNN(self.model)

    def get_embedding(self, embedding, user):
        return embedding[torch.tensor(user)]

    def predict(self, valid_data):
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []
        self.distribute([self.user_list_with_coldstart[i] for i in set(users)])
        for i in range(len(users)):
            res_temp = self.user_list_with_coldstart[users[i]].predict(items[i], self.user_embedding, self.item_embedding)
            res.append(float(res_temp))
        return np.array(res)


    def train(self):
        parameter_list = []
        users = sample(self.user_list, self.batch_size)
        # 分发模型
        self.distribute(users)
        # 客户端训练
        for user in users:
            u_embedding = self.get_embedding(self.user_embedding, user.id_self)
            parameter_list.append(user.train(u_embedding, self.item_embedding))

        # 聚合
        gradient_model, gradient_item, gradient_user = self.aggregator(parameter_list)

        ls_model_param = list(self.model.parameters())

        # 更新
        for i in range(len(ls_model_param)):
            ls_model_param[i].data = ls_model_param[i].data - self.lr * gradient_model[i]
        self.item_embedding = self.item_embedding -  self.lr * gradient_item
        self.user_embedding = self.user_embedding -  self.lr * gradient_user