import torch
import copy
from random import sample
import torch.nn as nn
import numpy as np
import dgl
import pdb
from model import model

class client():
    def __init__(self, id_self, items, ratings, neighbors, embed_size, clip, laplace_lambda, pseudo_sample, tp_server):
        self.pseudo_sample = pseudo_sample
        self.clip = clip
        self.laplace_lambda = laplace_lambda
        self.id_self = id_self
        self.items = items
        self.embed_size = embed_size
        self.ratings = ratings
        self.neighbors = neighbors
        self.model = model(embed_size, 1)
        self.graph = self.build_local_graph(id_self, items, neighbors)
        self.graph = dgl.add_self_loop(self.graph)
        self.user_feature = torch.randn(self.embed_size)
        self.tp_server = tp_server

    def build_local_graph(self, id_self, items, neighbors):
        G = dgl.DGLGraph()
        dic_user = {self.id_self: 0}
        dic_item = {}
        count = 1
        for n in neighbors:
            dic_user[n] =  count
            count += 1
        for item in items:
            dic_item[item] = count
            count += 1
        G.add_edges([i for i in range(1, len(dic_user))], 0)
        G.add_edges(list(dic_item.values()), 0)
        G.add_edges(0, 0)
        return G

    def user_embedding(self, embedding):
        return embedding[torch.tensor(self.id_self)]

    def item_embedding(self, embedding):
        return embedding[torch.tensor(self.items)]

    def GNN(self, embedding_user, embedding_item):
        self.tp_server.update_embedding(torch.clone(embedding_user).detach(),self.id_self)
        neighbor_embedding = self.tp_server.expanding(self.id_self)
        neighbor_embedding.requires_grad = True
        items_embedding = self.item_embedding(embedding_item)
        print
        features =  torch.cat((embedding_user.unsqueeze(0), neighbor_embedding, items_embedding), 0)
        user_feature = self.model(self.graph, features, len(self.neighbors) + 1)
        self.user_feature = user_feature.detach()
        predicted = torch.matmul(user_feature, items_embedding.t())
        return predicted

    def update_local_GNN(self, global_model):
        self.model = copy.deepcopy(global_model)

    def loss(self, predicted):
        # mse
        return torch.mean((predicted - torch.tensor(self.ratings))**2)

    def predict(self, item_id, embedding_user, embedding_item):
        item_embedding = embedding_item[item_id]
        return torch.matmul(self.user_feature, item_embedding.t())

    def pseudo_sample_item(self, grad):
        # 筛选未交互项目
        item_num, embed = grad.shape
        ls = [i for i in range(item_num) if i not in self.items]
        sampled_items = sample(ls, self.pseudo_sample)
        grad_value = torch.masked_select(grad, grad != 0)
        # 生成了一个标准正态分布的随机张量，并对其进行了缩放和平移，以匹配所需的均值和标准差。
        mean = torch.mean(grad_value)
        var = torch.std(grad_value)
        grad[torch.tensor(sampled_items)] += torch.randn((len(sampled_items), self.embed_size)) * var + mean

        returned_items = sampled_items + self.items
        return returned_items

    def LDP(self, tensor):
        # 裁剪
        tensor = torch.clamp(tensor, min=-self.clip, max=self.clip)
        # 加噪
        loc = torch.zeros_like(tensor)
        scale = torch.ones_like(tensor) * self.laplace_lambda
        # torch.distributions.laplace.Laplace()拉普拉斯分布
        tensor = tensor + torch.distributions.laplace.Laplace(loc, scale).sample()
        return tensor

    def train(self, embedding_user, embedding_item):
        embedding_user = torch.clone(embedding_user).detach()
        embedding_item = torch.clone(embedding_item).detach()
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        predicted = self.GNN(embedding_user, embedding_item)
        loss = self.loss(predicted)
        self.model.zero_grad()
        # 使用retain_graph=True 会影响训练速度
        # 
        loss.backward()
        model_grad = []
        for param in list(self.model.parameters()):
            grad = self.LDP(param.grad)
            model_grad.append(grad)
        returned_items = self.pseudo_sample_item(embedding_item.grad)
        item_grad = self.LDP(embedding_item.grad[returned_items, :])
        returned_user = self.id_self
        user_grad = self.LDP(embedding_user.grad)
        res = (model_grad, item_grad, user_grad, returned_items, returned_user)
        return res