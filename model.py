import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.conv import SAGEConv

# GAT in dgl
class model(nn.Module):
    def __init__(self, embed_size, head_num):
        super().__init__()
        self.GAT_layer = GATConv(embed_size, embed_size, head_num)

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def forward(self, graph, features_in, item_index):
        features = self.GAT_layer(graph, features_in)
        n = features.shape[0]
        features = features.reshape(n, -1)
        user_embedding = features[0, :]
        return user_embedding

# GCN model also based on dgl
'''
class model(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.GCN_layer = GraphConv(embed_size, embed_size)

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def forward(self, graph, features_in):
        features = self.GCN_layer(graph, features_in)
        n = features.shape[0]
        features = features.reshape(n, -1)
        user_embedding = features[0, :]
        return user_embedding
'''

# GraphSage model also based on dgl
'''
class model(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.SAGE_layer = SAGEConv(embed_size, embed_size, 'pool')

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def forward(self, graph, features_in):
        features = self.SAGE_layer(graph, features_in)
        n = features.shape[0]
        features = features.reshape(n, -1)
        user_embedding = features[0, :]
        return user_embedding
'''