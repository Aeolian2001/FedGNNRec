from encrypt import *
import torch
import torch.nn as nn
import random
import numpy as np

# 第三方服务器
class tp_server():
    def __init__(self, embed_size, users, items, local_cipher_itemid):
        self.users = users
        self.items = items
        self.local_cipher_itemid = local_cipher_itemid
        self.user_embedding = torch.randn(len(users), embed_size)
        self.nei_list = self.matching(local_cipher_itemid)
    '''
    match neighbor and return relation_list
    '''
    # 匹配加密后项目ID
    def matching(self,local_cipher_itemid):
        local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'):j for j in range(len(self.items))}
        cipher2userid = {}
        for userid,i in enumerate(local_cipher_itemid):
            for j in i:
                if j not in cipher2userid:
                    cipher2userid[j] = [userid]
                else:
                    cipher2userid[j].append(userid)

        neighbor_list = []
        for i in range(len(self.users)):
            neighbor_list.append([])
        # 生成返回邻居表
        for itemid,nei in cipher2userid.items():
            for n in nei:
                neighbor_list[n] = neighbor_list[n]+nei
                neighbor_list[n] = list(set(neighbor_list[n]))
                # 前期用户嵌入不准，限制邻居数目
                random.shuffle(neighbor_list[n])
                temp = neighbor_list[n][0:25]
                neighbor_list[n] = temp
        return neighbor_list
    # 通过此方法向用户u提供匿名邻居嵌入
    def expanding(self, u):
            
        return torch.clone(self.user_embedding[torch.tensor(self.nei_list[u])]).detach()

    # 通过此方法更新(上传)用户嵌入
    def update_embedding(self, embedding, u):
        self.user_embedding[u] = embedding


    
