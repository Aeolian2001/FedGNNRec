import random
import numpy as np
import h5py
import math


def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)

# 加载数据 xxx.mat
def load_matlab_file(path_file, name_field):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            print(jc)
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out


# 处理得到测试和验证数据集，具有相同结构，共用此方法
def generate_test_data(Otest,M):
    test_data=[]
    for i in range(Otest.shape[0]):
        for j in range(len(Otest[i])):
            if Otest[i][j]!=0:
                test_data.append((i,j,int(M[i][j])))
    return np.array(test_data)


# 生成交互
# [项目ID，评分]
# 用户ID顺序表示
def generate_interaction(Otraining, M):
    interaction=[]
    for i in range(Otraining.shape[0]):
        user_interaction=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0:
                user_interaction.append([j, int(M[i][j])])
        interaction.append(user_interaction)
    return interaction


# 用户历史
# 只包含项目id
def generate_history(Otraining, M):
    history=[]
    for i in range(Otraining.shape[0]):
        user_history=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0:
                user_history.append(j)
        history.append(user_history)
    return history


# 计算RMSE和MAE
def loss(server, valid_data):
    label = valid_data[:, -1]
    predicted = server.predict(valid_data)
    mae = sum(abs(label - predicted)) / len(label)
    rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))
    return mae, rmse