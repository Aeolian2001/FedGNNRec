import h5py
import torch.nn as nn
import torch
import numpy as np
from encrypt import *
from tp_server import tp_server

path_dataset =  'ml_100K.mat'
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
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out

def generate_test_data(Otest,M):
    test_data=[]
    for i in range(Otest.shape[0]):
        for j in range(len(Otest[i])):
            if Otest[i][j]!=0:
                test_data.append((i,j,int(M[i][j])))
    return np.array(test_data)
# preprocess data
# interactions->(item + rate)
def generate_interaction(Otraining, M):
    interaction=[]
    for i in range(Otraining.shape[0]):
        user_interaction=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0:
                user_interaction.append([j, int(M[i][j])])
        interaction.append(user_interaction)
    return interaction
# history of items
def generate_history(Otraining, M):
    history=[]
    for i in range(Otraining.shape[0]):
        user_history=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0:
                user_history.append(j)
        history.append(user_history)
    return history

M = load_matlab_file(path_dataset, 'M')
train_data = load_matlab_file(path_dataset, 'Otraining')
valid_data = load_matlab_file(path_dataset, 'Ovalid')
test_data = load_matlab_file(path_dataset, 'Otest')
print('There are %i interactions logs.'%np.sum(np.array(np.array(M,dtype='bool'),dtype='int32')))

user_id_list = []
for u in range(train_data.shape[0]):
    user_id_list.append(u)

item_id_list = []
for u in range(train_data.shape[1]):
    item_id_list.append(u)

test_data = generate_test_data(test_data,M)
valid_data = generate_test_data(valid_data,M)
history = generate_history(train_data, M)
interaction = generate_interaction(train_data, M)

generate_key()

alluser_embedding = torch.randn(len(users), embed_size)

local_cipher_itemid = []
for i in tqdm(history):
    text_count = 0
    messages = []
    for j in i:
        messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        text_count += 1
        if text_count > 30:
            break
    local_cipher_itemid.append(messages)

#local id-ciphertext mapping
local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'):j for j in range(train_data.shape[1])}
    
cipher2userid = {}
for userid,i in enumerate(local_cipher_itemid):
    for j in i:
        if j not in cipher2userid:
            cipher2userid[j] = [userid]
        else:
            cipher2userid[j].append(userid)

send_data = []
for userid,i in tqdm(enumerate(local_cipher_itemid)):
    neighbor_info={}
    for j in i:
        neighbor_id = [alluserembs[uid] for uid in cipher2userid[j]]
        if len(neighbor_id):
            neighbor_info[j] = neighbor_id
    send_data.append(neighbor_info)

