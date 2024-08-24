import pandas as pd
import os
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

data = pd.read_csv("double_checked_data_1_200.csv")
labels = pd.read_csv('labels.csv')
n_classes = 200
signers = []
for group_id, group_data in data.groupby("id"):
    if group_data.iloc[0]['label_id'] < n_classes:
        name = f"signer{group_data.iloc[0]['implementer'].split('_')[0]}"
        if name not in signers:
            signers.append(name)

train = 0.7
val = 0.15
test = 0.15

random.shuffle(signers)

train_signers = signers[:int(train*len(signers))]
val_signers = signers[int(train*len(signers)):int(train*len(signers))+int(val*len(signers))]
test_signers = signers[int(train*len(signers))+int(val*len(signers)):]


"""
train signers
['signer28', 'signer_11', 'signer06', 'signer30', 'signer29', 'signer02', 'signer05', 'signer17', 'signer01', 'signer07-Phu', 'signer09', 'signer15', 'signer22', 'signer18', 
'signer24', 'signer20', 'signer14', 'signer19', 'signer23', 'signer13', 'signer26']

val signers
['signer27', 'signer08', 'signer31', 'signer03']

test signers
['signer04', 'signer16', 'signer12', 'signer10', 'signer21']
"""
# 21,4,5,30
print(len(train_signers),len(val_signers),len(test_signers),len(signers))

train_csv = labels[(labels['name'].str.contains('|'.join(train_signers))) & (labels['label'] < n_classes) ]
test_csv = labels[(labels['name'].str.contains('|'.join(test_signers))) & (labels['label'] < n_classes)]
val_csv = labels[(labels['name'].str.contains('|'.join(val_signers))) & (labels['label'] < n_classes)]

full_data = labels[(labels['name'].str.contains(f"{'|'.join(train_signers)}|{'|'.join(test_signers)}|{'|'.join(val_signers)}")) & (labels['label'] < n_classes) ]

#4920 1948 1220 8592 504
print(train_csv.shape[0],test_csv.shape[0],val_csv.shape[0],labels.shape[0],full_data.shape,labels[(labels['label'] >= n_classes) ].shape[0]) 

# save file
train_csv.to_csv('train_1_200.csv',index=False)
test_csv.to_csv('test_1_200.csv',index=False)
val_csv.to_csv('val_1_200.csv',index=False)
full_data.to_csv("full_data_1_200.csv",index=False)
