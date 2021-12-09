import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import sys

import numpy as np

def getLocal(choselabels,alllabels):
    choseloc=np.argwhere(alllabels==choselabels)

    return choseloc


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def changeAtt(att,label,list):
    c = np.ones(att.size(1))
    g_label = []
    for j in list:
        c[j] = 0
    c = np.asarray(c)
    c = torch.from_numpy(c).cuda()
    attchange = att.clone().detach()
    attchange = torch.mul(attchange, c).float()
    labels = np.ones(att.size(0)).astype(int)
    labels = int(label) * labels
    g_label.extend(labels)
    return attchange,labels


class classifierSSL(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(classifierSSL, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.fc=nn.Linear(self.in_dim,self.out_dim)


    def forward(self,x):
        x = F.sigmoid(self.fc(x))

        return x


class classifierSSL_all(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(classifierSSL_all, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.fc=nn.Linear(self.in_dim,self.out_dim)


    def forward(self,x):
        x = F.sigmoid(F.relu(self.fc(x)))

        return x

class classifierSSL_S_U(nn.Module):
    def __init__(self,in_dim):
        super(classifierSSL_S_U, self).__init__()
        self.in_dim = in_dim
        self.out_dim=1
        self.fc=nn.Linear(self.in_dim,self.out_dim)


    def forward(self,x):
        x = F.sigmoid(self.fc(x))

        return x

def changeAtt2(att,dictionary,max):
    c = np.ones((att.size(0),att.size(1)))
    attchange = att.clone().detach()
    num_indexs=len(dictionary.keys())
    label=np.ones((att.size(0),num_indexs))
    for i in range(att.size(0)):
        num=random.randint(1,max)
        indexs=set()
        indexs_key=set()
        n=0
        while n<num:
            index_key=random.randint(0,num_indexs-1)
            if index_key not in indexs_key:
                indexs_key.add(index_key)
                q=dictionary[index_key+1]
                q=set(q)
                indexs = indexs | q
                n=n+1
        indexs=list(indexs)
        attchange[i,indexs]=0
        label[i,list(indexs_key)]=0
    g_label=label
    g_label = torch.from_numpy(g_label)

    return attchange,g_label