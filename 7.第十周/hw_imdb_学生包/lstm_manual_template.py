# 用Pytorch手写一个LSTM网络，在IMDB数据集上进行训练

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from utils import load_imdb_dataset, Accuracy
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

from utils import load_imdb_dataset, Accuracy
history = {
    'train_loss': [], 'train_acc': [], 
    'test_loss': [], 'test_acc': [],
    'target_acc':[],'final_test_loss':[],'final_test_acc':[]}
use_mlu = False
try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    global ct
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')



X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1

class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set
        label = self.y[index].astype('int32')
        return data, label

    def __len__(self):

        return len(self.y)

prev_h = np.random.random([1, 200, 64]).astype(np.float32)
prev_h = torch.FloatTensor(prev_h).to(device)
class LSTM(nn.Module):
    '''
    手写lstm，可以用全连接层nn.Linear，不能直接用nn.LSTM
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size=input_size
        self.Act=nn.Sigmoid()
        self.Act2=nn.Tanh()
        # LSTM层
        self.Wfx=nn.Linear(self.input_size,self.hidden_size)
        self.WfH=nn.Linear(self.hidden_size,self.hidden_size)
        self.Wix=nn.Linear(self.input_size,self.hidden_size)
        self.WiH=nn.Linear(self.hidden_size,self.hidden_size)
        self.Wcx=nn.Linear(self.input_size,self.hidden_size)
        self.WcH=nn.Linear(self.hidden_size,self.hidden_size)
        self.Wox=nn.Linear(self.input_size,self.hidden_size)
        self.WoH=nn.Linear(self.hidden_size,self.hidden_size)
        #self.Wi=nn.Linear(self.real_size,self.hidden_size)
        #self.Wc=nn.Linear(self.real_size,self.hidden_size)
        #self.Wo=nn.Linear(self.real_size,self.hidden_size)
        #self.Wy=nn.Linear(self.real_size,self.hidden_size)
        
        self.Bf=nn.Linear(1,self.hidden_size)
        self.Bi=nn.Linear(1,self.hidden_size)
        self.Bc=nn.Linear(1,self.hidden_size)
        self.Bo=nn.Linear(1,self.hidden_size)
        


class Net(nn.Module):
    '''
    一层LSTM的文本分类模型
    '''
    def __init__(self, embedding_size=64, hidden_size=64, num_classes=2):
        super(Net, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM层
        self.lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size)
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.act1= nn.ReLU()

    def forward(self, x):
        '''
        x: 输入, shape: (seq_len, batch_size)
        '''

        # 词嵌入
        #print("x 维度")
        #print(x.shape)
        x = self.embedding(x)
        #print("x 维度")
        #print(x.shape)
        # LSTM层
        #X=torch.cat((prev_h,x),dim=1)
        
        
        Ft=self.lstm.Act(self.lstm.Wfx(x)+self.lstm.WfH(prev_h))
        It=self.lstm.Act(self.lstm.Wix(x)+self.lstm.WiH(prev_h))
        Ct=Ft*prev_h+It*self.lstm.Act2(self.lstm.Wcx(x)+self.lstm.WcH(prev_h))
        Ot=self.lstm.Act(self.lstm.Wox(x)+self.lstm.WoH(prev_h))
        Ht=Ot*self.lstm.Act2(Ct)
        x = torch.mean(Ot, dim=1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x



n_epoch = 5
batch_size = 16
print_freq = 2

train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = Net()
metric = Accuracy()
print(net)

def train(model, device, train_loader, optimizer, epoch,test=0):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter= 0
    batch_idx=0
    for  (data, target) in tqdm(train_loader):
        batch_idx+=1
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        if(test==0):
            loss.backward()
            optimizer.step()
        metric.update(output,target)
        train_acc += metric.result()
        train_loss +=  loss.item()
        metric.reset()
        n_iter+=1
    if(train==0):
        print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(
            epoch, train_loss / n_iter,train_acc/n_iter))
        history['train_acc'].append(train_acc/n_iter)
        history['train_loss'].append(train_loss/n_iter)
    else :
        print('Test Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(
            epoch, train_loss / n_iter,train_acc/n_iter))
        for i in range(epoch):
            history['test_acc'].append(train_acc/n_iter)
            history['test_loss'].append(train_loss/n_iter)
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
train(net,device,test_loader,optimizer,epoch,1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['test_acc'], label='test_acc')
plt.legend()
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='target_loss')
plt.plot(history['test_loss'], label='test_loss')
plt.show()
plt.savefig('data/figure_work2_1.jpg')
plt.close()
filename='data/test_work2_1.txt'
if not os.path.exists(filename):
    os.mknod(filename)
with open(filename,'w') as file:
    file.write(str(history))
    file.close()