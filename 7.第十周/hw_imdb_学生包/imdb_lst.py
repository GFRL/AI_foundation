# The same set of code can switch the backend with one line
import os

import torch
from torch.nn import Module
from torch.nn import Linear, LSTM, Embedding
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import os
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

prev_h = np.random.random([1, 200, 64]).astype(np.float32)
prev_h = torch.FloatTensor(prev_h).to(device)

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1
print("vocab_size: ",vocab_size)


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


class ImdbNet(Module):

    def __init__(self):
        super(ImdbNet, self).__init__()
        DROPOUT=0.2
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
        self.lstm = LSTM(input_size=64, hidden_size=64)
        self.linear1 = Linear(in_features=64, out_features=64)
        self.act1 = torch.nn.ReLU()
        self.act11=torch.nn.Dropout(DROPOUT)
        self.linear2 = Linear(in_features=64, out_features=64)
        self.act2 = torch.nn.ReLU()
        self.act12=torch.nn.Dropout(DROPOUT)
        self.linear3 = Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x, [prev_h, prev_h])
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        return x


n_epoch = 5
batch_size = 16
print_freq = 2

train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = ImdbNet()
metric = Accuracy()
print(net)

def train(model, device, train_loader, optimizer, epoch,train=0):
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
        if(train==0):
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
train(net, device, test_loader, optimizer, n_epoch,1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['test_acc'], label='test_acc')
plt.legend()
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='target_loss')
plt.plot(history['test_loss'], label='test_loss')
plt.legend()
plt.show()
plt.savefig('data/figure_work1_64.jpg')
plt.close()
filename='data/test_work1_64.txt'
if not os.path.exists(filename):
    os.mknod(filename)
with open(filename,'w') as file:
    file.write(str(history))
    file.close()
