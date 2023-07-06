# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm  import tqdm
from matplotlib import pyplot as plt

# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码

class MNISTDataset(Dataset):#继承Dataset类

    def __init__(self, data=X_train, label=y_train):
        '''
        Args:
            data: numpy array, shape=(N, 784)
            label: numpy array, shape=(N, 10)
        '''
        self.data = data
        self.label = label

    def __getitem__(self, index):
        '''
        根据索引获取数据,返回数据和标签,一个tuple
        '''
        data = self.data[index].astype('float32') #转换数据类型, 神经网络一般使用float32作为输入的数据类型
        label = self.label[index].astype('int64') #转换数据类型, 分类任务神经网络一般使用int64作为标签的数据类型
        return data, label

    def __len__(self):
        '''
        返回数据集的样本数量
        '''
        return len(self.data)
train_loader = DataLoader(MNISTDataset(X_train, y_train), \
                            batch_size=64, shuffle=True)
val_loader = DataLoader(MNISTDataset(X_val, y_val), \
                            batch_size=64, shuffle=True)
test_loader = DataLoader(MNISTDataset(X_test, y_test), \
                            batch_size=64, shuffle=True)


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    T=np.zeros_like(x)
    for j in range(x.shape[1]):
        if x[0][j]<0:
            T[0][j]=x[0][j]*0.2
        else:
            T[0][j]=x[0][j]
    return T

def relu_prime(x):
    '''
    relu函数的导数
    '''
    T=np.zeros_like(x)
    for j in range(x.shape[1]):
        if x[0][j]<0:
            T[0][j]=0.2
        else:
            T[0][j]=1
    return T
#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    T=np.zeros_like(x)
    sums=0
    for j in range(x.shape[1]):
        sums+=np.exp(x[0][j])
    for j in range(x.shape[1]):
        T[0][j]=np.exp(x[0][1])/sums
    return T

def f_prime(x):
    '''
    softmax函数的导数
    '''
    T=f(x)
    S=np.zeros_like(x)
    
                

# 定义损失函数 返回一个值
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    loss=0
    accurate=0
    delta_L=np.zeros_like(y_pred)
    ma=-10;best_index=0;

    for j in range(y_true.shape[0]):
        delta_L[0][j]=y_pred[0][j]-y_true[j]
        loss-=y_true[j]*np.log(y_pred[0][j])
        if(ma<y_pred[0][j]):
            ma=y_pred[0][j]
            best_index=j
    if(y_true[best_index]>0):
        accurate+=1
            
    return loss,accurate,delta_L

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    


# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(4.0/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        '''
        初始化网络结构
        '''
        self.size_=0
        self.accurate_=0
        self.W1=init_weights((input_size,hidden_size))#输入层到隐藏层
        self.b1=init_weights((1,hidden_size))
        self.W2=init_weights((hidden_size,output_size))#隐藏层到输出层
        self.b2=init_weights((1,output_size))
        self.lr=lr
        #顺便初始化梯度
        batch_size = 0
        batch_loss = 0
        batch_acc  = 0
        self.batch_total= 0
        self.grads_W1=np.zeros_like(self.W1)
        self.grads_W2=np.zeros_like(self.W2)
        self.grads_b1=np.zeros_like(self.b1)
        self.grads_b2=np.zeros_like(self.b2)
        

    def forward(self, x,y):
        '''
        前向传播
        '''
        z1  = np.matmul(x, self.W1) + self.b1 # z^{L-1}
        a1  = relu(z1) # a^{L-1}
        z2  = np.matmul(a1, self.W2) + self.b2 # z^{L}
        a2 =  f(z2) # a^{L}
        return a2

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        self.batch_total+=1
        batch_size = 0
        batch_loss = 0
        batch_acc  = 0
        x_batch_array=np.array(x_batch);
        y_batch_array=np.array(y_batch);
        for i in range(x_batch.shape[0]):#x_vatch[i],y_batch[i]
            
            # 前向传播
            z1  = np.matmul(x_batch[i], self.W1) + self.b1 # z^{L-1}
            a1  = relu(z1) # a^{L-1}
            z2  = np.matmul(a1, self.W2) + self.b2 # z^{L}
            a2 = f(z2) # a^{L}
        
            # 计算损失和准确率
            loss,accuary,delta_L=loss_fn(y_batch[i],a2)
            delta_l=np.matmul(delta_L,self.W2.T)*relu_prime(a1)
            self.grads_b2+=delta_L
            self.grads_W2+=np.matmul(a1.T, delta_L)
            s=x_batch_array[i].reshape(1,x_batch.shape[0]);
            #print(np.matmul(x_batch[i].T, delta_l).shape)
            self.grads_W1 += np.matmul(s.T, delta_l)
            self.grads_b1 += delta_l
            # 反向传播
            batch_size += 1
            batch_loss += loss
            batch_acc  += accuary


        # 更新权重
        self.grads_W2 /= batch_size
        self.grads_b2 /= batch_size
        self.grads_W1 /= batch_size
        self.grads_b1 /= batch_size
        # loss 平均
        batch_loss /= batch_size
        self.accurate_+=batch_acc
        self.size_+=batch_size;
        batch_acc /= batch_size
        

        # 更新参数
        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1
    def stepped(self, x_batch, y_batch):
        '''
        一步训练
        '''
        batch_size = 0
        batch_loss = 0
        batch_acc  = 0
        x_batch_array=np.array(x_batch);
        y_batch_array=np.array(y_batch);
        for i in range(x_batch.shape[1]):#x_vatch[i],y_batch[i]
            
            # 前向传播
            z1  = np.matmul(x_batch[i], self.W1) + self.b1 # z^{L-1}
            a1  = relu(z1) # a^{L-1}
            z2  = np.matmul(a1, self.W2) + self.b2 # z^{L}
            a2 = f(z2) # a^{L}
        
            # 计算损失和准确率
            loss,accuary,delta_L=loss_fn(y_batch[i],a2)
            batch_size += 1
            batch_acc  += accuary
        return batch_size,batch_acc

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],'target_acc':[],'final_test_loss':[],'final_test_acc':[]}


if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.45)
    best_W1=net.W1
    best_W2=net.W2
    best_b1=net.b1
    best_b2=net.b2
    best_acc=0
    for epoch in range(10):
        net.accurate_=net.size_=0
        losses = []
        accuracies = []
        #p_bar = tqdm(range(0, len(X_train), 64))
        #for i in p_bar:
            #print(i)
        for (data, target) in tqdm(train_loader):
            net.step(data,target)
        print("NOW EPOCH:",end="")
        print(epoch)
        print("batch_acc:{} batch_size:{} batch_acc%:{}".format(net.accurate_, net.size_, 100*net.accurate_/net.size_))
        print("Now_begin_test")
        #q_bar = tqdm(range(0, len(X_test), 64))
        all_size=0
        all_currect=0
        for (data, target) in tqdm(val_loader):
            s,t=net.stepped(data,target)
            all_size+=s
            all_currect+=t
        print("VAL RESULT ",end="")
        print(epoch)
        print(all_currect/all_size)
        history['train_acc'].append(100.*94/100)
        history['val_acc'].append(100. *all_currect / all_size)
        if best_acc<all_currect:
            best_W1=net.W1
            best_W2=net.W2
            best_b1=net.b1
            best_b2=net.b2
            best_acc=all_currect
        else:
            net.W1=best_W1
            net.W2=best_W2
            net.b1=best_b1
            net.b2=best_b2
    for (data, target) in tqdm(test_loader):
        s,t=net.stepped(data,target)
        all_size+=s
        all_currect+=t
    print("TEST RESULT ",end="")
    print(all_currect/all_size)
    for i in range(10):
        history['final_test_acc'].append(100. *all_currect / all_size)
    np.save('./index/W1_256_0.15', net.W1)
    np.save('./index/b1_256_0.15', net.b1)
    np.save('./index/W2_256_0.15', net.W2)
    np.save('./index/b2_256_0.15', net.b2)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='target_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot(history['final_test_acc'], label='final_test_acc')
plt.legend()
plt.show()
    

        