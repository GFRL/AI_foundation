import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm  import tqdm
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        '''
        self.W1=init_weights((input_size,hidden_size))#输入层到隐藏层
        self.b1=init_weights((1,hidden_size))
        self.W2=init_weights((hidden_size,output_size))#隐藏层到输出层
        self.b2=init_weights((1,output_size))
        self.lr=lr
        #顺便初始化梯度
        batch_size = 0
        batch_loss = 0
        batch_acc  = 0
        
        self.grads_W1=np.zeros_like(self.W1)
        self.grads_W2=np.zeros_like(self.W2)
        self.grads_b1=np.zeros_like(self.b1)
        self.grads_b2=np.zeros_like(self.b2)
        pass

net = Network(input_size=5, hidden_size=3, output_size=2, lr=0.01)
print(net.W1)
def relu(x):
    '''
    relu函数
    '''
    for i in range(x.shape[0]):
       for j in range(x.shape[1]):
          if x[i][j]<0:
              x[i][j]*=0.1
    return x
    pass
print (net.W1[1][2])
print(relu(net.W1))