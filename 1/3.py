# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import torch_mlu
import torch_mlu.core.mlu_model as ct
global ct
use_mlu = torch.mlu.is_available()

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
#import adabound

# 定义超参数
batch_size = 64
learning_rate = 3e-2
final_learning_rate=3
num_epochs = 100


# 定义数据预处理方式
# 普通的数据预处理方式
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# 数据增强的数据预处理方式
transformw = transforms.Compose([
transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.RandomRotation(15), # 随机旋转
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), # 随机颜色变换
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
DROPOUT=0.2
# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(DROPOUT))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(DROPOUT))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(DROPOUT))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(DROPOUT))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(DROPOUT))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
             nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(DROPOUT))
        '''
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(DROPOUT))
        '''
        self.fc1 = nn.Linear(512 * 4 * 4, 10000)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.bn_1 = nn.BatchNorm1d(10000)
        self.relu_1 = nn.ReLU()
        '''
        self.fc2 = nn.Linear(10000, 2000)
        self.dropout2 = nn.Dropout(0.4)
        self.bn_2 = nn.BatchNorm1d(2000)
        self.relu_2 = nn.ReLU()
        '''
        self.fc3 = nn.Linear(10000, 400)
        self.dropout3 = nn.Dropout(DROPOUT)
        self.bn_3 = nn.BatchNorm1d(400)
        self.relu_3 = nn.ReLU()

        self.fc4 = nn.Linear(400, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        #x=self.conv7(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.relu_1(self.bn_1(self.dropout1(self.fc1(x))))
        #x = self.relu_2(self.bn_2(self.dropout2(self.fc2(x))))
        x = self.relu_3(self.bn_3(self.dropout3(self.fc3(x))))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
        

# 实例化模型
model = Net()
model1=model

use_mlu = False
try:
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

model = model.to(device)
best_value=0
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
decayRate=0.9
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = adabound.AdaBound(model.parameters(), lr=learning_rate, final_lr=final_learning_rate)
#my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    '''
    if epoch %5==4:
        for p in optimizer.param_groups:
            p['lr']*=decayRate
    '''
    model.train()
    i=0
    correctt = 0
    totall = 0
    for (images, labels) in tqdm(train_loader):
        optimizer.zero_grad()
        i+=1
        #if(i%100!=0):
         #   continue
        images = images.to(device)
        labels = labels.to(device)
        #print(images.shape)
        # 前向传播
        outputs = model(images)
        #print(outputs.shape)
        #print(outputs)
        #print(labels.shape)
        #print(labels)
        #print(labels.shape)
        #loss = criterion(outputs, labels)
        loss=criterion(outputs,labels)
        # 反向传播
        
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        '''
        if i % 782 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i, len(train_loader), loss.item(), accuracy.item() * 100))
        '''
        _, predicted = torch.max(outputs.data, 1)
        totall += labels.size(0)
        correctt += (predicted == labels).sum().item()
    history['train_acc'].append(100*correctt/totall)
    print('Epoch {} : Test Accuracy of the model on the 50000 train images: {} %'.format(epoch,100 * correctt / totall))
    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    history['val_acc'].append(100*correct/total)
    if correct>best_value:model1=model
    else:model=model1
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    #plt.show()
    #print(history)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    #plt.show()
    plt.savefig('data/figure_3.jpg')
    plt.close()
    filename='data/test_value_3.txt'
    if not os.path.exists(filename):
        os.mknod(filename)
    with open(filename,'w') as file:
        file.write(str(history))
        file.close()