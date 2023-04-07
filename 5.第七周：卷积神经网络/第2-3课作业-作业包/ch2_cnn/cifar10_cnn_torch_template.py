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


history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
#import adabound

# 定义超参数
batch_size = 64
learning_rate = 3e-2
final_learning_rate=3
num_epochs = 10


# 定义数据预处理方式
# 普通的数据预处理方式
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# 数据增强的数据预处理方式
# transform = transforms.Compose(
#T.Resize(256),
#T.CenterCrop(224),
#T.ToTensor(),
#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),



# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(64, 256, 3, padding=1)
        #self.conv2_2 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(256, 1024, 3, padding=1)
        #self.conv3_2 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(1024)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(1024 * 4 * 4, 4000)
        self.dropout1 = nn.Dropout(0.2)
        self.bn4 = nn.BatchNorm1d(4000)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(4000, 1000)
        self.dropout2 = nn.Dropout(0.2)
        self.bn5 = nn.BatchNorm1d(1000)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.pool1(self.conv1_2(self.conv1_1(x)))))
        #x = self.relu2(self.bn2(self.pool2(self.conv2_2(self.conv2_1(x)))))
        #x = self.relu3(self.bn3(self.pool3(self.conv3_2(self.conv3_1(x)))))
        #x = self.relu1(self.bn1(self.pool1(self.conv1_1(x))))
        x = self.relu2(self.bn2(self.pool2(self.conv2_1(x))))
        x = self.relu3(self.bn3(self.pool3(self.conv3_1(x))))
        x = x.view(-1, 1024 * 4 * 4)
        x = self.relu4(self.bn4(self.dropout1(self.fc1(x))))
        x = self.relu5(self.bn5(self.dropout2(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        

# 实例化模型
model = Net()


use_mlu = False
try:
    use_mlu = torch.mlu.is_available()
    #
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    global ct
    #
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

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
#optimizer = adabound.AdaBound(model.parameters(), lr=learning_rate, final_lr=final_learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
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
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.legend()
plt.show()
print(history)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.legend()
plt.show()