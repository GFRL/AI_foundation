# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

# 定义超参数
batch_size = 64
learning_rate = 1e-2
num_epochs = 100

# 定义数据预处理方式
# 普通的数据预处理方式
# transform = transforms.Compose([
#     transforms.ToTensor(),])
# 数据增强的数据预处理方式
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.RandomRotation(15), # 随机旋转
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), # 随机颜色变换
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.bn4 = nn.BatchNorm1d(1024)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.pool1(self.conv2(self.conv1(x)))))
        x = self.relu2(self.bn2(self.pool2(self.conv4(self.conv3(x)))))
        x = self.relu3(self.bn3(self.pool3(self.conv6(self.conv5(x)))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.relu4(self.bn4(self.dropout1(self.fc1(x))))
        x = self.relu5(self.bn5(self.dropout2(self.fc2(x))))
        x = self.fc3(x)
        return x


# 实例化模型
model = Net()

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

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    i=-1
    correctt = 0
    totall = 0
    for (images, labels) in tqdm(train_loader):
        i+=1
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        #print(outputs.shape)
        #print(labels.shape)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        totall += labels.size(0)
        correctt += (predicted == labels).sum().item()
        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        '''
        if (i + 1) % 780 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))
        '''
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