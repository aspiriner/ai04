import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
import time
import os

import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('mps')

from torchvision import transforms, datasets

## 定义数据加载器DataLoader
from torch.utils.data import DataLoader

from torchvision import models
import torch.optim as optim
# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                        ])

    # 数据集文件夹路径
    dataset_dir = 'fruit81_split'

    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'val')
    print('训练集路径', train_path)
    print('测试集路径', test_path)

    # 载入训练集
    train_dataset = datasets.ImageFolder(train_path, train_transform)

    # 载入测试集
    test_dataset = datasets.ImageFolder(test_path, test_transform)
    # print('训练集图像数量', len(train_dataset))
    # print('类别个数', len(train_dataset.classes))
    # print('各类别名称', train_dataset.classes)
    # print('测试集图像数量', len(test_dataset))
    # print('类别个数', len(test_dataset.classes))
    # print('各类别名称', test_dataset.classes)
    # 各类别名称
    class_names = train_dataset.classes
    n_class = len(class_names)
    # 映射关系：类别 到 索引号
    train_dataset.class_to_idx
    # 映射关系：索引号 到 类别
    idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
    # 保存为本地的 npy 文件
    np.save('idx_to_labels.npy', idx_to_labels)
    np.save('labels_to_idx.npy', train_dataset.class_to_idx)

    BATCH_SIZE = 32

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=16
                             )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=16
                            )

    ## 查看一个batch的图像和标注
    # DataLoader 是 python生成器，每次调用返回一个 batch 的数据
    images, labels = next(iter(train_loader))
    print(images.shape)
    # 将数据集中的Tensor张量转为numpy的array数据类型
    images = images.numpy()
    #%%
    # plt.hist(images[5].flatten(), bins=50)
    # plt.show()
    # batch 中经过预处理的图像
    # idx = 2
    # plt.imshow(images[idx].transpose((1,2,0))) # 转为(224, 224, 3)
    # plt.title('label:'+str(labels[idx].item()))


    ### 选择一：只微调训练模型最后一层（全连接分类层）
    model = models.resnet18(pretrained=True) # 载入预训练模型
    # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    # 新建的层默认 requires_grad=True
    model.fc = nn.Linear(model.fc.in_features, n_class)
    optimizer = optim.Adam(model.fc.parameters())
    optimizer = optim.Adam(model.fc.parameters())
    ### 选择二：微调训练所有层
    # model = models.resnet18(pretrained=True) # 载入预训练模型
    # model.fc = nn.Linear(model.fc.in_features, n_class)
    # optimizer = optim.Adam(model.parameters())
    ### 选择三：随机初始化模型全部权重，从头训练所有层
    # model = models.resnet18(pretrained=False) # 只载入模型结构，不载入预训练权重参数
    # model.fc = nn.Linear(model.fc.in_features, n_class)
    # optimizer = optim.Adam(model.parameters())

    model = model.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练轮次 Epoch
    EPOCHS = 20

    # full train
    # 遍历每个 EPOCH
    for epoch in tqdm(range(EPOCHS)):

        model.train()

        for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # 前向预测，获得当前 batch 的预测结果
            loss = criterion(outputs, labels)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数

            optimizer.zero_grad()
            loss.backward()  # 损失函数对神经网络权重反向传播求梯度
            optimizer.step()  # 优化更新神经网络权重


    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader): # 获取测试集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)              # 前向预测，获得当前 batch 的预测置信度
            _, preds = torch.max(outputs, 1)     # 获得最大置信度对应的类别，作为预测结果
            total += labels.size(0)
            correct += (preds == labels).sum()   # 预测正确样本个数

        print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))

    torch.save(model, 'checkpoint/fruit81_pytorch_C1.pth')