if __name__ == '__main__':

    import time
    import os
    from tqdm import tqdm

    import pandas as pd
    import numpy as np

    import torch.nn as nn
    import math

    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url
    import torch


    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F



    import matplotlib.pyplot as plt

    # 忽略烦人的红色提示
    import warnings
    from torchvision import models
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score

    import wandb

    wandb.init(project='fruit81', name=time.strftime('%m%d%H%M%S'))

    warnings.filterwarnings("ignore")

    # 获取计算硬件
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('mps')
    print('device', device)

    from torchvision import transforms

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

    dataset_dir = 'fruit81_split'

    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'val')
    print('训练集路径', train_path)
    print('测试集路径', test_path)

    from torchvision import datasets
    # 载入训练集
    train_dataset = datasets.ImageFolder(train_path, train_transform)
    # 载入测试集
    test_dataset = datasets.ImageFolder(test_path, test_transform)

    print('训练集图像数量', len(train_dataset))
    print('类别个数', len(train_dataset.classes))
    print('各类别名称', train_dataset.classes)
    print('测试集图像数量', len(test_dataset))
    print('类别个数', len(test_dataset.classes))
    print('各类别名称', test_dataset.classes)

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

    from torch.utils.data import DataLoader


    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    }

    class ChannelAttention(nn.Module):
        def __init__(self, in_planes, ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
            return self.sigmoid(out)


    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()

            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = 3 if kernel_size == 7 else 1

            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)


    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(BasicBlock, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class Bottleneck(nn.Module):
        # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        # while original implementation places the stride at the first 1x1 convolution(self.conv1)
        # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
        # This variant is also known as ResNet V1.5 and improves accuracy according to
        # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(Bottleneck, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class ResNet(nn.Module):

        def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                     groups=1, width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None):
            super(ResNet, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)

            # 网络的第一层加入注意力机制
            self.ca = ChannelAttention(self.inplanes)
            self.sa = SpatialAttention()

            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            # 网络的卷积层的最后一层加入注意力机制
            self.ca1 = ChannelAttention(self.inplanes)
            self.sa1 = SpatialAttention()

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.ca(x) * x
            x = self.sa(x) * x

            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.ca1(x) * x
            x = self.sa1(x) * x

            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)

            return x


    def _resnet(arch, block, layers, pretrained, progress, **kwargs):
        model = ResNet(block, layers, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            new_state_dict = model.state_dict()
            new_state_dict.update(state_dict)
            model.load_state_dict(new_state_dict)
        return model

    def resnet34(pretrained=False, progress=True, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                       **kwargs)



    BATCH_SIZE = 12

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4
                             )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4
                            )




    model_path = 'checkpoint/resnet34-333f7ec4.pth'  # 预训练参数的位置
    # 自己重写的网络
    model = resnet34(pretrained=True)
    # model_dict = model.state_dict()  # 网络层的参数
    # # 需要加载的预训练参数
    # pretrained_dict = torch.load(model_path)['state_dict']  # torch.load得到是字典，我们需要的是state_dict下的参数
    # pretrained_dict = {k.replace('module.', ''): v for k, v in
    #                    pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。
    #
    # # 删除pretrained_dict.items()中model所没有的东西
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
    # model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
    # model.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值

    ## 选择二：微调训练所有层
    #model = models.resnet50(pretrained=True) # 载入预训练模型
    #model = models.alexnet(pretrained=True)
    #model.fc = nn.Linear(model.fc.in_features, n_class)
    optimizer = optim.Adam(model.parameters(),lr=0.01)

    model = model.to(device)
    # weights = torch.FloatTensor([1, 1, 8, 8, 4])  # 类别权重分别是 1:1:8:8:4
    # # pos_weight_weight(tensor): 1-D tensor，n 个元素，分别代表 n 类的权重，
    # # 为每个批次元素的损失指定的手动重新缩放权重，
    # # 如果你的训练样本很不均衡的话，是非常有用的。默认值为 None。
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weights).cuda()
    #weights =torch.FloatTensor([8,15,8,1,15])
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练轮次 Epoch
    EPOCHS = 90

    # 学习率降低策略
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def train_one_batch(images, labels):
        '''
        运行一个 batch 的训练，返回当前 batch 的训练日志
        '''
        # 获得一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # 输入模型，执行前向预测
        loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

        # 优化更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 获取当前 batch 的标签类别和预测类别
        _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        log_train = {}
        log_train['epoch'] = epoch
        log_train['batch'] = batch_idx
        # 计算分类评估指标
        log_train['train_loss'] = loss
        log_train['train_accuracy'] = accuracy_score(labels, preds)
        # log_train['train_precision'] = precision_score(labels, preds, average='macro')
        # log_train['train_recall'] = recall_score(labels, preds, average='macro')
        # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

        return log_train


    def evaluate_testset():
        '''
        在整个测试集上评估，返回分类评估指标日志
        '''

        loss_list = []
        labels_list = []
        preds_list = []

        with torch.no_grad():
            for images, labels in test_loader:  # 生成一个 batch 的数据和标注
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 输入模型，执行前向预测

                # 获取整个测试集的标签类别和预测类别
                _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
                preds = preds.cpu().numpy()
                loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
                loss = loss.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(preds)

        log_test = {}
        log_test['epoch'] = epoch

        # 计算分类评估指标
        log_test['test_loss'] = np.mean(loss_list)
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

        return log_test

    epoch = 0
    batch_idx = 0
    best_test_accuracy = 0

    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    log_train = {}
    log_train['epoch'] = 0
    log_train['batch'] = 0
    images, labels = next(iter(train_loader))
    log_train.update(train_one_batch(images, labels))
    df_train_log = df_train_log.append(log_train, ignore_index=True)

    print(df_train_log)

    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    log_test = {}
    log_test['epoch'] = 0
    log_test.update(evaluate_testset())
    df_test_log = df_test_log.append(log_test, ignore_index=True)
    print(df_test_log)

    for epoch in range(1, EPOCHS + 1):

        print(f'Epoch {epoch}/{EPOCHS}')

        ## 训练阶段
        model.train()
        for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
            batch_idx += 1
            log_train = train_one_batch(images, labels)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
            wandb.log(log_train)

        lr_scheduler.step()

        ## 测试阶段
        model.eval()
        log_test = evaluate_testset()
        df_test_log = df_test_log.append(log_test, ignore_index=True)
        wandb.log(log_test)

        # 保存最新的最佳模型文件
        if log_test['test_accuracy'] > best_test_accuracy:
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            best_test_accuracy = log_test['test_accuracy']
            new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(log_test['test_accuracy'])
            torch.save(model, new_best_checkpoint_path)
            print('保存新的最佳模型', 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy))
            # best_test_accuracy = log_test['test_accuracy']

    df_train_log.to_csv('训练日志-训练集.csv', index=False)
    df_test_log.to_csv('训练日志-测试集.csv', index=False)

    # 载入最佳模型作为当前模型
    model = torch.load('checkpoint/best-{:.3f}.pth'.format(best_test_accuracy))
    model.eval()
    print(evaluate_testset())