import os
import shutil
import random
import pandas as pd

dataset_path = 'fer_ckplus'

dataset_name = dataset_path.split('_')[0]
#print('数据集', dataset_name)

classes = os.listdir(dataset_path)
#len(classes)

# 创建 train 文件夹
os.mkdir(os.path.join(dataset_path, 'train'))

# 创建 test 文件夹
os.mkdir(os.path.join(dataset_path, 'val'))

# 在 train 和 test 文件夹中创建各类别子文件夹
for fruit in classes:
    os.mkdir(os.path.join(dataset_path, 'train', fruit))
    os.mkdir(os.path.join(dataset_path, 'val', fruit))

test_frac = 0.2  # 测试集比例
random.seed(123) # 随机数种子，便于复现

df = []

print('{:^18} {:^18} {:^18}'.format('类别', '训练集数据个数', '测试集数据个数'))

for fruit in classes:  # 遍历每个类别

    # 读取该类别的所有图像文件名
    old_dir = os.path.join(dataset_path, fruit)
    images_filename = os.listdir(old_dir)
    random.shuffle(images_filename)  # 随机打乱

    # 划分训练集和测试集
    testset_numer = int(len(images_filename) * test_frac)  # 测试集图像个数
    testset_images = images_filename[:testset_numer]  # 获取拟移动至 test 目录的测试集图像文件名
    trainset_images = images_filename[testset_numer:]  # 获取拟移动至 train 目录的训练集图像文件名

    # 移动图像至 test 目录
    for image in testset_images:
        old_img_path = os.path.join(dataset_path, fruit, image)  # 获取原始文件路径
        new_test_path = os.path.join(dataset_path, 'val', fruit, image)  # 获取 test 目录的新文件路径
        shutil.move(old_img_path, new_test_path)  # 移动文件

    # 移动图像至 train 目录
    for image in trainset_images:
        old_img_path = os.path.join(dataset_path, fruit, image)  # 获取原始文件路径
        new_train_path = os.path.join(dataset_path, 'train', fruit, image)  # 获取 train 目录的新文件路径
        shutil.move(old_img_path, new_train_path)  # 移动文件

    # 删除旧文件夹
    assert len(os.listdir(old_dir)) == 0  # 确保旧文件夹中的所有图像都被移动走
    shutil.rmtree(old_dir)  # 删除文件夹

    # 工整地输出每一类别的数据个数
    print('{:^18} {:^18} {:^18}'.format(fruit, len(trainset_images), len(testset_images)))

    # 保存到表格中
    #df = df.append({'class': fruit, 'trainset': len(trainset_images), 'testset': len(testset_images)},
    #               ignore_index=True)
#df = pd.DataFrame(df)
# 重命名数据集文件夹
shutil.move(dataset_path, dataset_name + '_split')

# 数据集各类别数量统计表格，导出为 csv 文件
#df['total'] = df['trainset'] + df['testset']
#df.to_csv('数据量统计.csv', index=False)
