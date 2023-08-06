import os
import cv2
from tqdm import tqdm


"""
查看待删除的多余文件
find . -iname '__MACOSX'
find . -iname '.DS_Store'
find . -iname '.ipynb_checkpoints'

python查看代码
'.DS_Store' in os.listdir('dataset_delete_test/芒果')

删除多余文件
for i in `find . -iname '__MACOSX'`; do rm -rf $i;done
for i in `find . -iname '.DS_Store'`; do rm -rf $i;done
for i in `find . -iname '.ipynb_checkpoints'`; do rm -rf $i;done

https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/dataset_delete_test.zip
"""

dataset_path = 'fer_ckplus'
#删除gif格式的图像文件
for fruit in tqdm(os.listdir(dataset_path)):
    for file in os.listdir(os.path.join(dataset_path, fruit)):
        file_path = os.path.join(dataset_path, fruit, file)
        img = cv2.imread(file_path)
        if img is None:
            print(file_path, '读取错误，删除')
            os.remove(file_path)

# 删除非三通道的图像
import numpy as np
from PIL import Image
for fruit in tqdm(os.listdir(dataset_path)):
    for file in os.listdir(os.path.join(dataset_path, fruit)):
        file_path = os.path.join(dataset_path, fruit, file)
        img = np.array(Image.open(file_path))
        try:
            channel = img.shape[2]
            if channel != 3:
                print(file_path, '非三通道，删除')
                os.remove(file_path)
        except:
            print(file_path, '非三通道，删除')
            os.remove(file_path)