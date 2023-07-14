import os
import time
import shutil
import tempfile
from tqdm import tqdm

import cv2


import mmcv
input_video = '/Users/aspiriner/Documents/工作文件/华师大/图像识别与骨架分析/原视频.nosync/IMG_7467.MOV'

# 创建临时文件夹，存放每帧结果
#temp_out_dir = time.strftime('%Y%m%d%H%M%S')
temp_out_dir = "photo01"
os.mkdir(temp_out_dir)
print('创建文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))

# 读入待预测视频
imgs = mmcv.VideoReader(input_video)

prog_bar = mmcv.ProgressBar(len(imgs))

i = 0
idx = 1
# 对视频逐帧处理
for frame_id, img in enumerate(imgs):
    i = i + 1
    ## 处理单帧画面
    if i == 60 :
    # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
        cv2.imwrite(f'{temp_out_dir}/{idx:06d}.jpg', img)
        i = 0
        idx = idx + 1
    prog_bar.update()  # 更新进度条
