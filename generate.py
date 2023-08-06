import os
import cv2

# 输入和输出文件夹路径
input_folder = 'trainall/happiness/'
output_folder = 'trainall/test/'

# 旋转角度（逆时针为正方向）
angle = -20

# 获取输入文件夹中的图像文件列表
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 遍历每个图像文件并进行旋转
for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, f'new{image_file}')

    # 读取图像
    image = cv2.imread(input_path)

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # 执行图像旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # 保存旋转后的图像
    cv2.imwrite(output_path, rotated_image)

print('图像旋转和保存完成')
