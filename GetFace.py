#-*- coding: utf-8 -*-
# import 进openCV的库
import cv2
import os
import mmcv
###调用电脑摄像头检测人脸并截图

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    temp_out_dir = "photo10"
    #os.rmdir(temp_out_dir)
    os.mkdir(temp_out_dir)

    print('创建文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture('/Users/aspiriner/Documents/工作文件/华师大/图像识别与骨架分析/原视频.nosync/IMG_0011.MOV')
    #prog_bar = mmcv.ProgressBar(10000000)

    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    #classfier = cv2.CascadeClassifier("haarcascade_profileface.xml")

    #识别出人脸后要画的边框的颜色，RGB格式, color是一个不可增删的数组
    color = (0, 255, 0)
    i = 0
    idx = 0
    num = 0
    while cap.isOpened():
        i = i + 1
        #print(i)
        ok, frame = cap.read()  # 读取一帧数据
        if i == 30 :
            i = 0
            idx = idx + 1
            if not ok:
                break

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像

            #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
            faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.5, minNeighbors = 4, minSize = (120, 120), maxSize = (270,270))
            if len(faceRects) > 0:          #大于0则检测到人脸
                for faceRect in faceRects:  #单独框出每一张人脸
                    x, y, w, h = faceRect

                    #将当前帧保存为图片
                    img_name = "%s/%d.jpg" % (path_name, num)
                    #print(img_name)
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    if y -10 < 0 or x -10 < 0:
                        print(num)
                        pass
                    else:
                        cv2.imwrite(f'{temp_out_dir}/{idx:06d}.jpg', image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                    num += 1


                    #画出矩形框
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                    #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(frame,'num:%d/100' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)

                    #超过指定最大保存数量结束程序

        #prog_bar.update()  # 更新进度条

            #显示图像
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(2)
            if c & 0xFF == ord('q'):
               break

            #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 连续截100张图像，存进image文件夹中
    CatchPICFromVideo("get face", 0, 99, "/image")
