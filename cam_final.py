# -*- coding:utf-8 -*-
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import cv2

# Global Variables
SOURCE = 0  # 或者为后缀为.mp4格式的视频路径
INPUT_SIZE = (112, 112)  # 图片大小为112*112
MODEL_PATH = 'exported/'  # 存放预训练模型的位置为exported文件夹
detector = MTCNN()
mask_model = tf.keras.models.load_model(MODEL_PATH)
# 从MODEL_PATH加载预训练模型
cap = cv2.VideoCapture(SOURCE)


# 此时使用的是全局变量SOURCE
# 当全局变量SOURCE = 0时打开摄像头 当SOURCE为.mp4文件路径时，打开.mp4文件


def crop_img(end_x, end_y, frame, start_x, start_y):  # 剪裁图片
    face_img = frame[start_y:end_y, start_x:end_x, :]
    face_img = cv2.resize(face_img, INPUT_SIZE)
    face_img = face_img - 127.5
    face_img = face_img * 0.0078125
    return face_img  # 返回裁剪完成后的人脸图片


def show_text(frame, start_x, start_y, color):
    # 在图像中显示文字 文字的位置为(start_x, start_y - 10)
    if color == (0, 255, 0):
        text = '@MASKED@'
        size = 1
    else:
        text = '!UNMASKED!'
        size = 1

    frame = cv2.putText(frame,
                        text,
                        (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        size,
                        color,
                        2)
    # 在图像中将文字输出
    return frame  # 返回在已经完成添加文字的图像


def draw_bbox(frame, start_x, start_y, end_x, end_y, have_mask):
    if have_mask:
        color = (0, 255, 0)  # green
    else:
        color = (255, 0, 0)  # red(default color)
    # when have_mask == True
    # color from red turn into green
    frame = show_text(frame,
                      start_x, start_y,
                      color)
    # 通过显示文字函数将文字置放到矩形(start_x,start_y-10)的位置

    cv2.rectangle(frame,
                  (start_x, start_y),
                  (end_x, end_y),
                  color,
                  2)
    # 绘制带有颜色的矩形

    return frame  # 显示矩形


def video_loop():
    flg, frame = cap.read()  # 从摄像头读取照片

    frame = cv2.cvtColor(frame,
                         cv2.COLOR_BGR2RGB)  # 将获取到的图片进行格式转换

    face_locs = detector.detect_faces(frame)  # 人脸检测 返回值为列表格式

    for face_loc in face_locs:
        # 对face_locs列表进行遍历
        bbox = face_loc['box']

        start_x = bbox[0]
        start_y = bbox[1]

        end_x = bbox[0] + bbox[2]
        end_y = bbox[1] + bbox[3]
        # 对图片处理的起始位置设定

        face_img = crop_img(end_x,
                            end_y,
                            frame,
                            start_x,
                            start_y)  # 裁剪获取到的图片

        mask_result = mask_model.predict(np.expand_dims(face_img,
                                                        axis=0))[0]  # 图片检测

        have_mask = np.argmax(mask_result)  # 人脸边缘检索

        frame = draw_bbox(frame,
                          start_x,
                          start_y,
                          end_x,
                          end_y,
                          have_mask)  # 显示矩形
    # for end

    if flg:
        cv2.waitKey(0)
        cv2image = cv2.cvtColor(frame,
                                cv2.COLOR_RGB2RGBA)  # 转换颜色从BGR到RGB
        current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        root.after(1, video_loop)


def Fun1():
    print("Function1提示信息")


def Fun2():
    print("Function2提示信息")


root = Tk()
root.title('MaskHero')
panel = Label(root)  # initialize image panel
panel.pack(padx=10, pady=10)
root.config(cursor="arrow")

btn = Button(root, text='Function1', command=Fun1)
# todo 可以在函数Fun1处添加相关内容

btn1 = Button(root, text='Function2', command=Fun2)
# todo 可以在函数Fun2处添加相关内容


btn.pack(fill="both",
         expand=True,
         padx=10,
         pady=10)
btn1.pack(fill="both",
          expand=True,
          padx=10,
          pady=10)
# button1、2的相关属性

video_loop()
root.mainloop()
# 当一切都完成后，关闭摄像头并释放所占资源

cap.release()
cv2.destroyAllWindows()
