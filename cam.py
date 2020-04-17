import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import cv2

# Global Variables
SOURCE = 0  # 或者为后缀为.mp4格式的视频路径
INPUT_SIZE = (112, 112)  # 图片大小为112*112
MODEL_PATH = 'exported/'  # 存放预训练模型的位置为exported文件夹


def crop_img(end_x, end_y, frame, start_x, start_y):  # 剪裁图片
    face_img = frame[start_y:end_y, start_x:end_x, :]
    face_img = cv2.resize(face_img, INPUT_SIZE)
    face_img = face_img - 127.5
    face_img = face_img * 0.0078125
    return face_img  # 返回裁剪完成后的人脸图片


def show_text(frame, start_x, start_y, color):
    # 在图像中显示文字 文字的位置为(start_x, start_y - 10)
    if color == (0, 255, 0):
        text = '@_SAFE_@'
        size = 1.2
    else:
        text = '!_DANGER_!'
        size = 1.2

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


def main():
    detector = MTCNN()
    mask_model = tf.keras.models.load_model(MODEL_PATH)  # 加载预训练模型
    cap = cv2.VideoCapture(SOURCE)  # 此时使用的是全局变量SOURCE
    # 当全局变量SOURCE = 0时打开摄像头 当SOURCE为.mp4文件路径时，打开.mp4文件
    flg = True  # 设置循环变量(死循环)

    while flg:
        flg, frame = cap.read()  # 读取摄像头获取的图像
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

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 将处理完成后的结果进行再处理 使得处理后的图片从RGB格式返回BGR格式
        cv2.namedWindow('MaskHero', cv2.WINDOW_NORMAL)
        cv2.imshow('MaskHero', frame)  # 打开窗口显示处理后的结果

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # 等待直到接收到退出指令，退出循环

    # while end


if __name__ == '__main__':
    main()
