import numpy as np
import cv2

# 输入0调用系统摄像头
cap = cv2.VideoCapture(0)
# 获取本地视频
# cap = cv2.VideoCapture("./text.mp4")

# cap.isOPened()查看初始化是否成功，返回值是True或False
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源关闭窗口
cap.release()
cap.destroyAllWindows()