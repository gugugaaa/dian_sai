import cv2
import numpy as np

# 读取图像
img = cv2.imread('images/test/test.jpg')
if img is None:
    print("错误：无法读取图像文件")
    exit()

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi角点检测参数
corners = cv2.goodFeaturesToTrack(
    gray,           # 输入灰度图像
    maxCorners=100, # 最大角点数
    qualityLevel=0.1,  # 角点质量水平
    minDistance=10,     # 角点间最小距离
    blockSize=3,        # 角点检测窗口大小
    useHarrisDetector=False  # 使用Shi-Tomasi而非Harris
)


