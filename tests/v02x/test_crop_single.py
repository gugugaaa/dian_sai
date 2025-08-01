import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 导入 ShapeDetector、sort_corners、rotate_and_crop
from shape_detector import ShapeDetector

# 读取测试图像
image = cv2.imread("images/rotate_crop/single_test.png")
if image is None:
    raise FileNotFoundError("未找到 images/rotate_crop/single_test.png")

# 检测形状
detector = ShapeDetector()
shape, size, params = detector.detect_shape(image)

if shape == "square":
    print("检测到正方形，角点:", params)
    
    # 排序角点
    sorted_corners = detector.sort_corners(params)
    print("排序后角点 (TL, TR, BR, BL):", sorted_corners)
    
    # 在“Detected”图上绘制带标签的角点，来验证排序是否正确
    drawn_image_with_labels = image.copy()
    labels = ["TL", "TR", "BR", "BL"]
    for i, point in enumerate(sorted_corners):
        cv2.circle(drawn_image_with_labels, tuple(point), 5, (255, 0, 0), -1)
        cv2.putText(drawn_image_with_labels, labels[i], tuple(point + np.array([10, -10])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 旋转并裁切
    cropped_image = detector.rotate_and_crop(image, sorted_corners)

    # 显示结果
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    
    # 显示我们刚刚画好标签的图
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(drawn_image_with_labels, cv2.COLOR_BGR2RGB))
    plt.title("Detected & Sorted Corners")
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Cropped")
    
    plt.show()