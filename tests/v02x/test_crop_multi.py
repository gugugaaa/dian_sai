import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 导入 ShapeDetector
from shape_detector import ShapeDetector

# 读取测试图像
image = cv2.imread("images/rotate_crop/multi_test.png")
if image is None:
    raise FileNotFoundError("未找到 images/rotate_crop/multi_test.png")

# 检测所有正方形
detector = ShapeDetector()
squares = detector.detect_squares(image, min_area=1000)

print(f"检测到 {len(squares)} 个正方形")

if len(squares) == 0:
    print("未检测到正方形")
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original - No Squares Detected")
    plt.show()
else:
    # 绘制检测结果
    detected_image = image.copy()
    for i, square in enumerate(squares):
        points = square['params'].astype(np.int32)
        cv2.polylines(detected_image, [points], True, (0, 255, 0), 2)
        # 标记序号
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(detected_image, str(i+1), tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 对每个正方形进行旋转裁切
    cropped_images = []
    for i, square in enumerate(squares):
        print(f"处理第 {i+1} 个正方形...")
        corners = square['params']
        sorted_corners = detector.sort_corners(corners)
        cropped = detector.rotate_and_crop(image, sorted_corners)
        cropped_images.append(cropped)

    # 显示结果
    num_squares = len(squares)
    cols = min(3, num_squares + 1)  # 原图 + 最多3个裁切图
    rows = (num_squares + cols) // cols
    
    plt.figure(figsize=(4 * cols, 4 * rows))
    
    # 显示原图和检测结果
    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected {num_squares} Squares")
    plt.axis('off')
    
    # 显示每个裁切后的正方形
    for i, cropped in enumerate(cropped_images):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title(f"Square {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()