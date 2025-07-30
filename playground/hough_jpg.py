import cv2
import numpy as np
import matplotlib.pyplot as plt

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def hough_line_detection(image_path):
    """
    对指定图像进行霍夫变换直线检测
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

    # 创建结果图像副本
    result = image.copy()
    
    # 在图像上绘制检测到的直线
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        print(f"检测到 {len(lines)} 条直线")
    else:
        print("未检测到直线")
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('边缘检测')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('霍夫变换直线检测')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result

def hough_line_detection_probabilistic(image_path):
    """
    使用概率霍夫变换检测直线段
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 概率霍夫变换检测直线段
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=10)
    
    # 创建结果图像副本
    result = image.copy()
    
    # 在图像上绘制检测到的直线段
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        print(f"检测到 {len(lines)} 条直线段")
    else:
        print("未检测到直线段")
    
    # 显示结果
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('概率霍夫变换直线检测')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result

if __name__ == "__main__":
    # 图像路径
    image_path = "images/test/test.jpg"
    
    print("进行标准霍夫变换直线检测...")
    hough_line_detection(image_path)
    
    print("\n进行概率霍夫变换直线检测...")
    hough_line_detection_probabilistic(image_path)
