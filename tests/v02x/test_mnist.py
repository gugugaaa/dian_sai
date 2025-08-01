import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 导入 ShapeDetector
from shape_detector import ShapeDetector

class MNISTDigitRecognizer:
    """MNIST数字识别器"""
    def __init__(self):
        # 检查模型文件是否存在
        if not os.path.exists('models/45_mnist_model.pth'):
            raise FileNotFoundError(
                "未找到训练好的模型文件 '45_mnist_model.pth'。"
                "请先训练并保存PyTorch模型。"
            )
        
        # 加载PyTorch MNIST模型
        print("加载MNIST模型...")
        # 使用与训练脚本相同的模型结构
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(32 * 14 * 14, 128)
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load('models/10_mnist_model.pth', map_location='cpu'))
        self.model.eval()
        print("MNIST模型加载完成")
        
        self.digit_images = []  # 存储预处理后的数字图像用于显示
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # 不做归一化，保持0-1
        ])
    
    def predict(self, image_28x28):
        """
        预测28x28图像中的数字
        Args:
            image_28x28: 28x28的灰度图像 (numpy array)
        Returns:
            predicted_digit, confidence
        """
        # 转换为PIL图像再转换为tensor
        pil_image = Image.fromarray(image_28x28.astype(np.uint8))
        tensor_image = self.transform(pil_image).unsqueeze(0)  # 添加batch维度
        
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_digit = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
        
        return predicted_digit, confidence

class DigitExtractor:
    """从裁切的正方形中提取和识别数字"""
    
    def __init__(self, recognizer):
        self.recognizer = recognizer
        
    def preprocess_image(self, image):
        """
        预处理图像：Otsu阈值化 + 膨胀 + 闭运算
        返回处理后的closed图像（用于轮廓检测）和binary图像（用于数字提取）
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 放大gray五倍
        gray = cv2.resize(gray, (gray.shape[1]*5, gray.shape[0]*5), interpolation=cv2.INTER_LINEAR)
        # Otsu阈值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 膨胀和闭运算 (仅用于轮廓检测)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closed, binary

    def find_digit_contour(self, closed_image):
        """在closed图像中找到最大的轮廓（假设是数字）"""
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 检查轮廓面积是否足够大
        if cv2.contourArea(largest_contour) < 20:
            return None

        return largest_contour
    
    def get_bounding_rect_info(self, contour):
        """获取轮廓的最小外接矩形信息"""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 获取矩形的宽度和高度
        width = rect[1][0]
        height = rect[1][1]
        angle = rect[2]
        
        # 确定短边和长边
        if width > height:
            short_side = height
            long_side = width
            is_width_longer = True
        else:
            short_side = width
            long_side = height
            is_width_longer = False
            
        return rect, box, short_side, long_side, is_width_longer, angle
    
    def rotate_image_to_align_short_side(self, image, rect, target_position='bottom'):
        """
        旋转图像使短边对齐到目标位置
        target_position: 'bottom' 或 'top'
        
        对于轴对齐的外接矩形, 把它旋转到短边朝下/朝上
        """
        center = rect[0]
        width, height = rect[1]
        
        # 确定旋转角度
        if width < height:  # ！！！注意是width < height！！！
            if target_position == 'top':
                rotation_angle = 90
            else:  # bottom
                rotation_angle = 270
        else:  
            if target_position == 'top':
                rotation_angle = 0
            else:  # bottom
                rotation_angle = 180
        
        # 执行旋转
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        
        return rotated, rotation_angle
    
    def extract_and_resize_digit(self, binary_image, contour, target_height=20):
        """从binary图像中提取并调整数字区域的高度"""
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 裁切数字区域 (使用binary图像)
        digit_roi = binary_image[y:y+h, x:x+w]
        
        if digit_roi.size == 0:
            return None
        
        # 计算新的宽度保持宽高比
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        
        # 调整大小
        resized = cv2.resize(digit_roi, (new_width, target_height))
        
        return resized
    
    def create_28x28_canvas(self, digit_image):
        """将数字图像放置在28x28的黑色画布上"""
        if digit_image is None:
            return None
            
        canvas = np.zeros((28, 28), dtype=np.uint8)
        h, w = digit_image.shape[:2]
        
        # 计算居中位置
        start_x = max(0, (28 - w) // 2)
        start_y = max(0, (28 - h) // 2)
        
        # 确保不超出边界
        end_x = min(28, start_x + w)
        end_y = min(28, start_y + h)
        
        # 相应调整digit_image的尺寸
        actual_w = end_x - start_x
        actual_h = end_y - start_y
        
        if actual_w > 0 and actual_h > 0:
            digit_resized = cv2.resize(digit_image, (actual_w, actual_h))
            canvas[start_y:end_y, start_x:end_x] = digit_resized
        
        return canvas
    
    def process_cropped_square(self, cropped_image):
        """
        处理单个裁切的正方形，返回两个旋转版本的识别结果
        """
        results = []
        
        # 添加5像素的内边距裁切
        h, w = cropped_image.shape[:2]
        inset = 5
        if h > 2 * inset and w > 2 * inset:
            cropped_image = cropped_image[inset:h-inset, inset:w-inset]
        
        # 预处理图像，获取closed(用于轮廓)和binary(用于数字提取)
        closed, binary = self.preprocess_image(cropped_image)
        
        # 在closed图像中找到数字轮廓
        contour = self.find_digit_contour(closed)

        debug_info = {
            'closed': closed,
            'binary': binary,
            'contour': contour
        }
        
        if contour is None:
            return [], debug_info  # 返回空结果和调试信息
        
        # 获取最小外接矩形信息
        rect, box, short_side, long_side, is_width_longer, angle = self.get_bounding_rect_info(contour)
        
        # 添加矩形信息到调试信息
        debug_info['rect'] = rect
        debug_info['box'] = box
        
        # 生成两个旋转版本
        for target_pos in ['bottom', 'top']:
            # 旋转closed图像来重新寻找轮廓
            rotated_closed, rotation_angle = self.rotate_image_to_align_short_side(
                closed, rect, target_pos
            )
            
            # 同样旋转binary图像用于数字提取
            rotated_binary, _ = self.rotate_image_to_align_short_side(
                binary, rect, target_pos
            )
            
            # 在旋转后的closed图像中重新找到轮廓（因为旋转后位置改变了）
            new_contours, _ = cv2.findContours(rotated_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not new_contours:
                continue
                
            new_contour = max(new_contours, key=cv2.contourArea)
            
            # 从旋转后的binary图像中提取并调整数字大小
            digit_20px = self.extract_and_resize_digit(rotated_binary, new_contour)
            
            # 创建28x28画布
            canvas_28x28 = self.create_28x28_canvas(digit_20px)
            
            if canvas_28x28 is not None:
                # 识别数字
                predicted_digit, confidence = self.recognizer.predict(canvas_28x28)
                
                results.append({
                    'orientation': target_pos,
                    'rotation_angle': rotation_angle,
                    'binary_image': rotated_binary,  # 显示用于提取的binary图像
                    'digit_20px': digit_20px,
                    'canvas_28x28': canvas_28x28,
                    'predicted_digit': predicted_digit,
                    'confidence': confidence
                })
        
        return results, debug_info

def main():
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
        return

    # 初始化数字识别器
    try:
        recognizer = MNISTDigitRecognizer()
        extractor = DigitExtractor(recognizer)
    except FileNotFoundError as e:
        print(f"模型加载失败: {e}")
        return

    # 绘制检测结果
    detected_image = image.copy()
    for i, square in enumerate(squares):
        points = square['params'].astype(np.int32)
        cv2.polylines(detected_image, [points], True, (0, 255, 0), 2)
        # 标记序号
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(detected_image, str(i+1), tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 只处理第一个正方形，展示完整过程
    square = squares[0]
    print(f"详细处理第 1 个正方形...")
    corners = square['params']
    sorted_corners = detector.sort_corners(corners)
    cropped = detector.rotate_and_crop(image, sorted_corners)
    
    # 处理裁切的正方形进行数字识别
    digit_results, debug_info = extractor.process_cropped_square(cropped)
    
    # 创建详细的可视化显示
    # 计算布局：原图 + 裁切图 + 二值化图 + 轮廓图 + 外接矩形图 + 每个旋转方向的详细步骤
    num_steps = 5 + len(digit_results) * 3  # 原图、裁切图、二值化图、轮廓图、外接矩形图 + 每个方向3个步骤
    cols = 4
    rows = (num_steps + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    plot_idx = 1
    
    # 1. 显示原图和检测结果
    plt.subplot(rows, cols, plot_idx)
    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.title("1. Original with Detected Squares")
    plt.axis('off')
    plot_idx += 1
    
    # 2. 显示裁切后的原图
    plt.subplot(rows, cols, plot_idx)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("2. S1 - Cropped Square")
    plt.axis('off')
    plot_idx += 1
    
    # 3. 显示二值化处理后的图像
    plt.subplot(rows, cols, plot_idx)
    plt.imshow(debug_info['binary'], cmap='gray')
    plt.title("3. S1 - Binary Processed")
    plt.axis('off')
    plot_idx += 1
    
    # 4. 显示检测到的轮廓
    plt.subplot(rows, cols, plot_idx)
    if debug_info['contour'] is not None:
        # 创建轮廓显示图像
        contour_image = cv2.cvtColor(debug_info['closed'], cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_image, [debug_info['contour']], -1, (0, 255, 0), 2)
        plt.imshow(contour_image)
        plt.title("4. S1 - Detected Contour")
    else:
        plt.imshow(debug_info['closed'], cmap='gray')
        plt.title("4. S1 - No Contour Found")
    plt.axis('off')
    plot_idx += 1
    
    # 5. 显示最小外接矩形
    plt.subplot(rows, cols, plot_idx)
    if debug_info['contour'] is not None and 'box' in debug_info:
        # 创建外接矩形显示图像
        rect_image = cv2.cvtColor(debug_info['closed'], cv2.COLOR_GRAY2RGB)
        cv2.drawContours(rect_image, [debug_info['contour']], -1, (0, 255, 0), 2)
        cv2.drawContours(rect_image, [debug_info['box']], -1, (255, 0, 0), 2)
        plt.imshow(rect_image)
        plt.title("5. S1 - Min Area Rect")
    else:
        plt.imshow(debug_info['closed'], cmap='gray')
        plt.title("5. S1 - No Rect Found")
    plt.axis('off')
    plot_idx += 1
    
    # 6-N. 显示每个旋转方向的详细步骤
    for j, digit_result in enumerate(digit_results):
        orientation = digit_result['orientation']
        
        # 显示旋转后的二值化图像
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(digit_result['binary_image'], cmap='gray')
        plt.title(f"6.{j+1}a. S1-{orientation} Rotated Binary")
        plt.axis('off')
        plot_idx += 1
        
        # 显示提取的20px数字
        if digit_result['digit_20px'] is not None:
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(digit_result['digit_20px'], cmap='gray')
            plt.title(f"6.{j+1}b. S1-{orientation} Digit 20px")
            plt.axis('off')
        else:
            plt.subplot(rows, cols, plot_idx)
            plt.text(0.5, 0.5, 'No digit\ndetected', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f"6.{j+1}b. S1-{orientation} No Digit")
            plt.axis('off')
        plot_idx += 1
        
        # 显示28x28画布和识别结果
        if digit_result['canvas_28x28'] is not None:
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(digit_result['canvas_28x28'], cmap='gray')
            plt.title(f"6.{j+1}c. S1-{orientation}\nDigit: {digit_result['predicted_digit']} "
                     f"({digit_result['confidence']:.2f})")
            plt.axis('off')
        else:
            plt.subplot(rows, cols, plot_idx)
            plt.text(0.5, 0.5, 'No canvas\ncreated', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f"6.{j+1}c. S1-{orientation} No Canvas")
            plt.axis('off')
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # 打印S1的详细结果
    print("\n=== S1 数字识别详细结果 ===")
    print(f"正方形 1:")
    
    if not digit_results:
        print("  未检测到数字")
    else:
        for digit_result in digit_results:
            print(f"  {digit_result['orientation']} 方向:")
            print(f"    旋转角度: {digit_result['rotation_angle']:.1f}°")
            print(f"    识别数字: {digit_result['predicted_digit']}")
            print(f"    置信度: {digit_result['confidence']:.3f}")
            print()
    
    # 新增：处理所有正方形并选择最佳识别结果
    print("\n=== 处理所有正方形 ===")
    all_square_results = []
    
    for i, square in enumerate(squares):
        print(f"处理正方形 {i+1}...")
        corners = square['params']
        sorted_corners = detector.sort_corners(corners)
        cropped = detector.rotate_and_crop(image, sorted_corners)
        
        # 处理裁切的正方形进行数字识别
        digit_results, _ = extractor.process_cropped_square(cropped)
        
        if digit_results:
            # 选择置信度最高的结果
            best_result = max(digit_results, key=lambda x: x['confidence'])
            all_square_results.append({
                'square_index': i,
                'center': np.mean(corners, axis=0).astype(int),
                'best_result': best_result
            })
            print(f"  最佳结果: {best_result['orientation']} 方向")
            print(f"  识别数字: {best_result['predicted_digit']}")
            print(f"  置信度: {best_result['confidence']:.3f}")
        else:
            all_square_results.append({
                'square_index': i,
                'center': np.mean(corners, axis=0).astype(int),
                'best_result': None
            })
            print(f"  未检测到数字")
    
    # 新增：创建结果总览窗口
    plt.figure(figsize=(12, 6))
    
    # 左侧：原图
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # 右侧：带识别结果的图像
    plt.subplot(1, 2, 2)
    result_image = image.copy()
    
    # 绘制正方形和识别结果
    for i, square in enumerate(squares):
        points = square['params'].astype(np.int32)
        
        # 根据是否有识别结果选择颜色
        square_result = all_square_results[i]
        if square_result['best_result'] is not None:
            color = (0, 255, 0)  # 绿色：有识别结果
            
            # 在正方形中心标注识别的数字和置信度
            center = square_result['center']
            digit = square_result['best_result']['predicted_digit']
            confidence = square_result['best_result']['confidence']
            
            # 绘制白色背景圆圈
            cv2.circle(result_image, tuple(center), 25, (255, 255, 255), -1)
            cv2.circle(result_image, tuple(center), 25, color, 2)
            
            # 标注数字
            cv2.putText(result_image, str(digit), 
                       (center[0]-10, center[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # 在正方形下方标注置信度
            cv2.putText(result_image, f"{confidence:.2f}", 
                       (center[0]-20, center[1]+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            color = (0, 0, 255)  # 红色：无识别结果
            center = square_result['center']
            
            # 绘制红色X标记
            cv2.line(result_image, (center[0]-15, center[1]-15), 
                    (center[0]+15, center[1]+15), color, 3)
            cv2.line(result_image, (center[0]-15, center[1]+15), 
                    (center[0]+15, center[1]-15), color, 3)
        
        # 绘制正方形边框
        cv2.polylines(result_image, [points], True, color, 2)
        
        # 标记正方形序号
        cv2.putText(result_image, f"S{i+1}", 
                   (points[0][0]-10, points[0][1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Recognition Results (Best Confidence)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 打印总体结果统计
    print("\n=== 总体识别结果 ===")
    successful_count = sum(1 for result in all_square_results if result['best_result'] is not None)
    print(f"成功识别: {successful_count}/{len(squares)} 个正方形")
    
    for result in all_square_results:
        i = result['square_index']
        if result['best_result'] is not None:
            best = result['best_result']
            print(f"S{i+1}: 数字 {best['predicted_digit']} (置信度: {best['confidence']:.3f}, {best['orientation']}方向)")
        else:
            print(f"S{i+1}: 未识别")

if __name__ == "__main__":
    main()