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

# 导入测量系统和形状检测器
from system_initializer import MeasurementSystem
from shape_detector import ShapeDetector

class ImprovedCNN(nn.Module):
    """改进的CNN架构 - 与训练代码保持一致"""
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Dropout2d(0.25),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MNISTDigitRecognizer:
    """改进的MNIST数字识别器"""
    def __init__(self):
        # 检查模型文件是否存在
        model_path = 'models/improved_mnist_model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到训练好的模型文件 '{model_path}'。"
                "请先运行改进的训练脚本。"
            )
        
        # 加载改进的PyTorch MNIST模型
        print("加载改进的MNIST模型...")
        
        self.model = ImprovedCNN(dropout_rate=0.5)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print("改进的MNIST模型加载完成")
        
        # 使用Otsu预处理，与训练时保持一致
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # 确保是灰度图
            self.otsu_transform,     # 自定义Otsu变换
        ])
    
    def otsu_transform(self, pil_image):
        """应用Otsu阈值化，与训练时保持一致"""
        # 转换为numpy数组
        img_np = np.array(pil_image).astype(np.uint8)
        
        # 使用Otsu阈值化
        _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 转换为float32并归一化到0-1，然后转为tensor
        binary = binary.astype(np.float32) / 255.0
        tensor = torch.from_numpy(binary).unsqueeze(0)  # (1,28,28)
        
        return tensor
    
    def predict(self, image_28x28):
        """
        预测28x28图像中的数字
        Args:
            image_28x28: 28x28的灰度图像 (numpy array)
        Returns:
            predicted_digit, confidence
        """
        # 转换为PIL图像
        pil_image = Image.fromarray(image_28x28.astype(np.uint8))
        
        # 应用预处理变换
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
        """
        center = rect[0]
        width, height = rect[1]
        
        # 确定旋转角度
        if width < height:  # 横向矩形，数字沿高度方向（短边）
            if target_position == 'top':
                rotation_angle = 90
            else:  # bottom
                rotation_angle = 270
        else:  # 纵向矩形，数字沿宽度方向（短边）
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
        处理单个裁切的正方形，返回最佳识别结果
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
        
        if contour is None:
            return None  # 没有找到数字
        
        # 获取最小外接矩形信息
        rect, box, short_side, long_side, is_width_longer, angle = self.get_bounding_rect_info(contour)
        
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
                    'predicted_digit': predicted_digit,
                    'confidence': confidence,
                    'canvas_28x28': canvas_28x28
                })
        
        # 返回置信度最高的结果
        if results:
            return max(results, key=lambda x: x['confidence'])
        else:
            return None

class MovingAverageFilter:
    """移动平均值滤波器，用于减少测量结果的跳动"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.size_history = []
        self.distance_history = []
    
    def update(self, size, distance):
        """更新历史数据并返回平均值"""
        # 更新尺寸历史
        self.size_history.append(size)
        if len(self.size_history) > self.window_size:
            self.size_history.pop(0)
        
        # 更新距离历史
        self.distance_history.append(distance)
        if len(self.distance_history) > self.window_size:
            self.distance_history.pop(0)
        
        # 返回平均值
        avg_size = sum(self.size_history) / len(self.size_history)
        avg_distance = sum(self.distance_history) / len(self.distance_history)
        
        return avg_size, avg_distance
    
    def reset(self):
        """重置历史数据"""
        self.size_history.clear()
        self.distance_history.clear()

def main():
    """实时数字识别主函数"""
    print("初始化测量系统...")
    
    try:
        system = MeasurementSystem("calib.yaml", 500)
        print("测量系统初始化成功")
    except Exception as e:
        print(f"测量系统初始化失败: {e}")
        return
    
    try:
        recognizer = MNISTDigitRecognizer()
        extractor = DigitExtractor(recognizer)
        print("数字识别系统初始化成功")
    except FileNotFoundError as e:
        print(f"数字识别系统初始化失败: {e}")
        print("请先运行改进的训练脚本生成 models/improved_mnist_model.pth")
        return
    
    # 初始化移动平均值滤波器
    avg_filter = MovingAverageFilter(window_size=10)
    
    print("开始实时数字识别测试...")
    print("按 'q' 退出, 按 'r' 重置平均值")
    
    while True:
        try:
            # 捕获帧
            frame = system.capture_frame()
            
            # 显示原始摄像头画面
            cv2.imshow("Camera Feed - Original", frame)
            
            # 预裁剪 - 遵循标准流程
            cropped_frame, ok = system.preprocessor.pre_crop(frame)
            if not ok:
                print("预裁剪失败，无法检测闭合轮廓")
                cv2.imshow("Real-time Digit Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    avg_filter.reset()
                    print("已重置平均值滤波器")
                continue
            
            # 预处理（去畸变和边缘检测）
            edges = system.preprocessor.preprocess(cropped_frame)
            
            # 检测A4纸边框并获取角点
            ok, corners = system.border_detector.detect_border(edges, cropped_frame)
            if not ok:
                print("无法检测A4边框")
                cv2.imshow("Real-time Digit Recognition", cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    avg_filter.reset()
                    print("已重置平均值滤波器")
                continue
            
            # 基于A4边框进行后裁切
            post_cropped_frame, adjusted_corners = system.border_detector.post_crop(cropped_frame, corners, inset_pixels=3)
            
            # 使用PnP计算距离D
            D_raw, _ = system.distance_calculator.calculate_D(corners, system.K)
            if D_raw is None:
                print("PnP求解失败")
                cv2.imshow("Real-time Digit Recognition", post_cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    avg_filter.reset()
                    print("已重置平均值滤波器")
                continue
            
            D_corrected = D_raw
            
            # 检测多个正方形
            squares = system.shape_detector.detect_squares(post_cropped_frame)
            
            # 创建结果显示图像
            result_frame = post_cropped_frame.copy()
            
            if squares:
                print(f"检测到 {len(squares)} 个正方形")
                
                # 对每个正方形进行数字识别
                for i, square in enumerate(squares):
                    corners_sq = square['params']
                    sorted_corners = system.shape_detector.sort_corners(corners_sq)
                    cropped_square = system.shape_detector.rotate_and_crop(post_cropped_frame, sorted_corners)
                    
                    # 数字识别
                    digit_result = extractor.process_cropped_square(cropped_square)
                    
                    # 计算正方形的边长（像素）
                    sides = []
                    for j in range(len(corners_sq)):
                        p1 = corners_sq[j]
                        p2 = corners_sq[(j + 1) % len(corners_sq)]
                        side = np.linalg.norm(p1 - p2)
                        sides.append(side)
                    x_pix = np.mean(sides)
                    
                    # 计算实际尺寸
                    x = system.shape_detector.calculate_X(x_pix, D_corrected, system.K, adjusted_corners)
                    
                    # 绘制正方形
                    if digit_result is not None:
                        # 成功识别数字 - 绿色
                        color = (0, 255, 0)
                        confidence = digit_result['confidence']
                        digit = digit_result['predicted_digit']
                        
                        # 在正方形中心标注识别结果
                        center = np.mean(corners_sq, axis=0).astype(int)
                        
                        # 绘制白色背景圆圈
                        cv2.circle(result_frame, tuple(center), 25, (255, 255, 255), -1)
                        cv2.circle(result_frame, tuple(center), 25, color, 2)
                        
                        # 标注数字
                        cv2.putText(result_frame, str(digit), 
                                   (center[0]-10, center[1]+5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        # 在正方形下方标注信息
                        cv2.putText(result_frame, f"S{i+1}: {x:.1f}cm", 
                                   (center[0]-30, center[1]+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cv2.putText(result_frame, f"Conf: {confidence:.2f}", 
                                   (center[0]-30, center[1]+55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        print(f"  S{i+1}: 数字 {digit} (置信度: {confidence:.3f}, 尺寸: {x:.1f}cm)")
                        
                    else:
                        # 未识别到数字 - 黄色
                        color = (0, 255, 255)
                        center = np.mean(corners_sq, axis=0).astype(int)
                        
                        # 绘制黄色X标记
                        cv2.line(result_frame, (center[0]-15, center[1]-15), 
                                (center[0]+15, center[1]+15), color, 3)
                        cv2.line(result_frame, (center[0]-15, center[1]+15), 
                                (center[0]+15, center[1]-15), color, 3)
                        
                        # 标注尺寸信息
                        cv2.putText(result_frame, f"S{i+1}: {x:.1f}cm", 
                                   (center[0]-30, center[1]+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cv2.putText(result_frame, "No digit", 
                                   (center[0]-30, center[1]+55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        print(f"  S{i+1}: 未识别数字 (尺寸: {x:.1f}cm)")
                    
                    # 绘制正方形边框
                    cv2.polylines(result_frame, [corners_sq.astype(int)], True, color, 2)
                
                # 使用移动平均值滤波器（只对最小正方形）
                if squares:
                    square_infos = []
                    for sq in squares:
                        points = sq['params']
                        sides = []
                        for j in range(len(points)):
                            p1 = points[j]
                            p2 = points[(j + 1) % len(points)]
                            side = np.linalg.norm(p1 - p2)
                            sides.append(side)
                        x_pix = np.mean(sides)
                        x = system.shape_detector.calculate_X(x_pix, D_corrected, system.K, adjusted_corners)
                        square_infos.append({'x': x, 'params': points})
                    
                    min_square = min(square_infos, key=lambda s: s['x'])
                    x_avg, D_avg = avg_filter.update(min_square['x'], D_corrected)
                    
                    # 在图像上显示统计信息
                    cv2.putText(result_frame, f"Min Size: {x_avg:.1f}cm", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Distance: {D_avg:.1f}cm", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(result_frame, f"Squares: {len(squares)}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(result_frame, f"Samples: {len(avg_filter.size_history)}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                print("未检测到正方形")
            
            # 显示结果
            cv2.imshow("Real-time Digit Recognition", result_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                avg_filter.reset()
                print("已重置平均值滤波器")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            continue
    
    # 清理资源
    cv2.destroyAllWindows()
    system.cap.release()
    print("实时数字识别测试结束")

if __name__ == "__main__":
    main()