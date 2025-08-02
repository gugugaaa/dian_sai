import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from moving_average_filter import MovingAverageFilter

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

class DigitRecognizer:
    """改进的MNIST数字识别器"""
    def __init__(self, model_path='models/improved_mnist_model.pth'):
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到训练好的模型文件 '{model_path}'。"
                "请先运行改进的训练脚本。"
            )
        
        # 加载改进的PyTorch MNIST模型
        self.model = ImprovedCNN(dropout_rate=0.5)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
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
    
    def __init__(self, recognizer, inset_pixels=5, target_height=20, upscale_factor=5):
        self.recognizer = recognizer
        self.inset_pixels = inset_pixels  # 内边距裁切像素数
        self.target_height = target_height  # 数字调整后的目标高度
        self.upscale_factor = upscale_factor  # 图像放大倍数
        
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
        
        # 放大图像
        gray = cv2.resize(gray, (gray.shape[1]*self.upscale_factor, gray.shape[0]*self.upscale_factor), 
                         interpolation=cv2.INTER_LINEAR)
        # Otsu阈值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 膨胀和闭运算 (仅用于轮廓检测)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closed, binary

    def find_digit_contour(self, closed_image, min_contour_area=20):
        """在closed图像中找到最大的轮廓（假设是数字）"""
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 检查轮廓面积是否足够大
        if cv2.contourArea(largest_contour) < min_contour_area:
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
    
    def extract_and_resize_digit(self, binary_image, contour):
        """从binary图像中提取并调整数字区域的高度"""
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 裁切数字区域 (使用binary图像)
        digit_roi = binary_image[y:y+h, x:x+w]
        
        if digit_roi.size == 0:
            return None
        
        # 计算新的宽度保持宽高比
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        
        # 调整大小
        resized = cv2.resize(digit_roi, (new_width, self.target_height))
        
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
        
        # 添加内边距裁切
        h, w = cropped_image.shape[:2]
        if h > 2 * self.inset_pixels and w > 2 * self.inset_pixels:
            cropped_image = cropped_image[self.inset_pixels:h-self.inset_pixels, 
                                        self.inset_pixels:w-self.inset_pixels]
        
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

class DigitDetectionAlgorithm:
    """数字检测算法类 - 封装完整的从post_crop到数字识别的流程"""
    
    def __init__(self, measurement_system, **kwargs):
        """
        初始化数字检测算法
        
        Args:
            measurement_system: 测量系统实例
            **kwargs: 可调参数
                - model_path: 模型文件路径 (默认: 'models/improved_mnist_model.pth')
                - filter_window_size: 移动平均值窗口大小 (默认: 10)
                - confidence_threshold: 置信度阈值 (默认: 0.5)
                - inset_pixels: 内边距裁切像素 (默认: 5)
                - target_height: 数字目标高度 (默认: 20)
                - upscale_factor: 图像放大倍数 (默认: 5)
                - min_contour_area: 最小轮廓面积 (默认: 20)
                - enable_size_filtering: 是否启用尺寸过滤 (默认: True)
                - min_square_size_cm: 最小正方形尺寸(cm) (默认: 1.0)
                - max_square_size_cm: 最大正方形尺寸(cm) (默认: 20.0)
        """
        
        # 存储测量系统引用
        self.measurement_system = measurement_system
        self.shape_detector = measurement_system.shape_detector
        
        # 可调参数
        self.config = {
            'model_path': kwargs.get('model_path', 'models/improved_mnist_model.pth'),
            'filter_window_size': kwargs.get('filter_window_size', 10),
            'confidence_threshold': kwargs.get('confidence_threshold', 0.5),
            'inset_pixels': kwargs.get('inset_pixels', 5),
            'target_height': kwargs.get('target_height', 20),
            'upscale_factor': kwargs.get('upscale_factor', 5),
            'min_contour_area': kwargs.get('min_contour_area', 20),
            'enable_size_filtering': kwargs.get('enable_size_filtering', True),
            'min_square_size_cm': kwargs.get('min_square_size_cm', 1.0),
            'max_square_size_cm': kwargs.get('max_square_size_cm', 20.0)
        }
        
        # 初始化组件
        try:
            self.recognizer = DigitRecognizer(self.config['model_path'])
            self.extractor = DigitExtractor(
                self.recognizer, 
                inset_pixels=self.config['inset_pixels'],
                target_height=self.config['target_height'],
                upscale_factor=self.config['upscale_factor']
            )
            self.avg_filter = MovingAverageFilter(window_size=self.config['filter_window_size'])
            
            print("数字检测算法初始化成功")
            print(f"配置参数: {self.config}")
            
        except Exception as e:
            raise RuntimeError(f"数字检测算法初始化失败: {e}")
    
    def update_config(self, **kwargs):
        """动态更新配置参数"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"已更新参数 {key}: {value}")
            else:
                print(f"警告: 未知参数 {key}")
        
        # 如果更新了滤波器窗口大小，需要重新创建滤波器
        if 'filter_window_size' in kwargs:
            self.avg_filter = MovingAverageFilter(window_size=self.config['filter_window_size'])
    
    def process(self, post_cropped_frame, adjusted_corners, D_corrected, K):
        """
        处理post_crop后的图像，返回正方形-数字-置信度映射
        
        Args:
            post_cropped_frame: post_crop后的图像
            adjusted_corners: 调整后的A4纸角点
            D_corrected: 修正后的距离
            K: 相机内参矩阵
            
        Returns:
            dict: 包含所有检测结果的字典
        """
        try:
            # 1. 检测正方形
            squares = self.shape_detector.detect_squares(post_cropped_frame)
            
            if not squares:
                return {
                    'squares': [],
                    'statistics': {
                        'total_squares': 0,
                        'recognized_digits': 0,
                        'min_size_filtered': None,
                        'distance_filtered': D_corrected,
                        'filter_samples': len(self.avg_filter.size_history)
                    },
                    'success': True,
                    'message': 'no squares detected'
                }
            
            # 2. 处理每个正方形
            square_results = []
            recognized_count = 0
            
            for i, square in enumerate(squares):
                square_info = self._process_single_square(
                    square, post_cropped_frame, adjusted_corners, D_corrected, K, i
                )
                
                # 应用尺寸过滤
                if self.config['enable_size_filtering']:
                    size_cm = square_info['size_cm_raw']
                    if (size_cm < self.config['min_square_size_cm'] or 
                        size_cm > self.config['max_square_size_cm']):
                        square_info['filtered_out'] = True
                        square_info['filter_reason'] = f"尺寸超出范围 [{self.config['min_square_size_cm']}, {self.config['max_square_size_cm']}]cm"
                        continue
                
                square_results.append(square_info)
                
                if square_info['recognition_success']:
                    recognized_count += 1
            
            # 3. 应用移动平均值滤波器（针对最小正方形）
            min_size_filtered = None
            if square_results:
                # 找到尺寸最小的正方形
                valid_squares = [s for s in square_results if not s.get('filtered_out', False)]
                if valid_squares:
                    min_square = min(valid_squares, key=lambda s: s['size_cm_raw'])
                    min_size_filtered, distance_filtered = self.avg_filter.update(
                        min_square['size_cm_raw'], D_corrected
                    )
                    
                    # 计算滤波比例并更新所有正方形的滤波尺寸
                    if min_square['size_cm_raw'] > 0:
                        filter_ratio = min_size_filtered / min_square['size_cm_raw']
                        for square_info in square_results:
                            if not square_info.get('filtered_out', False):
                                square_info['size_cm'] = square_info['size_cm_raw'] * filter_ratio
                    else:
                        # 如果最小正方形尺寸为0，保持原始值
                        for square_info in square_results:
                            if not square_info.get('filtered_out', False):
                                square_info['size_cm'] = square_info['size_cm_raw']


            # 4. 构建返回结果
            result = {
                'squares': square_results,
                'statistics': {
                    'total_squares': len(squares),
                    'valid_squares': len(square_results),
                    'recognized_digits': recognized_count,
                    'min_size_filtered': min_size_filtered,
                    'distance_filtered': D_corrected,  # 这里可以用distance_filtered如果需要
                    'filter_samples': len(self.avg_filter.size_history)
                },
                'success': True,
                'message': f'detected {len(squares)} squares, recognized {recognized_count} digits'
            }
            
            return result
            
        except Exception as e:
            return {
                'squares': [],
                'statistics': {
                    'total_squares': 0,
                    'recognized_digits': 0,
                    'min_size_filtered': None,
                    'distance_filtered': D_corrected,
                    'filter_samples': len(self.avg_filter.size_history)
                },
                'success': False,
                'message': f'error: {str(e)}'
            }
    
    def _process_single_square(self, square, post_cropped_frame, adjusted_corners, D_corrected, K, square_id):
        """处理单个正方形"""
        corners_sq = square['params']
        
        # 计算正方形的边长（像素）
        sides = []
        for j in range(len(corners_sq)):
            p1 = corners_sq[j]
            p2 = corners_sq[(j + 1) % len(corners_sq)]
            side = np.linalg.norm(p1 - p2)
            sides.append(side)
        x_pix = np.mean(sides)
        
        # 计算实际尺寸
        x_cm = self.shape_detector.calculate_X(x_pix, D_corrected, K, adjusted_corners)
        
        # 获取正方形中心点
        center = np.mean(corners_sq, axis=0).astype(int)
        
        # 裁切和识别数字
        sorted_corners = self.shape_detector.sort_corners(corners_sq)
        cropped_square = self.shape_detector.rotate_and_crop(post_cropped_frame, sorted_corners)
        
        # 数字识别
        digit_result = self.extractor.process_cropped_square(cropped_square)
        
        # 构建结果
        square_info = {
            'id': square_id,
            'corners': corners_sq,
            'center': center.tolist(),
            'size_cm_raw': x_cm,
            'size_cm': x_cm,  # 初始值，稍后可能被滤波值覆盖
            'size_pixels': x_pix,
            'cropped_image': cropped_square,
            'recognition_success': False,
            'digit': None,
            'confidence': 0.0,
            'recognition_details': None,
            'filtered_out': False,
            'filter_reason': None
        }
        
        if digit_result is not None:
            # 检查置信度阈值
            if digit_result['confidence'] >= self.config['confidence_threshold']:
                square_info.update({
                    'recognition_success': True,
                    'digit': digit_result['predicted_digit'],
                    'confidence': digit_result['confidence'],
                    'recognition_details': {
                        'orientation': digit_result['orientation'],
                        'rotation_angle': digit_result['rotation_angle'],
                        'canvas_28x28': digit_result['canvas_28x28']
                    }
                })
            else:
                square_info['filter_reason'] = f"置信度{digit_result['confidence']:.3f}低于阈值{self.config['confidence_threshold']}"
        
        return square_info
    
    def reset_filter(self):
        """重置移动平均值滤波器"""
        self.avg_filter.reset()
        print("已重置移动平均值滤波器")
    
    def get_config(self):
        """获取当前配置"""
        return self.config.copy()
    
    def get_statistics(self):
        """获取当前统计信息"""
        return {
            'filter_samples': len(self.avg_filter.size_history),
            'size_history': self.avg_filter.size_history.copy(),
            'distance_history': self.avg_filter.distance_history.copy()
        }