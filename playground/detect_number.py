import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from system_initializer import MeasurementSystem

class MNISTDigitRecognizer:
    """MNIST数字识别器"""
    def __init__(self):
        # 检查模型文件是否存在
        if not os.path.exists('models/mnist_model.h5'):
            raise FileNotFoundError(
                "未找到训练好的模型文件 'mnist_model.h5'。"
                "请先运行 'python train_mnist_model.py' 来训练模型。"
            )
        
        # 加载预训练的MNIST模型
        print("加载MNIST模型...")
        self.model = keras.models.load_model('models/mnist_model.h5')
        print("MNIST模型加载完成")
        
        self.digit_images = []  # 存储预处理后的数字图像用于显示
    
    def preprocess_image(self, img_region):
        """预处理图像区域用于MNIST识别"""
        # 转换为灰度图
        if len(img_region.shape) == 3:
            gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_region
        
        # 不做任何模糊处理
        
        # 使用阈值分割来找到白色数字区域（假设黑底白字，高对比度）
        # 这里用Otsu自动阈值，适合二值化
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 找到轮廓（contours），假设数字是一个主要轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果没找到轮廓，返回全黑图像（或处理错误）
            print("未检测到数字轮廓")
            return np.zeros((1, 28, 28), dtype='float32')
        
        # 找到最大的轮廓（假设是数字）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取最小包围矩形（bounding box）来裁切数字区域
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 裁切数字区域（添加少量padding以包含整个数字，避免裁切边缘）
        padding = 2  # 小padding以防裁切太紧
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2 * padding, gray.shape[1])
        h = min(h + 2 * padding, gray.shape[0])
        digit_roi = gray[y:y+h, x:x+w]
        
        # 现在，对digit_roi进行等比缩放，使其最大边接近20像素（MNIST数字大小）
        target_size = 20  # 目标数字大小（MNIST约20x20）
        height, width = digit_roi.shape
        scale = target_size / max(height, width)  # 计算缩放比例，保持纵横比
        
        # 等比缩放，使用INTER_LINEAR插值（平滑，但不模糊过多；可换成INTER_NEAREST避免任何平滑）
        resized_digit = cv2.resize(digit_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # 创建28x28的黑色画布
        canvas = np.zeros((28, 28), dtype=np.uint8)
        
        # 将缩放后的数字居中放置在画布上
        rh, rw = resized_digit.shape
        start_y = (28 - rh) // 2
        start_x = (28 - rw) // 2
        canvas[start_y:start_y + rh, start_x:start_x + rw] = resized_digit
        
        # 反转颜色（如果需要匹配MNIST：黑底白字；你的原始是黑底白字，所以不反转？但原代码有反转，假设MNIST期望白底黑字？）
        # 注意：MNIST标准是背景0（黑），数字1（白），所以如果你的canvas现在是背景0（黑），数字高值（白），就不需反转。
        # 但原代码有 inverted = 255 - resized，如果你确认需要，保留；否则注释掉。
        # 这里假设保留原逻辑：
        inverted = 255 - canvas
        
        # 归一化
        normalized = inverted.astype('float32') / 255.0
        
        # 添加批次维度
        processed_image = normalized.reshape(1, 28, 28)
        
        # 保存用于显示的图像（28x28格式）
        display_image = (normalized * 255).astype(np.uint8)
        self.digit_images.append(display_image)
        
        return processed_image
    
    def predict_digit(self, img_region):
        """预测图像区域中的数字"""
        try:
            # 预处理图像
            processed = self.preprocess_image(img_region)
            
            # 预测
            prediction = self.model.predict(processed, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            return digit, confidence
        except Exception as e:
            print(f"数字识别错误: {e}")
            return None, 0.0
    
    def get_digit_display(self, max_images=10):
        """获取拼接的数字显示图像"""
        if not self.digit_images:
            return np.zeros((28, 28), dtype=np.uint8)
        
        # 限制显示的图像数量
        images_to_show = self.digit_images[-max_images:]
        
        # 计算拼接布局
        num_images = len(images_to_show)
        cols = min(5, num_images)  # 最多5列
        rows = (num_images + cols - 1) // cols  # 计算需要的行数
        
        # 创建拼接画布
        canvas_height = rows * 28
        canvas_width = cols * 28
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        
        # 拼接图像
        for i, img in enumerate(images_to_show):
            row = i // cols
            col = i % cols
            y_start = row * 28
            x_start = col * 28
            canvas[y_start:y_start+28, x_start:x_start+28] = img
        
        return canvas
    
    def clear_digit_images(self):
        """清空数字图像缓存"""
        self.digit_images.clear()

class MovingAverageFilter:
    """移动平均值滤波器，用于减少测量结果的跳动"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.all_sizes_history = []  # 存储所有正方形尺寸
        self.distance_history = []
    
    def update(self, all_sizes, distance):
        """更新历史数据并返回平均值"""
        # 更新所有尺寸历史
        self.all_sizes_history.append(all_sizes)
        if len(self.all_sizes_history) > self.window_size:
            self.all_sizes_history.pop(0)
        
        # 更新距离历史
        self.distance_history.append(distance)
        if len(self.distance_history) > self.window_size:
            self.distance_history.pop(0)
        
        # 返回平均值
        avg_distance = sum(self.distance_history) / len(self.distance_history)
        
        return self.all_sizes_history, avg_distance
    
    def reset(self):
        """重置历史数据"""
        self.all_sizes_history.clear()
        self.distance_history.clear()

def test_shape_detection():
    """测试形状检测功能"""
    print("初始化测量系统...")
    
    try:
        system = MeasurementSystem("calib.yaml", 500)
        print("系统初始化成功")
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return
    
    # 初始化MNIST数字识别器
    digit_recognizer = MNISTDigitRecognizer()
    
    # 初始化移动平均值滤波器
    avg_filter = MovingAverageFilter(window_size=10)
    
    print("开始形状检测和数字识别测试...")
    print("按 'q' 退出, 按 'r' 重置平均值, 按 'c' 清空数字显示")
    
    while True:
        try:
            # 清空本帧的数字图像
            digit_recognizer.digit_images.clear()
            
            # 捕获帧
            frame = system.capture_frame()
            
            # 显示原始摄像头画面，方便瞄准
            cv2.imshow("Camera Feed - Original", frame)
            
            # 预裁剪 - 遵循主函数流程
            cropped_frame, ok = system.preprocessor.pre_crop(frame)
            if not ok:
                print("预裁剪失败，无法检测闭合轮廓")
                # 显示原始帧
                cv2.imshow("Shape Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    avg_filter.reset()
                    print("已重置平均值滤波器")
                continue
            
            # 预处理（去畸变和边缘检测，使用裁剪后的帧）
            edges = system.preprocessor.preprocess(cropped_frame)
            
            # 检测A4纸边框并获取角点
            ok, corners = system.border_detector.detect_border(edges, cropped_frame)
            if not ok:
                print("无法检测A4边框")
                # 显示预裁剪后的帧
                cv2.imshow("Shape Detection", cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    avg_filter.reset()
                    print("已重置平均值滤波器")
                continue
            
            # 基于A4边框进行后裁切，避免边框干扰形状检测
            post_cropped_frame, adjusted_corners = system.border_detector.post_crop(cropped_frame, corners, inset_pixels=5)
            
            # 使用PnP计算距离D - 直接使用system.K
            D_raw, _ = system.distance_calculator.calculate_D(corners, system.K)
            if D_raw is None:
                print("PnP求解失败")
                cv2.imshow("Shape Detection", post_cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    avg_filter.reset()
                    print("已重置平均值滤波器")
                continue
            
            # 直接使用PnP计算的距离，不进行映射校正
            D_corrected = D_raw
            
            # 检测多个正方形 - 使用后裁剪后的画面
            squares = system.shape_detector.detect_squares(post_cropped_frame)
            
            # 创建结果显示图像 - 使用后裁剪后的画面
            result_frame = post_cropped_frame.copy()
            
            # 绘制A4边框角点和边框线（在原始裁剪帧上）
            border_frame = cropped_frame.copy()
            # 绘制角点
            for i, corner in enumerate(corners):
                cv2.circle(border_frame, tuple(corner.astype(int)), 8, (0, 255, 0), 2)
                # 标注角点序号 (左上, 右上, 右下, 左下)
                labels = ['TL', 'TR', 'BR', 'BL']
                cv2.putText(border_frame, labels[i], 
                           tuple((corner + [10, -10]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
            
            # 绘制边框线
            cv2.polylines(border_frame, [corners.astype(int)], True, (255, 0, 0), 2)
            
            if squares:
                # 计算每个正方形的平均边长 x_pix 并进行数字识别
                square_infos = []
                for sq in squares:
                    points = sq['params']
                    sides = []
                    for i in range(len(points)):
                        p1 = points[i]
                        p2 = points[(i + 1) % len(points)]
                        side = np.linalg.norm(p1 - p2)
                        sides.append(side)
                    x_pix = np.mean(sides)
                    
                    # 提取正方形内的图像区域进行数字识别
                    mask = np.zeros(post_cropped_frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points.astype(int)], 255)
                    
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(points.astype(int))
                    
                    # 确保边界框在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, post_cropped_frame.shape[1] - x)
                    h = min(h, post_cropped_frame.shape[0] - y)
                    
                    if w > 0 and h > 0:
                        # 提取区域
                        roi = post_cropped_frame[y:y+h, x:x+w]
                        roi_mask = mask[y:y+h, x:x+w]
                        
                        # 应用遮罩
                        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)
                        
                        # 数字识别
                        digit, confidence = digit_recognizer.predict_digit(masked_roi)
                    else:
                        digit, confidence = None, 0.0
                    
                    square_infos.append({
                        'x_pix': x_pix, 
                        'params': points,
                        'digit': digit,
                        'confidence': confidence
                    })
                
                # 计算所有正方形的尺寸
                all_sizes = []
                for sq_info in square_infos:
                    x = system.shape_detector.calculate_X(sq_info['x_pix'], D_corrected, system.K, adjusted_corners)
                    x_corrected = x - 0  # 修正值
                    all_sizes.append(x_corrected)
                
                # 使用移动平均值滤波器
                all_sizes_history, D_avg = avg_filter.update(all_sizes, D_corrected)

                print(f"检测到 {len(squares)} 个正方形")
                print(f"当前正方形尺寸: {[f'{size:.2f}cm' for size in all_sizes]}")
                print(f"距离: {D_avg:.2f}cm")
                
                # 打印所有识别的数字
                for i, sq_info in enumerate(square_infos):
                    if sq_info['digit'] is not None:
                        print(f"正方形 {i+1}: 数字 {sq_info['digit']} (置信度: {sq_info['confidence']:.2f})")
                
                # 绘制所有正方形为同一颜色，并显示识别的数字
                for sq_info in square_infos:
                    result_frame = system.shape_detector.draw_shape(
                        result_frame, 'square', sq_info['params'], color=(0, 255, 0), thickness=3
                    )
                    
                    # 在正方形中心绘制识别的数字
                    if sq_info['digit'] is not None and sq_info['confidence'] > 0.5:
                        center = np.mean(sq_info['params'], axis=0).astype(int)
                        text = f"{sq_info['digit']}"
                        
                        # 设置字体大小（根据正方形大小调整）
                        font_scale = max(0.8, sq_info['x_pix'] / 100)
                        
                        # 绘制数字文本（白色背景黑色文字）
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                        text_x = center[0] - text_size[0] // 2
                        text_y = center[1] + text_size[1] // 2
                        
                        # 绘制白色背景
                        cv2.rectangle(result_frame, 
                                    (text_x - 5, text_y - text_size[1] - 5),
                                    (text_x + text_size[0] + 5, text_y + 5),
                                    (255, 255, 255), -1)
                        
                        # 绘制黑色数字
                        cv2.putText(result_frame, text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
                
                # 在图像上绘制信息（只显示距离和样本数）
                cv2.putText(result_frame, f"Distance: {D_avg:.1f}cm", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(result_frame, f"Samples: {len(avg_filter.distance_history)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(result_frame, f"Detected digits: {len([s for s in square_infos if s['digit'] is not None])}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                print("未检测到形状")
            
            # 显示数字识别区域
            digit_display = digit_recognizer.get_digit_display()
            if digit_display.shape[0] > 0 and digit_display.shape[1] > 0:
                # 放大显示以便观察
                scale_factor = 4
                enlarged_display = cv2.resize(digit_display, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                cv2.imshow("MNIST Digit Regions", enlarged_display)
            
            # 显示结果
            cv2.imshow("A4 Border Detection", border_frame)
            cv2.imshow("Shape Detection", result_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                avg_filter.reset()
                print("已重置平均值滤波器")
            elif key == ord('c'):
                digit_recognizer.clear_digit_images()
                print("已清空数字显示")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            continue
    
    # 清理资源
    cv2.destroyAllWindows()
    system.cap.release()
    print("测试结束")

if __name__ == "__main__":
    test_shape_detection()