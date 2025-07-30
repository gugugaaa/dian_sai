# shape_detector.py
import cv2
import numpy as np

class ShapeDetector:
    def detect_shape(self, frame):
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 阈值化找到黑色形状（白色背景）
        _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0, None
        
        # 找到目标形状轮廓（最大的内部轮廓）
        target_contour = max(contours, key=cv2.contourArea)
        
        # 逼近轮廓
        epsilon = 0.02 * cv2.arcLength(target_contour, True)
        approx = cv2.approxPolyDP(target_contour, epsilon, True)
        
        num_sides = len(approx)
        shape_params = None
        
        if num_sides == 3:
            shape = "triangle"
            # 返回三个角点
            shape_params = approx.reshape(-1, 2)
        elif num_sides == 4:
            shape = "square"
            # 返回四个角点
            shape_params = approx.reshape(-1, 2)
        else:
            # 检查圆形使用圆度
            area = cv2.contourArea(target_contour)
            perimeter = cv2.arcLength(target_contour, True)
            if perimeter == 0:
                shape = None
            else:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.8:
                    shape = "circle"
                    # 返回圆心和半径
                    (x, y), radius = cv2.minEnclosingCircle(target_contour)
                    shape_params = {"center": (int(x), int(y)), "radius": int(radius)}
                else:
                    shape = None
        
        if shape is None:
            return None, 0, None
        
        # 计算像素尺寸（边长或直径）
        if shape == "circle":
            x_pix = 2 * shape_params["radius"]
        else:
            # 对于三角形或正方形，计算平均边长
            sides = []
            for i in range(num_sides):
                p1 = approx[i][0]
                p2 = approx[(i+1) % num_sides][0]
                side = np.linalg.norm(p1 - p2)
                sides.append(side)
            x_pix = np.mean(sides)
        
        return shape, x_pix, shape_params
    
    def draw_shape(self, frame, shape_type, shape_params, color=(0, 255, 0), thickness=2):
        """
        在画面上绘制检测到的形状
        Args:
            frame: 输入图像
            shape_type: 形状类型 ("triangle", "square", "circle")
            shape_params: 形状参数
            color: 绘制颜色，默认绿色
            thickness: 线条粗细，默认2
        Returns:
            绘制后的图像
        """
        if shape_params is None:
            return frame
        
        result_frame = frame.copy()
        
        if shape_type == "circle":
            center = shape_params["center"]
            radius = shape_params["radius"]
            cv2.circle(result_frame, center, radius, color, thickness)
        elif shape_type in ["triangle", "square"]:
            # 绘制多边形
            points = shape_params.astype(np.int32)
            cv2.polylines(result_frame, [points], True, color, thickness)
        
        return result_frame
    
    def calculate_X(self, x_pix, D, K, a4_corners=None):
        """
        计算实际尺寸 - 综合三种方法
        Args:
            x_pix: 像素尺寸
            D: 距离值 (cm)
            K: 相机内参矩阵
            a4_corners: A4纸的四个角点 (4x2)
        Returns:
            x: 实际尺寸 (cm)
        """
        # A4纸黑边内边缘尺寸 (cm)
        A4_WIDTH = 17.0
        A4_HEIGHT = 25.7
        
        results = []
        weights = []
        
        # 方法1: 内参计算公式
        x1 = x_pix * D / K[1, 1]
        results.append(x1)
        weights.append(1.0)  # 基础权重
        
        if a4_corners is not None:
            # 计算A4纸的像素宽度和高度
            # corners顺序为: 左上, 右上, 右下, 左下
            width_top = np.linalg.norm(a4_corners[1] - a4_corners[0])
            width_bottom = np.linalg.norm(a4_corners[2] - a4_corners[3])
            height_left = np.linalg.norm(a4_corners[3] - a4_corners[0])
            height_right = np.linalg.norm(a4_corners[2] - a4_corners[1])
            
            # 取平均值
            a4_width_pix = (width_top + width_bottom) / 2
            a4_height_pix = (height_left + height_right) / 2
            
            # 方法2: 基于A4纸宽度的比例计算
            if a4_width_pix > 0:
                x2 = x_pix * A4_WIDTH / a4_width_pix
                results.append(x2)
                weights.append(1.5)  # A4参考通常更准确
            
            # 方法3: 基于A4纸高度的比例计算
            if a4_height_pix > 0:
                x3 = x_pix * A4_HEIGHT / a4_height_pix
                results.append(x3)
                weights.append(1.5)  # A4参考通常更准确
        
        # 加权平均计算最终结果
        if len(results) > 1:
            weighted_sum = sum(r * w for r, w in zip(results, weights))
            total_weight = sum(weights)
            final_result = weighted_sum / total_weight
            
            # 可选：输出调试信息
            print(f"尺寸计算结果: 内参法={results[0]:.2f}cm", end="")
            if len(results) > 1:
                print(f", A4宽度法={results[1]:.2f}cm", end="")
            if len(results) > 2:
                print(f", A4高度法={results[2]:.2f}cm", end="")
            print(f" -> 综合结果={final_result:.2f}cm")
            
            return final_result
        else:
            return results[0]