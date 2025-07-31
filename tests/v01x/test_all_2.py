import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from system_initializer import MeasurementSystem

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

def test_shape_detection():
    """测试形状检测功能"""
    print("初始化测量系统...")
    
    try:
        system = MeasurementSystem("calib.yaml", 500)
        print("系统初始化成功")
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return
    
    # 初始化移动平均值滤波器
    avg_filter = MovingAverageFilter(window_size=10)
    
    print("开始形状检测测试...")
    print("按 'q' 退出, 按 'r' 重置平均值")
    
    while True:
        try:
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
            
            # 检测形状 - 使用后裁剪后的画面
            shape, x_pix, shape_params = system.shape_detector.detect_shape(post_cropped_frame)
            
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
            
            if shape:
                # 计算实际尺寸x（直接使用system.K和adjusted_corners）
                x = system.shape_detector.calculate_X(x_pix, D_corrected, system.K, adjusted_corners)
                
                # 使用移动平均值滤波器
                x_avg, D_avg = avg_filter.update(x, D_corrected)
                
                print(f"检测到形状: {shape}")
                print(f"尺寸: {D_corrected:.2f}cm")
                print(f"平均值 - 尺寸: {x_avg:.2f}cm, 距离: {D_avg:.2f}cm")
                
                # 绘制检测到的形状
                result_frame = system.shape_detector.draw_shape(
                    result_frame, shape, shape_params, color=(0, 255, 0), thickness=3
                )
                
                # 在图像上绘制信息（显示平均值）
                cv2.putText(result_frame, f"Size: {x_avg:.1f}cm", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(result_frame, f"Distance: {D_avg:.1f}cm", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(result_frame, f"Samples: {len(avg_filter.size_history)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                print("未检测到形状")
            
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