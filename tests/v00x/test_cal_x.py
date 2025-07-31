import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from system_initializer import MeasurementSystem

def test_shape_detection():
    """测试形状检测功能"""
    print("初始化测量系统...")
    
    try:
        system = MeasurementSystem("calib.yaml", 500)
        print("系统初始化成功")
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return
    
    print("开始形状检测测试...")
    print("按 'q' 退出")
    
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
                continue
            
            # 基于A4边框进行后裁切，避免边框干扰形状检测
            post_cropped_frame, adjusted_corners = system.border_detector.post_crop(cropped_frame, corners, inset_pixels=5)
            
            # 使用PnP计算距离D
            D, _ = system.distance_calculator.calculate_D(corners)
            if D is None:
                print("PnP求解失败")
                cv2.imshow("Shape Detection", post_cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # 检测形状 - 使用后裁剪后的画面
            shape, x_pix, shape_params = system.shape_detector.detect_shape(post_cropped_frame)
            
            # 创建结果显示图像 - 使用后裁剪后的画面
            result_frame = post_cropped_frame.copy()
            
            if shape:
                # 计算实际尺寸x
                x = system.shape_detector.calculate_X(x_pix, D, system.K, adjusted_corners)
                
                print(f"检测到形状: {shape}, 像素尺寸: {x_pix:.2f}, 距离D: {D:.2f}cm, 实际尺寸: {x:.2f}cm")
                
                # 绘制检测到的形状
                result_frame = system.shape_detector.draw_shape(
                    result_frame, shape, shape_params, color=(0, 255, 0), thickness=3
                )
                
                # 在图像上绘制边长信息
                cv2.putText(result_frame, f"Size: {x:.2f}cm", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                cv2.putText(result_frame, f"Distance: {D:.2f}cm", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            else:
                print("未检测到形状")
            
            # 显示结果
            cv2.imshow("Shape Detection", result_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
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