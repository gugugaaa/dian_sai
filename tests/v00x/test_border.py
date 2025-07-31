import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import glob
from system_initializer import MeasurementSystem

def test_border_detection():
    """测试边框检测并可视化角点"""
    try:
        # 初始化测量系统
        system = MeasurementSystem()
        
        print("按空格键进行边框检测，按ESC退出...")
        
        while True:
            # 捕获帧
            frame = system.capture_frame()
            
            # 显示原始摄像头画面，方便瞄准
            cv2.imshow("Camera Feed - Original", frame)
            # 预裁剪
            cropped_frame, ok = system.preprocessor.pre_crop(frame)
            if not ok:
                print("预裁剪失败，无法检测闭合轮廓")
                cv2.imshow("Original", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                continue
            
            # 预处理（去畸变和边缘检测）
            edges = system.preprocessor.preprocess(cropped_frame)
            
            # 检测A4纸边框并获取角点
            ok, corners = system.border_detector.detect_border(edges, cropped_frame)
            
            # 创建可视化图像
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            if ok:
                # 在edges图像上绘制角点
                for i, corner in enumerate(corners):
                    # 绘制角点圆圈
                    cv2.circle(edges_colored, tuple(corner.astype(int)), 10, (0, 255, 0), 2)
                    # 标注角点序号 (左上, 右上, 右下, 左下)
                    labels = ['TL', 'TR', 'BR', 'BL']
                    cv2.putText(edges_colored, labels[i], 
                               tuple((corner + [10, -10]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 绘制边框线
                cv2.polylines(edges_colored, [corners.astype(int)], True, (255, 0, 0), 2)
                
                print("检测到A4边框角点:")
                for i, corner in enumerate(corners):
                    print(f"  {['左上', '右上', '右下', '左下'][i]}: ({corner[0]:.1f}, {corner[1]:.1f})")
            else:
                print("无法检测A4边框")
                cv2.putText(edges_colored, "No A4 border detected", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示图像
            cv2.imshow("Original Frame", cropped_frame)
            cv2.imshow("Edges with Corners", edges_colored)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                break
            elif key == ord(' '):  # 空格键保存当前检测结果
                if ok:
                    cv2.imwrite(f"border_test_result_{int(cv2.getTickCount())}.jpg", edges_colored)
                    print("检测结果已保存")
        
        # 清理
        system.cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_border_detection()