import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from system_initializer import MeasurementSystem

# 初始化系统
system = MeasurementSystem("calib.yaml", 500)

# 主循环
while True:
    frame = system.capture_frame()
    if frame is None:
        break
    
    # 调用pre_crop裁切
    cropped_frame, success = system.preprocessor.pre_crop(frame)
    
    if not success or cropped_frame is None:
        # 如果裁切失败，显示原图
        cv2.imshow("Origin", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 预处理（去畸变和边缘检测，使用裁剪后的帧）
    edges = system.preprocessor.preprocess(cropped_frame)
    
    # 检测A4纸边框并获取角点
    ok, corners = system.border_detector.detect_border(edges, cropped_frame)
    if not ok:
        print("无法检测A4边框")
        cv2.imshow("Shape Detection", cropped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 基于A4边框进行后裁切，避免边框干扰形状检测
    post_cropped_frame, adjusted_corners = system.border_detector.post_crop(cropped_frame, corners, inset_pixels=5)
    
    # 使用PnP计算距离D
    D_raw, _ = system.distance_calculator.calculate_D(corners, system.K)
    if D_raw is None:
        print("PnP求解失败")
        cv2.imshow("Shape Detection", post_cropped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 检测形状 - 使用后裁剪后的画面
    squares = system.shape_detector.detect_squares_debug(post_cropped_frame)
    
    # 显示结果
    result_frame = post_cropped_frame.copy()
    for i, square in enumerate(squares):
        points = square['params'].astype(np.int32)
        cv2.polylines(result_frame, [points], True, (0, 255, 0), 2)
        # 标记中心点
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(result_frame, f"Square {i+1}", tuple(center-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Shape Detection", result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
system.cap.release()
cv2.destroyAllWindows()
