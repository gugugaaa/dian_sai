import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from system_initializer import MeasurementSystem

# 初始化系统
system = MeasurementSystem("calib.yaml",500)

# 主循环
while True:
    frame = system.capture_frame()
    if frame is None:
        break
    
    
    # 调用预处理
    edges = system.preprocessor.preprocess(frame)
    
    # 调用pre_crop裁切
    cropped, success = system.preprocessor.pre_crop(frame)
    
    # 显示裁切结果
    if success and cropped is not None:
        cv2.imshow("Pre-cropped", cropped)
    else:
        # 如果裁切失败，显示原图
        cv2.imshow("Origin", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
system.cap.release()
cv2.destroyAllWindows()

