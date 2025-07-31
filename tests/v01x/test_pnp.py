import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from border_detector import BorderDetector
from distance_calculator import DistanceCalculator

# 创建默认相机内参矩阵
K = np.array([
    [1144, 0, 1920/2],
    [0, 1144, 1080/2],
    [0, 0, 1]
], dtype=np.float32)

detector = BorderDetector()
distance_calc = DistanceCalculator(K)
img_dir = os.path.join(os.path.dirname(__file__), "../../images/fake_angle")
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for fname in img_files:
    img_path = os.path.join(img_dir, fname)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    found, corners = detector.detect_border(edges, img)
    vis = img.copy()
    if found:
        pts = corners.astype(int)
        
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,0), 2)
        
        # 计算距离和角度
        D, theta_deg = distance_calc.calculate_D(corners)
        if D is not None:
            # 在图像上显示距离和角度
            cv2.putText(vis, f"Distance: {D:.1f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(vis, f"Angle: {theta_deg:.1f} deg", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    else:
        cv2.putText(vis, "No border found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow(f"Result - {fname}", vis)
    cv2.waitKey(0)
cv2.destroyAllWindows()