import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from border_detector import BorderDetector

detector = BorderDetector()
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
        labels = ['TL', 'TR', 'BR', 'BL']
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,0), 2)
        for i, p in enumerate(pts):
            cv2.circle(vis, tuple(p), 6, (0,0,255), -1)
            cv2.putText(vis, labels[i], (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cropped, _ = detector.post_crop(img, corners)
        cv2.imshow(f"{fname} - corners", vis)
        cv2.imshow(f"{fname} - cropped", cropped)
    else:
        cv2.imshow(f"{fname} - corners", vis)
    cv2.waitKey(0)
cv2.destroyAllWindows()