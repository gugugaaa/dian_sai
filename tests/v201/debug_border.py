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


def resize_image(img, scale_factor):
    """调整图像尺寸"""
    if img is None:
        return None
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def debug_border_detection(edges, frame):
    """
    Debug版本的边框检测，返回轮廓图和裁切预览图
    
    Args:
        edges: 边缘检测后的图像
        frame: 原始pre-crop图像
    
    Returns:
        contour_img: 轮廓可视化图像
        post_crop_img: 裁切预览图像
        corners: 检测到的角点
        success: 是否成功检测
    """
    h, w = frame.shape[:2]
    
    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建轮廓可视化图像
    contour_img = frame.copy()
    
    if not contours:
        print("❌ 没有找到轮廓")
        return contour_img, frame.copy(), None, False
    
    print(f"🔍 找到 {len(contours)} 个轮廓")
    
    # 2. 绘制所有轮廓（蓝色）
    for i, contour in enumerate(contours):
        cv2.drawContours(contour_img, [contour], -1, (255, 0, 0), 1)  # 蓝色，细线
        # 标注轮廓编号和面积
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contour)
            cv2.putText(contour_img, f"{i}:{int(area)}", (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 3. 找到最大的轮廓
    areas = [cv2.contourArea(c) for c in contours]
    largest_idx = np.argmax(areas)
    largest_contour = contours[largest_idx]
    largest_area = areas[largest_idx]
    
    print(f"📊 最大轮廓索引: {largest_idx}, 面积: {int(largest_area)}")
    
    # 4. 逼近轮廓为多边形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    print(f"🔷 轮廓逼近结果: {len(approx)} 个顶点")
    
    if len(approx) != 4:
        print(f"❌ 不是四边形，有 {len(approx)} 个顶点")
        # 仍然高亮最大轮廓
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 3)  # 绿色粗线
        return contour_img, frame.copy(), None, False
    
    # 5. 检查层次结构和内部轮廓
    has_inner_contours = False
    current_contour = largest_contour
    current_approx = approx
    selected_idx = largest_idx
    
    if hierarchy is not None:
        first_child = hierarchy[0][largest_idx][2]  # [2]是第一个子轮廓索引
        if first_child != -1:
            has_inner_contours = True
            print(f"🔗 发现内部轮廓，索引: {first_child}")
    
    # 6. A4纸外框检测逻辑
    rect = cv2.boundingRect(largest_contour)
    rect_width, rect_height = rect[2], rect[3]
    rect_ratio = rect_width / rect_height if rect_height > 0 else 0
    
    contour_area = cv2.contourArea(largest_contour)
    frame_area = w * h
    area_ratio = contour_area / frame_area
    
    print(f"📏 边界框比例: {rect_ratio:.2f}, 面积占比: {area_ratio:.2f}")
    
    # 判断是否为A4纸外框
    is_outer_frame = (1.3 <= rect_ratio <= 1.6 and area_ratio > 0.3)
    print(f"📄 是否为A4纸外框: {is_outer_frame}")
    
    if is_outer_frame and has_inner_contours:
        # 尝试使用内部轮廓
        inner_contour = contours[first_child]
        inner_epsilon = 0.02 * cv2.arcLength(inner_contour, True)
        inner_approx = cv2.approxPolyDP(inner_contour, inner_epsilon, True)
        
        if len(inner_approx) == 4:
            inner_rect = cv2.boundingRect(inner_contour)
            inner_ratio = inner_rect[2] / inner_rect[3] if inner_rect[3] > 0 else 0
            
            print(f"🔍 内轮廓比例: {inner_ratio:.2f}")
            
            if 1.3 <= inner_ratio <= 1.6:
                current_contour = inner_contour
                current_approx = inner_approx
                selected_idx = first_child
                print("✅ 使用内部轮廓作为边框")
            else:
                print("⚠️  内轮廓比例不符合A4纸，使用外轮廓")
        else:
            print(f"⚠️  内轮廓不是四边形 ({len(inner_approx)} 个顶点)，使用外轮廓")
    else:
        print("ℹ️  使用最大轮廓作为边框")
    
    # 用绿色粗线高亮选中的轮廓
    cv2.drawContours(contour_img, [current_contour], -1, (0, 255, 0), 3)  # 绿色粗线
    
    # 7. 计算凸包并排序角点
    hull = cv2.convexHull(current_approx)
    corners = hull.reshape(-1, 2).astype(np.float32)
    
    if len(corners) != 4:
        print(f"❌ 凸包不是四边形，有 {len(corners)} 个顶点")
        return contour_img, frame.copy(), None, False
    
    # 角点排序：基于x+y和x-y
    sums = corners[:, 0] + corners[:, 1]
    diffs = corners[:, 0] - corners[:, 1]
    tl_idx = np.argmin(sums)      # 左上
    br_idx = np.argmax(sums)      # 右下
    tr_idx = np.argmax(diffs)     # 右上
    bl_idx = np.argmin(diffs)     # 左下
    
    # 检查索引唯一性
    if len(set([tl_idx, tr_idx, br_idx, bl_idx])) != 4:
        print("❌ 角点索引不唯一，检测失败")
        return contour_img, frame.copy(), None, False
    
    sorted_corners = np.array([corners[tl_idx], corners[tr_idx], corners[br_idx], corners[bl_idx]])
    
    # 验证是否凸四边形
    if not cv2.isContourConvex(sorted_corners.astype(np.int32)):
        print("❌ 不是凸四边形")
        return contour_img, frame.copy(), None, False
    
    # 8. 生成裁切预览图
    # 计算目标尺寸
    width_top = np.linalg.norm(sorted_corners[1] - sorted_corners[0])
    width_bottom = np.linalg.norm(sorted_corners[2] - sorted_corners[3])
    height_left = np.linalg.norm(sorted_corners[3] - sorted_corners[0])
    height_right = np.linalg.norm(sorted_corners[2] - sorted_corners[1])
    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2
    
    # 判断方向并计算目标尺寸
    is_landscape = avg_width > avg_height
    A4_RATIO = 257 / 170
    if is_landscape:
        target_height = max(int(avg_width / A4_RATIO), 1)
        target_width = max(int(avg_width), 1)
    else:
        target_width = max(int(avg_height / A4_RATIO), 1)
        target_height = max(int(avg_height), 1)
    
    # 定义目标角点（考虑内缩）
    inset_pixels = 2
    dst_corners = np.array([
        [inset_pixels, inset_pixels],                                    # 左上
        [target_width - inset_pixels, inset_pixels],                     # 右上
        [target_width - inset_pixels, target_height - inset_pixels],     # 右下
        [inset_pixels, target_height - inset_pixels]                     # 左下
    ], dtype=np.float32)
    
    # 透视变换
    perspective_matrix = cv2.getPerspectiveTransform(sorted_corners, dst_corners)
    post_crop_img = cv2.warpPerspective(frame, perspective_matrix, (target_width, target_height))
    
    # 在post_crop_img上绘制内缩区域边框
    cv2.rectangle(post_crop_img, 
                  (inset_pixels, inset_pixels), 
                  (target_width - inset_pixels, target_height - inset_pixels), 
                  (0, 0, 255), 2)  # 红色边框
    
    return contour_img, post_crop_img, sorted_corners, True


def draw_annotations(img, corners, inset_pixels, target_width, target_height):
    """在放大后的图像上绘制角点标注和图例"""
    corner_labels = ['TL', 'TR', 'BR', 'BL']
    corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (corner, label, color) in enumerate(zip(corners, corner_labels, corner_colors)):
        cv2.circle(img, tuple(corner.astype(int)), 5, color, -1)
        cv2.putText(img, f"{label}", 
                    (int(corner[0]+8), int(corner[1]+8)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # 图例
    cv2.putText(img, "Original Border", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Post-Crop Area", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, f"Inset: {inset_pixels}px", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def test_border_detection():
    """测试边框检测并实时显示所有视图"""
    try:
        # 初始化测量系统
        system = MeasurementSystem()
        
        print("=== 边框检测 Debug 模式 ===")
        print("实时显示所有视图，按ESC退出...")
        
        while True:
            # 捕获帧
            frame = system.capture_frame()
            
            # 显示原始摄像头画面
            cv2.imshow("1. Camera Feed - Original", frame)
            
            # 预裁剪
            cropped_frame, ok = system.preprocessor.pre_crop(frame)
            if not ok:
                print("预裁剪失败，无法检测闭合轮廓")
                # 显示空白图像
                blank = np.zeros_like(frame)
                cv2.putText(blank, "Pre-crop failed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("2. Pre-cropped", blank)
                cv2.imshow("3. Edges", blank)
                cv2.imshow("4. Contours (2x)", blank)
                cv2.imshow("5. Post-crop Preview (3x)", blank)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                continue
            
            # 显示预裁剪结果
            cv2.imshow("2. Pre-cropped", cropped_frame)
            
            # 预处理（边缘检测）
            edges = system.preprocessor.preprocess(cropped_frame)
            cv2.imshow("3. Edges", edges)
            
            # Debug检测
            contour_img, post_crop_img, corners, success = debug_border_detection(edges, cropped_frame)

            # 放大轮廓图和裁切预览图
            contour_img_2x = resize_image(contour_img, 2.0)
            post_crop_img_3x = resize_image(post_crop_img, 3.0)

            # 在放大后的图像上绘制标注和图例
            if success and corners is not None:
                # 计算目标尺寸和inset_pixels（与debug_border_detection一致）
                width_top = np.linalg.norm(corners[1] - corners[0])
                width_bottom = np.linalg.norm(corners[2] - corners[3])
                height_left = np.linalg.norm(corners[3] - corners[0])
                height_right = np.linalg.norm(corners[2] - corners[1])
                avg_width = (width_top + width_bottom) / 2
                avg_height = (height_left + height_right) / 2
                is_landscape = avg_width > avg_height
                A4_RATIO = 257 / 170
                if is_landscape:
                    target_height = max(int(avg_width / A4_RATIO), 1)
                    target_width = max(int(avg_width), 1)
                else:
                    target_width = max(int(avg_height / A4_RATIO), 1)
                    target_height = max(int(avg_height), 1)
                inset_pixels = 2

                # 轮廓图绘制角点
                if contour_img_2x is not None:
                    draw_annotations(contour_img_2x, corners * 2.0, inset_pixels * 2, target_width * 2, target_height * 2)
                    cv2.imshow("4. Contours (2x)", contour_img_2x)
                # 裁切预览图绘制图例
                if post_crop_img_3x is not None:
                    draw_annotations(post_crop_img_3x, corners * 3.0, inset_pixels * 3, target_width * 3, target_height * 3)
                    cv2.imshow("5. Post-crop Preview (3x)", post_crop_img_3x)
            else:
                if contour_img_2x is not None:
                    cv2.imshow("4. Contours (2x)", contour_img_2x)
                if post_crop_img_3x is not None:
                    cv2.imshow("5. Post-crop Preview (3x)", post_crop_img_3x)
            
            if success:
                print("✅ 边框检测成功")
            else:
                print("❌ 边框检测失败")
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_border_detection()