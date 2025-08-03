import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from system_initializer import MeasurementSystem


def draw_contour_info(image, contour, text, color=(0, 255, 0)):
    """在图像上绘制轮廓信息"""
    if contour is not None:
        # 绘制轮廓
        cv2.drawContours(image, [contour], -1, color, 2)
        
        # 计算轮廓面积和矩形度
        area = cv2.contourArea(contour)
        contour_area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = contour_area / rect_area if rect_area > 0 else 0
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # 添加文本信息
        info_text = f"{text}: Area={int(area)}, Rect={rectangularity:.3f}"
        cv2.putText(image, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def debug_crop_detection(preprocessor, frame):
    """调试裁切区域检测过程"""
    print(f"\n=== Frame {preprocessor.frame_count + 1} 检测开始 ===")
    
    # 1. 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("✓ 转换为灰度图")
    
    # 2. 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("✓ 高斯模糊降噪")
    
    # 3. 直接进行边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    print("✓ Canny边缘检测")
    
    # 4. 轮廓检测
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"✓ 检测到 {len(contours)} 个轮廓")
    
    # 创建可视化图像
    vis_frame = frame.copy()
    
    # 6. 分析每个轮廓
    valid_contours = []
    min_area_threshold = 1000
    min_rectangularity = 0.7
    
    print(f"\n--- 轮廓分析 (最小面积: {min_area_threshold}, 最小矩形度: {min_rectangularity}) ---")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area < min_area_threshold:
            continue
            
        # 计算矩形度
        contour_area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = contour_area / rect_area if rect_area > 0 else 0
        
        if rectangularity < min_rectangularity:
            continue
            
        # 检查是否有子轮廓
        has_inner_contour = hierarchy[0][i][2] != -1
        
        # 综合评分
        area_score = min(area / 50000, 1.0)
        inner_bonus = 0.1 if has_inner_contour else 0
        total_score = 0.6 * area_score + 0.3 * rectangularity + inner_bonus
        
        print(f"轮廓 {i}: 面积={int(area)}, 矩形度={rectangularity:.3f}, "
              f"有内轮廓={has_inner_contour}, 总分={total_score:.3f}")
        
        valid_contours.append((contour, total_score, i))
        
        # 在可视化图像上绘制所有有效轮廓
        color = (0, 255, 255)  # 黄色表示候选轮廓
        draw_contour_info(vis_frame, contour, f"Cand{i}", color)
    
    # 7. 选择最佳轮廓
    best_contour = None
    best_score = 0
    
    if valid_contours:
        # 按分数排序
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        best_contour, best_score, best_idx = valid_contours[0]
        
        print(f"\n✓ 选择最佳轮廓: 索引{best_idx}, 分数={best_score:.3f}")
        
        # 高亮显示最佳轮廓
        draw_contour_info(vis_frame, best_contour, "Best", (0, 255, 0))
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(best_contour)
        detected_region = (x, y, w, h)
        print(f"✓ 检测区域: x={x}, y={y}, w={w}, h={h}")
        
        # 只显示轮廓检测结果
        cv2.imshow("Contours Detection", vis_frame)
        
        return detected_region, True
    else:
        print("✗ 未找到有效轮廓")
        
        # 显示轮廓检测结果（即使检测失败）
        cv2.imshow("Contours Detection", vis_frame)
        
        return None, False


def visualize_iou_calculation(frame, current_region, new_region, iou_value):
    """可视化IoU计算过程"""
    if current_region is None or new_region is None:
        return
    
    vis_frame = frame.copy()
    
    x1, y1, w1, h1 = current_region
    x2, y2, w2, h2 = new_region
    
    # 绘制当前区域（绿色）
    cv2.rectangle(vis_frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    cv2.putText(vis_frame, "Current", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 绘制新区域（蓝色）
    cv2.rectangle(vis_frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
    cv2.putText(vis_frame, "New", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 计算并绘制交集（红色）
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        cv2.rectangle(vis_frame, (inter_x1, inter_y1), (inter_x2, inter_y2), (0, 0, 255), -1, cv2.LINE_AA)
        # 半透明效果
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (inter_x1, inter_y1), (inter_x2, inter_y2), (0, 0, 255), -1)
        cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0, vis_frame)
    
    # 显示IoU值
    cv2.putText(vis_frame, f"IoU: {iou_value:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 根据IoU值显示状态
    if iou_value < 0.85:  # 更新为新的阈值
        status_text = "DRAMATIC CHANGE!"
        status_color = (0, 0, 255)  # 红色
    else:
        status_text = "Normal Change"
        status_color = (0, 255, 0)  # 绿色
    
    cv2.putText(vis_frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    cv2.imshow("IoU Calculation", vis_frame)


def enhanced_region_processing(preprocessor, new_region, detection_success):
    """简化的区域处理逻辑 - 修复跳动问题"""
    
    # 初始化新增属性（如果不存在）
    if not hasattr(preprocessor, 'fast_update_mode'):
        preprocessor.fast_update_mode = False
    if not hasattr(preprocessor, 'fast_update_remaining'):
        preprocessor.fast_update_remaining = 0
    
    print(f"\n--- 简化区域处理 ---")
    print(f"快速更新模式: {preprocessor.fast_update_mode} (剩余{preprocessor.fast_update_remaining}帧)")
    
    if not detection_success:
        print("✗ 检测失败，跳过处理")
        return False
    
    # 更新快速更新模式计数器
    if preprocessor.fast_update_mode:
        preprocessor.fast_update_remaining -= 1
        if preprocessor.fast_update_remaining <= 0:
            preprocessor.fast_update_mode = False
            print("📍 快速更新模式结束")
    
    # 检测大幅变化（只在非快速模式下检测）
    if not preprocessor.fast_update_mode and preprocessor.current_crop_region:
        is_dramatic, iou_value = is_dramatic_change(preprocessor.current_crop_region, new_region)
        print(f"大幅变化检测: IoU={iou_value:.3f}, 变化={'是' if is_dramatic else '否'}")
        
        if is_dramatic:
            print("🚀 IoU过低，检测到大幅变化，启动快速更新模式")
            preprocessor.fast_update_mode = True
            preprocessor.fast_update_remaining = 3  # 持续3帧
    
    # 区域更新逻辑
    if preprocessor.fast_update_mode:
        # 快速模式：直接使用新区域（best contour）
        preprocessor.current_crop_region = new_region
        preprocessor.is_region_valid = True
        print(f"⚡ 快速模式直接更新: {new_region}")
        
        # 清空历史记录，避免平均化影响
        preprocessor.region_history.clear()
        preprocessor.region_history.append(new_region)
        
    else:
        # 正常模式：直接使用新区域，无合理性检查
        # 删除了原来的合理性检查逻辑
        if len(preprocessor.region_history) >= 3:  # 减少历史记录数量，提高响应性
            preprocessor.region_history.popleft()
        preprocessor.region_history.append(new_region)
        
        # 简化的平均化：只使用最近几个区域
        avg_region = preprocessor._calculate_average_region()
        preprocessor.current_crop_region = avg_region
        preprocessor.is_region_valid = True
        print(f"✅ 正常模式更新: {avg_region}")
    
    return True


def calculate_iou(region1, region2):
    """计算两个区域的IoU (Intersection over Union)"""
    if region1 is None or region2 is None:
        return 0.0
    
    x1, y1, w1, h1 = region1
    x2, y2, w2, h2 = region2
    
    # 计算交集
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # 计算并集
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def is_dramatic_change(current_region, new_region, iou_threshold=0.7):
    """检测是否为大幅变化 - 降低IoU阈值，减少过度敏感"""
    iou = calculate_iou(current_region, new_region)
    is_dramatic = iou < iou_threshold  # 从0.9降低到0.7
    return is_dramatic, iou


def display_region_stats(preprocessor):
    """显示区域统计信息"""
    stats = preprocessor.get_region_stats()
    print(f"\n=== 区域统计信息 ===")
    print(f"当前区域: {stats['current_region']}")
    print(f"历史记录数量: {stats['history_count']}")
    print(f"区域有效性: {stats['is_valid']}")
    print(f"帧计数: {stats['frame_count']}")
    
    # 显示简化后的状态信息
    print(f"快速更新模式: {getattr(preprocessor, 'fast_update_mode', False)}")
    print(f"快速更新剩余帧数: {getattr(preprocessor, 'fast_update_remaining', 0)}")
    
    # 显示当前区域和最新检测的IoU
    if hasattr(preprocessor, 'last_detection_region') and preprocessor.current_crop_region:
        iou = calculate_iou(preprocessor.current_crop_region, preprocessor.last_detection_region)
        print(f"当前区域与最新检测IoU: {iou:.3f}")
    
    if len(preprocessor.region_history) > 0:
        print(f"历史区域:")
        for i, region in enumerate(preprocessor.region_history):
            print(f"  {i+1}: {region}")


# 修改主函数中的重置逻辑和状态显示
def main():
    # 初始化系统
    system = MeasurementSystem("calib.yaml", 500)
    
    print("=== 修复版Pre-Crop 调试模式 ===")
    print("修复内容：")
    print("  ✅ 删除合理性检查，避免跳动")
    print("  ✅ 快速模式直接使用best contour")
    print("  ✅ 降低IoU阈值减少过度敏感")
    print("\n可视化窗口:")
    print("  📊 Contours Detection - 轮廓检测结果")
    print("  📐 Crop Region - 当前裁剪区域")
    print("  🔢 IoU Calculation - 交并集计算可视化")
    print("\n按 'q' 退出程序")
    
    while True:
        frame = system.capture_frame()
        if frame is None:
            break
        
        # 检查退出键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        print(f"\n{'='*60}")
        print(f"处理帧 {system.preprocessor.frame_count + 1}")
        
        # 检查是否需要更新区域
        should_update = system.preprocessor._should_update_region()
        print(f"是否需要更新区域: {should_update}")
        
        if should_update:
            print("\n--- 开始区域检测 ---")
            new_region, detection_success = debug_crop_detection(system.preprocessor, frame)
            
            # 保存最新检测结果用于IoU计算
            if detection_success:
                system.preprocessor.last_detection_region = new_region
                
                # 可视化IoU计算
                if system.preprocessor.current_crop_region:
                    iou_value = calculate_iou(system.preprocessor.current_crop_region, new_region)
                    visualize_iou_calculation(frame, system.preprocessor.current_crop_region, new_region, iou_value)
            
            # 使用修复后的区域处理逻辑
            update_success = enhanced_region_processing(system.preprocessor, new_region, detection_success)
            
            if not update_success:
                print("❌ 区域更新被拒绝")
        
        # 更新帧计数
        system.preprocessor.frame_count += 1
        
        # 执行预裁剪
        print("\n--- 执行预裁剪 ---")
        cropped, crop_success = system.preprocessor.pre_crop(frame)
        
        if crop_success:
            print(f"✓ 预裁剪成功, 裁剪后尺寸: {cropped.shape[:2]}")
            
            # 在原图上绘制当前使用的裁切区域
            if system.preprocessor.current_crop_region:
                x, y, w, h = system.preprocessor.current_crop_region
                frame_with_region = frame.copy()
                
                # 根据模式选择颜色
                if getattr(system.preprocessor, 'fast_update_mode', False):
                    color = (0, 165, 255)  # 橙色 - 快速更新模式
                    mode_text = "FAST UPDATE"
                else:
                    color = (0, 255, 0)    # 绿色 - 正常模式
                    mode_text = "NORMAL"
                
                cv2.rectangle(frame_with_region, (x-10, y-10), (x+w+10, y+h+10), color, 3)
                cv2.putText(frame_with_region, f"Crop Region ({mode_text})", (x, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 显示状态信息
                status_y = 30
                if getattr(system.preprocessor, 'fast_update_mode', False):
                    remaining = getattr(system.preprocessor, 'fast_update_remaining', 0)
                    cv2.putText(frame_with_region, f"Fast Mode: {remaining} frames left", 
                               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                cv2.imshow("Crop Region", frame_with_region)
        else:
            print("✗ 预裁剪失败")
        
        # 显示统计信息
        display_region_stats(system.preprocessor)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()