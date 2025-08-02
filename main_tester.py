# main_tester.py
import sys
import os
# 如果你的项目结构需要，请保留这个路径添加
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np

# 导入系统和算法模块
from system_initializer import MeasurementSystem
from single_shape_detection_algo import SingleShapeDetectionAlgorithm
from overlap_square_detection_algo import OverlapSquareDetectionAlgorithm
from digit_detection_algo import DigitDetectionAlgorithm
from result_stabilizer import ResultStabilizer
from drawing_utils import DrawingUtils

def main():
    """主函数，负责初始化和运行测试循环"""
    print("初始化测量系统...")
    try:
        # 确保 calib.yaml 在正确的路径
        system = MeasurementSystem("calib.yaml", 500)
    except Exception as e:
        print(f"错误：系统初始化失败: {e}")
        return

    # 初始化所有检测算法
    try:
        algorithms = {
            'single': SingleShapeDetectionAlgorithm(system, filter_window_size=10),
            'overlap': OverlapSquareDetectionAlgorithm(system, filter_window_size=10),
            'digit': DigitDetectionAlgorithm(system, model_path='models/improved_mnist_model.pth')
        }
        print("所有检测算法初始化成功。")
    except Exception as e:
        print(f"错误：算法初始化失败: {e}")
        return

    # 初始化结果稳定器和可视化工具
    stabilizer = ResultStabilizer()
    drawer = DrawingUtils()

    # 设置初始模式
    current_mode = 'single'
    
    print("\n--- 控制台 ---")
    print("按 '1' 切换到 [单个形状] 检测模式")
    print("按 '2' 切换到 [重叠正方形] 检测模式")
    print("按 '3' 切换到 [数字识别] 检测模式")
    print("按 'r' 重置当前模式的滤波器和稳定器状态")
    print("按 'q' 退出")
    print("---------------\n")

    while True:
        try:
            # 1. 捕获和通用预处理流程
            frame = system.capture_frame()
            if frame is None:
                print("无法捕获帧，退出...")
                break

            # 为了方便调试，显示原始画面
            cv2.imshow("Original Camera Feed", frame)
            
            # --- 与 test_all_2.py 相同的预处理步骤 ---
            cropped_frame, ok = system.preprocessor.pre_crop(frame)
            if not ok:
                cv2.imshow("Result", frame) # 显示原始帧并提示
                cv2.putText(frame, "Pre-crop failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if handle_key_press(cv2.waitKey(1), current_mode, algorithms, stabilizer) == 'exit': break
                continue

            edges = system.preprocessor.preprocess(cropped_frame)
            ok, corners = system.border_detector.detect_border(edges, cropped_frame)
            if not ok:
                cv2.imshow("Result", cropped_frame)
                cv2.putText(cropped_frame, "A4 Border not found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if handle_key_press(cv2.waitKey(1), current_mode, algorithms, stabilizer) == 'exit': break
                continue
            
            post_cropped_frame, adjusted_corners = system.border_detector.post_crop(cropped_frame, corners, inset_pixels=5)
            D_raw, _ = system.distance_calculator.calculate_D(corners, system.K)
            if D_raw is None:
                cv2.imshow("Result", post_cropped_frame)
                cv2.putText(post_cropped_frame, "PnP failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if handle_key_press(cv2.waitKey(1), current_mode, algorithms, stabilizer) == 'exit': break
                continue
            
            D_corrected = D_raw # 暂不使用距离校正

            # 2. 模式选择与算法处理
            active_algorithm = algorithms[current_mode]
            raw_result = active_algorithm.process(post_cropped_frame, adjusted_corners, D_corrected, system.K)

            # 3. 结果稳定化
            stabilized_result = stabilizer.process(raw_result, current_mode)

            # 4. 可视化
            display_frame = post_cropped_frame.copy()
            display_frame = drawer.draw(display_frame, stabilized_result, current_mode)
            
            # 在画面左上角显示当前模式
            cv2.putText(display_frame, f"Mode: {current_mode.upper()}", (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Result", display_frame)

            # 5. 键盘事件处理
            new_mode = handle_key_press(cv2.waitKey(1), current_mode, algorithms, stabilizer)
            if new_mode == 'exit':
                break
            elif new_mode:
                current_mode = new_mode

        except KeyboardInterrupt:
            print("用户中断。")
            break
        except Exception as e:
            print(f"主循环发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 清理
    cv2.destroyAllWindows()
    system.cap.release()
    print("测试结束。")

def handle_key_press(key, current_mode, algorithms, stabilizer):
    """处理键盘输入，返回新的模式名或'exit'"""
    key &= 0xFF
    if key == ord('q'):
        return 'exit'
    elif key == ord('1'):
        if current_mode != 'single':
            print("\n切换到 [单个形状] 检测模式")
            return 'single'
    elif key == ord('2'):
        if current_mode != 'digit':
            print("\n切换到 [数字识别] 检测模式")
            return 'digit'
    elif key == ord('3'):
        if current_mode != 'overlap':
            print("\n切换到 [重叠正方形] 检测模式")
            return 'overlap'
    elif key == ord('r'):
        print(f"\n重置 [{current_mode}] 模式的状态...")
        algorithms[current_mode].reset_filter()
        stabilizer.reset(current_mode)
    
    return None # 没有模式切换

if __name__ == "__main__":
    main()