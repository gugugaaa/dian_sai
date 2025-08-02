import tkinter as tk
from tkinter import ttk, font, messagebox
import cv2
import queue
import threading
import time
from collections import Counter

# 导入你的所有项目模块
# 确保这些 .py 文件与 gui_app.py 在同一目录下
from system_initializer import MeasurementSystem
from single_shape_detection_algo import SingleShapeDetectionAlgorithm
from overlap_square_detection_algo import OverlapSquareDetectionAlgorithm
from digit_detection_algo import DigitDetectionAlgorithm
from result_stabilizer import ResultStabilizer
from drawing_utils import DrawingUtils

class DigitKeypadDialog:
    """数字键盘对话框"""
    def __init__(self, parent):
        self.parent = parent
        self.selected_digit = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("选择要测量的数字")
        self.dialog.geometry("300x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 使对话框居中
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self.create_keypad()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def create_keypad(self):
        # 提示标签
        label = ttk.Label(self.dialog, text="请选择要测量的数字:", font=("Arial", 14))
        label.pack(pady=10)
        
        # 键盘框架
        keypad_frame = ttk.Frame(self.dialog)
        keypad_frame.pack(pady=10)
        
        # 创建数字按钮 0-9
        buttons = []
        # 1-9 按钮
        for i in range(9):
            row = i // 3
            col = i % 3
            btn = ttk.Button(keypad_frame, text=str(i+1), 
                           command=lambda x=i+1: self.on_digit_click(x),
                           width=8)
            btn.grid(row=row, column=col, padx=5, pady=5, ipadx=10, ipady=10)
            buttons.append(btn)
        
        # 0 按钮在底部中间
        btn_0 = ttk.Button(keypad_frame, text="0", 
                         command=lambda: self.on_digit_click(0),
                         width=8)
        btn_0.grid(row=3, column=1, padx=5, pady=5, ipadx=10, ipady=10)
        
        # 取消按钮
        cancel_btn = ttk.Button(self.dialog, text="取消", command=self.on_cancel)
        cancel_btn.pack(pady=20)
        
    def on_digit_click(self, digit):
        self.selected_digit = digit
        self.dialog.destroy()
        
    def on_cancel(self):
        self.selected_digit = None
        self.dialog.destroy()
        
    def get_digit(self):
        self.dialog.wait_window()
        return self.selected_digit

class DigitMatcher:
    """处理数字匹配和排除法逻辑"""
    def __init__(self, timeout_seconds=10):
        self.timeout_seconds = timeout_seconds
        self.recognized_digits_history = []  # 存储历史识别结果
        self.target_digit = None
        self.start_time = None
        
    def set_target_digit(self, digit):
        """设置目标数字并重置计时器"""
        self.target_digit = digit
        self.start_time = time.time()
        self.recognized_digits_history.clear()
        
    def add_recognition_result(self, result):
        """添加新的识别结果"""
        if result.get('squares'):
            recognized_digits = []
            for sq in result['squares']:
                if sq.get('recognition_success') and not sq.get('filtered_out'):
                    recognized_digits.append({
                        'digit': sq.get('digit'),
                        'size_cm': sq.get('size_cm'),
                        'confidence': sq.get('confidence', 0)
                    })
            if recognized_digits:
                self.recognized_digits_history.append({
                    'timestamp': time.time(),
                    'digits': recognized_digits
                })
    
    def find_match(self):
        """尝试找到匹配的数字"""
        if self.target_digit is None:
            return None, "no_target"
            
        # 检查是否超时
        if time.time() - self.start_time > self.timeout_seconds:
            return self._apply_exclusion_method(), "exclusion"
        
        # 首先尝试直接匹配
        for record in reversed(self.recognized_digits_history[-5:]):  # 只看最近5次
            for digit_info in record['digits']:
                if digit_info['digit'] == self.target_digit:
                    return digit_info['size_cm'], "direct"
        
        return None, "waiting"
    
    def _apply_exclusion_method(self):
        """应用排除法"""
        if not self.recognized_digits_history:
            return None
            
        # 获取最新的识别结果
        latest = self.recognized_digits_history[-1]['digits']
        
        # 统计所有历史识别中出现的数字
        all_recognized = set()
        for record in self.recognized_digits_history:
            for digit_info in record['digits']:
                all_recognized.add(digit_info['digit'])
        
        # 找出可能是误识别的数字（不应该出现的）
        # 假设真实数字集合是连续的或有规律的
        potential_misrecognitions = []
        
        for digit_info in latest:
            # 如果这个数字很少出现，可能是误识别
            occurrences = sum(1 for record in self.recognized_digits_history 
                            for d in record['digits'] if d['digit'] == digit_info['digit'])
            if occurrences < len(self.recognized_digits_history) * 0.3:  # 出现率低于30%
                potential_misrecognitions.append(digit_info)
        
        # 如果有潜在的误识别，返回其中一个的尺寸
        if potential_misrecognitions:
            # 选择置信度最低的作为目标
            return min(potential_misrecognitions, key=lambda x: x.get('confidence', 1))['size_cm']
        
        # 否则返回任意一个
        if latest:
            return latest[0]['size_cm']
            
        return None

class DisplayAggregator:
    """
    负责在UI上显示数值前的最后一道稳定关卡。
    它在设定的时间窗口内收集数据，并仅在某个值足够"稳定"（出现频率够高）时才提供它。
    """
    def __init__(self, window_seconds=1, stable_threshold_percent=50):
        self.window_seconds = window_seconds
        self.stable_threshold = stable_threshold_percent / 100.0
        self.history = {
            'size_cm': [],
            'distance_cm': []
        }
        # 锁定状态：一旦显示了稳定值，就锁定不再更新
        self.locked_values = {
            'size_cm': None,
            'distance_cm': None
        }
        self.is_locked = False

    def add_result(self, result_dict, mode):
        """根据传入的算法结果，提取关键信息并记录下来"""
        # 如果已经锁定，就不再接受新的结果
        if self.is_locked:
            return
            
        current_time = time.time()

        # 记录距离
        if result_dict.get('statistics', {}).get('distance_filtered') is not None:
            dist = result_dict['statistics']['distance_filtered']
            self.history['distance_cm'].append((current_time, dist))

        # 根据模式记录尺寸
        if mode == 'single' and result_dict.get('shape', {}).get('size_cm') is not None:
            size = result_dict['shape']['size_cm']
            self.history['size_cm'].append((current_time, size))
        
        elif mode == 'overlap' and result_dict.get('squares') and result_dict['squares'][0].get('size_cm') is not None:
            size = result_dict['squares'][0]['size_cm']
            self.history['size_cm'].append((current_time, size))

        # 数字模式现在由专门的逻辑处理，不在这里处理

    def add_size_value(self, size_cm):
        """直接添加尺寸值（用于数字识别模式）"""
        if self.is_locked:
            return
        current_time = time.time()
        self.history['size_cm'].append((current_time, size_cm))

    def get_stable_values(self):
        """分析历史记录，返回稳定的值"""
        # 如果已经锁定，直接返回锁定的值
        if self.is_locked:
            return self.locked_values.copy()
            
        current_time = time.time()
        stable_values = {}

        # 1. 清理过期数据
        for key in self.history:
            self.history[key] = [(t, v) for t, v in self.history[key] if current_time - t < self.window_seconds]

        # 2. 计算稳定值
        for key, records in self.history.items():
            if not records:
                stable_values[key] = None
                continue

            values = [v for t, v in records]
            
            # 对连续值（如尺寸和距离）进行处理
            if key in ['size_cm', 'distance_cm']:
                # 策略：使用滑动窗口内的平均值作为稳定值
                # 只有当窗口内有足够样本时才更新
                if len(values) > 3: # 需要至少3个样本
                    stable_values[key] = sum(values) / len(values)
                else:
                    stable_values[key] = None
        
        # 检查是否应该锁定值
        if (stable_values.get('size_cm') is not None and 
            stable_values.get('distance_cm') is not None):
            # 都有稳定值了，锁定这些值
            self.locked_values = stable_values.copy()
            self.is_locked = True
            print(f"数值已锁定: 距离={self.locked_values['distance_cm']:.2f}cm, 尺寸={self.locked_values['size_cm']:.2f}cm")
        
        return stable_values

    def reset(self):
        """清空所有历史记录并解锁"""
        for key in self.history:
            self.history[key].clear()
        self.locked_values = {
            'size_cm': None,
            'distance_cm': None
        }
        self.is_locked = False
        print("显示聚合器已重置并解锁")

class MeasurementApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("综合测量系统控制面板")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 初始化状态和控制变量
        self.current_mode = tk.StringVar(value='single')
        self.stop_event = threading.Event()
        self.results_queue = queue.Queue()
        
        # 数字识别模式相关
        self.digit_mode_active = False
        self.digit_matcher = DigitMatcher()
        self.waiting_for_digit = False

        # 初始化后端组件
        try:
            print("正在初始化测量系统...")
            self.system = MeasurementSystem("calib.yaml", 500)
            self.algorithms = {
                'single': SingleShapeDetectionAlgorithm(self.system, filter_window_size=10),
                'overlap': OverlapSquareDetectionAlgorithm(self.system, filter_window_size=10),
                'digit': DigitDetectionAlgorithm(self.system, model_path='models/improved_mnist_model.pth')
            }
            self.stabilizer = ResultStabilizer()
            self.drawer = DrawingUtils()
            self.aggregator = DisplayAggregator()
            print("系统组件初始化成功。")
        except Exception as e:
            print(f"错误：系统初始化失败: {e}")
            self.root.destroy()
            return
        
        # 创建UI
        self.create_widgets()

        # 启动后台处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        # 启动UI更新循环
        self._update_gui()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- 左侧参数显示面板 ---
        params_frame = ttk.LabelFrame(main_frame, text="测量结果", padding="10")
        params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.param_vars = {
            "距离 (cm)": tk.StringVar(value='-'),
            "边长/直径 (cm)": tk.StringVar(value='-'),
            "电压 (V)": tk.StringVar(value='-'),
            "电流 (A)": tk.StringVar(value='-'),
            "功率 (W)": tk.StringVar(value='-'),
            "最大功率 (W)": tk.StringVar(value='-'),
        }

        header_font = font.Font(family="Helvetica", size=12, weight="bold")
        value_font = font.Font(family="Courier", size=14)

        for i, (name, var) in enumerate(self.param_vars.items()):
            ttk.Label(params_frame, text=name, font=header_font).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Label(params_frame, textvariable=var, font=value_font, foreground="cyan", background="black", padding=5).grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)

        # --- 右侧控制面板 ---
        controls_frame = ttk.LabelFrame(main_frame, text="控制选项", padding="10")
        controls_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        mode_buttons = [
            ("单个形状", "single"),
            ("重叠正方形", "overlap"),
            ("数字识别", "digit")
        ]
        
        for i, (text, mode) in enumerate(mode_buttons):
            rb = ttk.Radiobutton(controls_frame, text=text, variable=self.current_mode, value=mode, command=self._on_mode_change)
            rb.grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Separator(controls_frame, orient='horizontal').grid(row=len(mode_buttons), column=0, sticky='ew', pady=10)

        refresh_button = ttk.Button(controls_frame, text="刷新状态", command=self._refresh_all)
        refresh_button.grid(row=len(mode_buttons)+1, column=0, sticky='ew', padx=5, pady=5)

        # 数字识别模式状态显示
        self.digit_status_var = tk.StringVar(value="")
        self.digit_status_label = ttk.Label(controls_frame, textvariable=self.digit_status_var, 
                                          foreground="blue", wraplength=200)
        self.digit_status_label.grid(row=len(mode_buttons)+2, column=0, sticky='ew', padx=5, pady=5)

    def _on_mode_change(self):
        new_mode = self.current_mode.get()
        print(f"\n切换到模式: [{new_mode}]")
        self._reset_mode_state(new_mode)
        
        # 如果切换到数字识别模式，显示键盘
        if new_mode == 'digit':
            self.root.after(100, self._show_digit_keypad)
            self.digit_mode_active = True
        else:
            self.digit_mode_active = False
            self.waiting_for_digit = False
            self.digit_status_var.set("")

    def _show_digit_keypad(self):
        """显示数字键盘并获取用户选择"""
        keypad = DigitKeypadDialog(self.root)
        selected_digit = keypad.get_digit()
        
        if selected_digit is not None:
            print(f"用户选择了数字: {selected_digit}")
            self.digit_matcher.set_target_digit(selected_digit)
            self.waiting_for_digit = True
            self.digit_status_var.set(f"正在寻找数字 {selected_digit}...")
        else:
            # 用户取消了，切换回单个形状模式
            self.current_mode.set('single')
            self.digit_mode_active = False
            self.waiting_for_digit = False

    def _refresh_all(self):
        print("\n--- 正在刷新所有状态 ---")
        for mode in self.algorithms.keys():
            self.algorithms[mode].reset_filter()
            self.stabilizer.reset(mode)
        self.aggregator.reset()
        self.digit_matcher = DigitMatcher()
        self.waiting_for_digit = False
        self.digit_status_var.set("")
        for var in self.param_vars.values():
            var.set('-')

    def _reset_mode_state(self, mode_name):
        """重置特定模式的算法和稳定器，并清空聚合器"""
        if mode_name in self.algorithms:
            self.algorithms[mode_name].reset_filter()
        self.stabilizer.reset(mode_name)
        self.aggregator.reset()
        # 切换模式时也清空显示
        for var in self.param_vars.values():
            var.set('-')

    def _processing_loop(self):
        """后台处理循环，负责CV任务"""
        while not self.stop_event.is_set():
            try:
                frame = self.system.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # 为了方便调试，显示原始画面
                cv2.imshow("Original Camera Feed", frame)

                # --- 核心预处理步骤 ---
                cropped_frame, ok = self.system.preprocessor.pre_crop(frame)
                if not ok: continue
                edges = self.system.preprocessor.preprocess(cropped_frame)
                ok, corners = self.system.border_detector.detect_border(edges, cropped_frame)
                if not ok: continue
                post_cropped_frame, adjusted_corners = self.system.border_detector.post_crop(cropped_frame, corners, inset_pixels=2)
                D_raw, _ = self.system.distance_calculator.calculate_D(corners, self.system.K)
                if D_raw is None: continue
                D_corrected = D_raw 

                # --- 算法处理 ---
                
                mode = self.current_mode.get()
                active_algorithm = self.algorithms[mode]
                raw_result = active_algorithm.process(post_cropped_frame, adjusted_corners, D_corrected, self.system.K)
                
                # --- 结果稳定与可视化 ---
                stabilized_result = self.stabilizer.process(raw_result, mode)
                
                if stabilized_result.get('success', False):
                    # 数字模式特殊处理
                    if mode == 'digit' and self.waiting_for_digit:
                        self.digit_matcher.add_recognition_result(stabilized_result)
                    else:
                        self.results_queue.put((stabilized_result, mode))

                display_frame = self.drawer.draw(post_cropped_frame.copy(), stabilized_result, mode)
                cv2.putText(display_frame, f"Mode: {mode.upper()}", (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Result", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()

            except Exception as e:
                print(f"处理循环中发生错误: {e}")
                time.sleep(1) # 避免错误刷屏

        self.system.cap.release()
        cv2.destroyAllWindows()
        print("处理线程已停止。")

    
    def _update_gui(self):
        """从队列中获取结果并更新UI"""
        try:
            # 处理数字识别模式
            if self.digit_mode_active and self.waiting_for_digit:
                size_cm, match_type = self.digit_matcher.find_match()
                
                if size_cm is not None:
                    # 找到匹配
                    if match_type == "direct":
                        self.digit_status_var.set(f"找到数字 {self.digit_matcher.target_digit}!")
                    elif match_type == "exclusion":
                        self.digit_status_var.set(f"使用排除法确定数字 {self.digit_matcher.target_digit}")
                    elif match_type == "six_nine_special":
                        # 新增：6/9特殊匹配的状态显示
                        self.digit_status_var.set(f"检测到唯一的6/9，自动匹配数字 {self.digit_matcher.target_digit}!")
                    
                    # 既然已经找到了确定的值，就应该直接显示，而不是再通过聚合器
                    print(f"匹配成功: 类型={match_type}, 尺寸={size_cm:.2f}cm。直接更新UI。")
                    self.param_vars['边长/直径 (cm)'].set(f"{size_cm:.2f}")

                    # 为了防止后续数据覆盖，可以手动锁定聚合器的值
                    self.aggregator.locked_values['size_cm'] = size_cm
                    # 只有当距离值也稳定后才完全锁定
                    # self.aggregator.is_locked = True 

                    self.waiting_for_digit = False # 停止寻找
                                            
                elif match_type == "waiting":
                    # 更新等待状态
                    elapsed = time.time() - self.digit_matcher.start_time
                    self.digit_status_var.set(f"正在寻找数字 {self.digit_matcher.target_digit}... ({elapsed:.1f}s)")
            
            # 处理其他模式 或 数字模式锁定后的距离更新
            if not self.aggregator.is_locked:
                # (这里的代码保持不变)
                while not self.results_queue.empty():
                    result, mode = self.results_queue.get_nowait()
                    self.aggregator.add_result(result, mode)

                stable_values = self.aggregator.get_stable_values()
                
                # 更新距离
                dist = stable_values.get('distance_cm')
                if dist is not None:
                    self.param_vars['距离 (cm)'].set(f"{dist:.2f}")
                
                # 更新尺寸 (在非数字模式下)
                size = stable_values.get('size_cm')
                if size is not None and not self.digit_mode_active:
                    self.param_vars['边长/直径 (cm)'].set(f"{size:.2f}")
            else:
                # 如果已锁定，清空队列但不处理结果
                while not self.results_queue.empty():
                    self.results_queue.get_nowait()

        except Exception as e:
            print(f"GUI更新时发生错误: {e}")
        
        # 安排下一次更新
        self.root.after(100, self._update_gui)

    def on_closing(self):
        """处理窗口关闭事件"""
        print("正在关闭应用程序...")
        self.stop_event.set()
        # 等待后台线程安全退出
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # 可以设置一个主题
    # style = ttk.Style(root)
    # style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'
    app = MeasurementApp(root)
    root.mainloop()