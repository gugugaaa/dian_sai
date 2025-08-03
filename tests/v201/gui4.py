import tkinter as tk
from tkinter import ttk, font, messagebox
import cv2
import queue
import threading
import time
import serial
import re
from collections import Counter

# 导入你的所有项目模块
# 确保这些 .py 文件与 gui_app.py 在同一目录下
from system_initializer import MeasurementSystem
from single_shape_detection_algo import SingleShapeDetectionAlgorithm
from overlap_square_detection_algo import OverlapSquareDetectionAlgorithm
from digit_detection_algo import DigitDetectionAlgorithm
from result_stabilizer import ResultStabilizer
from drawing_utils import DrawingUtils

# --- 串口配置 ---
SERIAL_PORT = 'COM4'  # 根据你的USB转TTL模块实际连接的COM端口进行修改
BAUD_RATE = 115200

class SerialDataProcessor:
    """处理串口数据的类"""
    def __init__(self):
        self.max_power = 0.00
        self.voltage = 5.00  # 固定电压
        self.current = 0.00
        self.power = 0.00
        self.serial_port = None
        self.stop_reading = False
        
    def parse_serial_data(self, data_string):
        """解析 'AX.XXX' 格式的数据"""
        match = re.match(r'A(\d+\.\d+)', data_string)
        if match:
            try:
                # 解析得到的电流值
                self.current = float(match.group(1))
                # 计算功率
                self.power = self.current * self.voltage
                # 更新最大功率
                if self.power > self.max_power:
                    self.max_power = self.power
                return True
            except ValueError:
                print(f"Error: Could not convert '{match.group(1)}' to float.")
                return False
        return False
    
    
    def get_electrical_values(self):
        """获取当前电学量数值"""
        return {
            'voltage': self.voltage,
            'current': self.current,
            'power': self.power,
            'max_power': self.max_power
        }
    
    def start_serial_reading(self, update_callback):
        """启动串口读取线程"""
        def read_serial():
            try:
                self.serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                self.serial_port.flushInput()
                print(f"串口 {SERIAL_PORT} 已打开，波特率 {BAUD_RATE}")
                
                while not self.stop_reading:
                    if self.serial_port.in_waiting > 0:
                        try:
                            line = self.serial_port.readline().decode('utf-8').strip()
                            if line and self.parse_serial_data(line):
                                # 通知GUI更新
                                if update_callback:
                                    update_callback(self.get_electrical_values())
                        except UnicodeDecodeError:
                            pass  # 静默处理解码错误
                        except Exception as e:
                            print(f"Serial read error: {e}")
                    time.sleep(0.01)
                    
            except serial.SerialException as e:
                print(f"Error opening serial port {SERIAL_PORT}: {e}")
            except Exception as e:
                print(f"Serial thread error: {e}")
            finally:
                if self.serial_port and self.serial_port.is_open:
                    self.serial_port.close()
                    print(f"串口 {SERIAL_PORT} 已关闭")
        
        self.stop_reading = False
        self.serial_thread = threading.Thread(target=read_serial, daemon=True)
        self.serial_thread.start()
    
    def stop_serial_reading(self):
        """停止串口读取"""
        self.stop_reading = True
        if hasattr(self, 'serial_thread') and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2)

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

class OneSecondDisplayAggregator:
    """
    新的显示聚合器：所有模式都监听1秒钟，根据不同模式采用不同的稳定策略
    """
    def __init__(self, monitoring_duration=1.0):
        self.monitoring_duration = monitoring_duration  # 监听时长（秒）
        self.monitoring_start_time = None
        self.is_monitoring = False
        self.current_mode = None
        self.target_digit = None  # 用于数字模式
        
        # 收集的数据
        self.collected_data = {
            'distance_values': [],  # (timestamp, value)
            'size_values': [],      # (timestamp, value) for single/overlap
            'digit_data': []        # (timestamp, digit_info) for digit mode
        }
        
        # 最终稳定的结果
        self.stable_result = {
            'distance_cm': None,
            'size_cm': None
        }

    def start_monitoring(self, mode, target_digit=None):
        """开始监听1秒钟"""
        print(f"开始监听模式: {mode}, 目标数字: {target_digit}")
        self.monitoring_start_time = time.time()
        self.is_monitoring = True
        self.current_mode = mode
        self.target_digit = target_digit
        
        # 清空之前的数据
        for key in self.collected_data:
            self.collected_data[key].clear()
        
        # 重置结果
        self.stable_result = {
            'distance_cm': None,
            'size_cm': None
        }

    def add_data(self, result_dict, mode):
        """添加数据到监听缓冲区"""
        if not self.is_monitoring or mode != self.current_mode:
            return
            
        current_time = time.time()
        
        # 收集距离数据
        if result_dict.get('statistics', {}).get('distance_filtered') is not None:
            dist = result_dict['statistics']['distance_filtered']
            self.collected_data['distance_values'].append((current_time, dist))
        
        # 根据模式收集尺寸数据
        if mode == 'single':
            if result_dict.get('shape', {}).get('size_cm') is not None:
                size = result_dict['shape']['size_cm']
                self.collected_data['size_values'].append((current_time, size))
                
        elif mode == 'overlap':
            if result_dict.get('squares') and result_dict['squares'][0].get('size_cm') is not None:
                size = result_dict['squares'][0]['size_cm']
                self.collected_data['size_values'].append((current_time, size))
                
        elif mode == 'digit':
            if result_dict.get('squares'):
                for sq in result_dict['squares']:
                    if sq.get('recognition_success') and not sq.get('filtered_out'):
                        digit_info = {
                            'digit': sq.get('digit'),
                            'size_cm': sq.get('size_cm'),
                            'confidence': sq.get('confidence', 0)
                        }
                        self.collected_data['digit_data'].append((current_time, digit_info))

    def check_monitoring_complete(self):
        """检查监听是否完成，如果完成则计算稳定结果"""
        if not self.is_monitoring:
            return False
            
        elapsed = time.time() - self.monitoring_start_time
        if elapsed >= self.monitoring_duration:
            self._calculate_stable_results()
            self.is_monitoring = False
            print(f"监听完成，稳定结果: {self.stable_result}")
            return True
        return False

    def _calculate_stable_results(self):
        """根据模式计算稳定的结果"""
        # 1. 处理距离 - 所有模式都求平均值
        if self.collected_data['distance_values']:
            distances = [v for t, v in self.collected_data['distance_values']]
            self.stable_result['distance_cm'] = sum(distances) / len(distances)
        
        # 2. 根据模式处理尺寸
        if self.current_mode == 'single':
            self._process_single_mode()
        elif self.current_mode == 'overlap':
            self._process_overlap_mode()
        elif self.current_mode == 'digit':
            self._process_digit_mode()

    def _process_single_mode(self):
        """single模式：求平均值"""
        if self.collected_data['size_values']:
            sizes = [v for t, v in self.collected_data['size_values']]
            self.stable_result['size_cm'] = sum(sizes) / len(sizes)
            print(f"Single模式: 平均尺寸 {self.stable_result['size_cm']:.2f}cm")

    def _process_overlap_mode(self):
        """overlap模式：找符合6-12范围的、出现过的最小的"""
        if self.collected_data['size_values']:
            sizes = [v for t, v in self.collected_data['size_values']]
            # 筛选6-12范围内的值
            valid_sizes = [s for s in sizes if 6 <= s <= 12]
            if valid_sizes:
                self.stable_result['size_cm'] = min(valid_sizes)
                print(f"Overlap模式: 6-12范围内最小值 {self.stable_result['size_cm']:.2f}cm")
            else:
                # 如果没有6-12范围内的值，取所有值的最小值
                self.stable_result['size_cm'] = min(sizes)
                print(f"Overlap模式: 无6-12范围值，取最小值 {self.stable_result['size_cm']:.2f}cm")

    def _process_digit_mode(self):
        """digit模式：复杂的跳变处理逻辑"""
        if not self.collected_data['digit_data']:
            print("Digit模式: 无识别数据")
            return
            
        # 提取所有识别的数字和尺寸
        all_digits = []
        digit_sizes = {}  # digit -> [sizes]
        
        for timestamp, digit_info in self.collected_data['digit_data']:
            digit = digit_info['digit']
            size = digit_info['size_cm']
            all_digits.append(digit)
            
            if digit not in digit_sizes:
                digit_sizes[digit] = []
            digit_sizes[digit].append(size)
        
        if not all_digits:
            print("Digit模式: 无有效数字")
            return
            
        digit_counts = Counter(all_digits)
        unique_digits = set(all_digits)
        
        print(f"Digit模式分析: 数字频率 {dict(digit_counts)}, 目标数字 {self.target_digit}")
        
        # 情况1: 如果全程是某个数字（或者由于阈值过滤没识别到任何数字）
        if len(unique_digits) == 1:
            digit = list(unique_digits)[0]
            avg_size = sum(digit_sizes[digit]) / len(digit_sizes[digit])
            self.stable_result['size_cm'] = avg_size
            print(f"Digit模式: 全程单一数字 {digit}, 尺寸 {avg_size:.2f}cm")
            return
            
        # 情况2: 如果这1秒内，出现了5-9之间的跳变
        high_digits = [d for d in unique_digits if 5 <= d <= 9]
        low_digits = [d for d in unique_digits if d in [3, 4]]
        
        if high_digits:
            # 情况2.1: 如果出现了5-9之间的跳变夹杂着3或者4，那么就采用3或4
            if low_digits:
                preferred_digit = low_digits[0]  # 优选3或4
                avg_size = sum(digit_sizes[preferred_digit]) / len(digit_sizes[preferred_digit])
                self.stable_result['size_cm'] = avg_size
                print(f"Digit模式: 5-9跳变夹杂3/4，优选 {preferred_digit}, 尺寸 {avg_size:.2f}cm")
                return
            
            # 情况2.2: 如果这1秒内，出现了5-9之间的跳变，就用出现次数最多的
            most_common_digit = max(high_digits, key=lambda d: digit_counts[d])
            avg_size = sum(digit_sizes[most_common_digit]) / len(digit_sizes[most_common_digit])
            self.stable_result['size_cm'] = avg_size
            print(f"Digit模式: 5-9跳变，最多次数 {most_common_digit}, 尺寸 {avg_size:.2f}cm")
            return
            
        # 情况3: 如果场内只有6或者9中的一个，那么把用户输入的6或者9定位到那个正方形
        if self.target_digit in [6, 9]:
            six_nine_digits = [d for d in unique_digits if d in [6, 9]]
            if len(six_nine_digits) == 1:
                found_digit = six_nine_digits[0]
                avg_size = sum(digit_sizes[found_digit]) / len(digit_sizes[found_digit])
                self.stable_result['size_cm'] = avg_size
                print(f"Digit模式: 场内只有6/9之一({found_digit})，定位用户目标 {self.target_digit}, 尺寸 {avg_size:.2f}cm")
                return
        
        # 情况4: 最后用排除法 - 取出现频率最低的（可能是误识别）
        least_common_digit = min(unique_digits, key=lambda d: digit_counts[d])
        avg_size = sum(digit_sizes[least_common_digit]) / len(digit_sizes[least_common_digit])
        self.stable_result['size_cm'] = avg_size
        print(f"Digit模式: 排除法，最少次数 {least_common_digit}, 尺寸 {avg_size:.2f}cm")

    def get_result(self):
        """获取稳定的结果"""
        return self.stable_result.copy()

    def is_result_ready(self):
        """检查结果是否准备好"""
        return (not self.is_monitoring and 
                self.stable_result['distance_cm'] is not None and 
                self.stable_result['size_cm'] is not None)

    def reset(self):
        """重置聚合器"""
        self.is_monitoring = False
        self.monitoring_start_time = None
        self.current_mode = None
        self.target_digit = None
        
        for key in self.collected_data:
            self.collected_data[key].clear()
            
        self.stable_result = {
            'distance_cm': None,
            'size_cm': None
        }
        print("1秒监听聚合器已重置")

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
        self.target_digit = None
        self.waiting_for_measurement = False

        # 串口数据处理器
        self.serial_processor = SerialDataProcessor()

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
            self.aggregator = OneSecondDisplayAggregator()  # 使用新的聚合器
            print("系统组件初始化成功。")
        except Exception as e:
            print(f"错误：系统初始化失败: {e}")
            self.root.destroy()
            return
        
        # 创建UI
        self.create_widgets()

        # 启动串口数据读取
        self.serial_processor.start_serial_reading(self.update_electrical_values)

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

        start_button = ttk.Button(controls_frame, text="开始测量", command=self._start_measurement)
        start_button.grid(row=len(mode_buttons)+1, column=0, sticky='ew', padx=5, pady=5)

        refresh_button = ttk.Button(controls_frame, text="刷新状态", command=self._refresh_all)
        refresh_button.grid(row=len(mode_buttons)+2, column=0, sticky='ew', padx=5, pady=5)

        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(controls_frame, textvariable=self.status_var, 
                                    foreground="blue", wraplength=200)
        self.status_label.grid(row=len(mode_buttons)+3, column=0, sticky='ew', padx=5, pady=5)

        # 串口状态显示
        self.serial_status_var = tk.StringVar(value=f"串口: {SERIAL_PORT}")
        self.serial_status_label = ttk.Label(controls_frame, textvariable=self.serial_status_var, 
                                           foreground="green", wraplength=200)
        self.serial_status_label.grid(row=len(mode_buttons)+4, column=0, sticky='ew', padx=5, pady=5)

    def update_electrical_values(self, electrical_data):
        """更新电学量显示值"""
        try:
            self.param_vars['电压 (V)'].set(f"{electrical_data['voltage']:.2f}")
            self.param_vars['电流 (A)'].set(f"{electrical_data['current']:.3f}")
            self.param_vars['功率 (W)'].set(f"{electrical_data['power']:.3f}")
            self.param_vars['最大功率 (W)'].set(f"{electrical_data['max_power']:.3f}")
        except Exception as e:
            print(f"更新电学量显示时出错: {e}")

    def _on_mode_change(self):
        new_mode = self.current_mode.get()
        print(f"\n切换到模式: [{new_mode}]")
        self._reset_mode_state()

    def _start_measurement(self):
        """开始测量"""
        mode = self.current_mode.get()
        
        if mode == 'digit':
            # 数字识别模式需要先选择数字
            keypad = DigitKeypadDialog(self.root)
            selected_digit = keypad.get_digit()
            
            if selected_digit is not None:
                self.target_digit = selected_digit
                self.digit_mode_active = True
                self.aggregator.start_monitoring(mode, selected_digit)
                self.waiting_for_measurement = True
                self.status_var.set(f"正在测量数字 {selected_digit}... (1秒)")
            else:
                self.status_var.set("已取消")
                return
        else:
            # 其他模式直接开始
            self.aggregator.start_monitoring(mode)
            self.waiting_for_measurement = True
            self.status_var.set(f"正在测量 {mode} 模式... (1秒)")

    def _refresh_all(self):
        print("\n--- 正在刷新所有状态 ---")
        for mode in self.algorithms.keys():
            self.algorithms[mode].reset_filter()
            self.stabilizer.reset(mode)
        self.aggregator.reset()
        self.digit_mode_active = False
        self.target_digit = None
        self.waiting_for_measurement = False
        self.status_var.set("就绪")
        # 只重置距离和尺寸显示，保留电学量显示
        self.param_vars['距离 (cm)'].set('-')
        self.param_vars['边长/直径 (cm)'].set('-')

    def _reset_mode_state(self):
        """重置状态"""
        for mode in self.algorithms.keys():
            self.algorithms[mode].reset_filter()
            self.stabilizer.reset(mode)
        self.aggregator.reset()
        self.digit_mode_active = False
        self.target_digit = None
        self.waiting_for_measurement = False
        self.status_var.set("就绪")
        # 切换模式时只清空距离和尺寸显示
        self.param_vars['距离 (cm)'].set('-')
        self.param_vars['边长/直径 (cm)'].set('-')

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
                    # 如果正在监听，添加数据到聚合器
                    if self.waiting_for_measurement:
                        self.aggregator.add_data(stabilized_result, mode)

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
        """更新GUI状态"""
        try:
            # 检查1秒监听是否完成
            if self.waiting_for_measurement:
                if self.aggregator.check_monitoring_complete():
                    # 监听完成，获取结果
                    result = self.aggregator.get_result()
                    
                    if result['distance_cm'] is not None:
                        self.param_vars['距离 (cm)'].set(f"{result['distance_cm']:.2f}")
                    
                    if result['size_cm'] is not None:
                        self.param_vars['边长/直径 (cm)'].set(f"{result['size_cm']:.2f}")
                    
                    self.waiting_for_measurement = False
                    self.status_var.set("测量完成")
                else:
                    # 更新监听进度
                    if self.aggregator.is_monitoring:
                        elapsed = time.time() - self.aggregator.monitoring_start_time
                        remaining = max(0, self.aggregator.monitoring_duration - elapsed)
                        mode_text = "数字" if self.digit_mode_active else self.current_mode.get()
                        self.status_var.set(f"正在测量 {mode_text}... ({remaining:.1f}s)")

        except Exception as e:
            print(f"GUI更新时发生错误: {e}")
        
        # 安排下一次更新
        self.root.after(100, self._update_gui)

    def on_closing(self):
        """处理窗口关闭事件"""
        print("正在关闭应用程序...")
        self.stop_event.set()
        
        # 停止串口读取
        self.serial_processor.stop_serial_reading()
        
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