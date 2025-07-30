import cv2
import numpy as np
import time
import yaml
import tkinter as tk
from tkinter import messagebox

from preprocessor import Preprocessor
from border_detector import BorderDetector
from shape_detector import ShapeDetector
from distance_calculator import DistanceCalculator

class SystemInitializer:
    def __init__(self, calib_file="calib.yaml", crop_width=600):
        # 加载标定参数
        self.K, self.dist = self.load_calibration(calib_file)
        # 初始化硬件
        self.cap, self.root, self.button_var = self.init_hardware(crop_width)
        # 初始化预处理器
        self.preprocessor = Preprocessor(self.K, self.dist)

    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix'], dtype=np.float32).reshape(3, 3)
        dist = np.array(data['dist_coeff'], dtype=np.float32)
        return K, dist

    def init_hardware(self, crop_width=600):
        cap = cv2.VideoCapture(1)  # 连接usb摄像头
        if not cap.isOpened():
            raise ValueError("无法打开摄像头")
        # 显式设置分辨率为1920x1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # A4纸竖放的长宽比 (宽:高 = 1:√2)
        A4_RATIO = 1.414  # √2
        
        # 使用传入的裁切宽度（像素）
        self.crop_width = crop_width
        
        # 根据A4比例计算高度
        self.crop_height = int(self.crop_width * A4_RATIO)
        
        # 确保裁切区域不超出原始画面
        if self.crop_height > 1080:
            self.crop_height = 1080
            self.crop_width = int(self.crop_height / A4_RATIO)
        
        # 计算居中裁切的起始位置
        self.crop_x = (1920 - self.crop_width) // 2
        self.crop_y = (1080 - self.crop_height) // 2
        
        # 使用tkinter模拟按钮（主线程）
        root = tk.Tk()
        root.title("测量模拟器")
        root.geometry("300x100")
        
        measure_var = tk.BooleanVar(value=False)
        
        def on_measure():
            measure_var.set(True)
        
        button = tk.Button(root, text="测量", command=on_measure)
        button.pack(pady=20)
        
        return cap, root, measure_var

class MeasurementSystem:
    def __init__(self, calib_file="calib.yaml", crop_width=600):
        # 使用 SystemInitializer 进行初始化
        self.initializer = SystemInitializer(calib_file, crop_width)
        self.K = self.initializer.K
        self.dist = self.initializer.dist
        self.cap = self.initializer.cap
        self.root = self.initializer.root
        self.button_var = self.initializer.button_var
        
        # 获取裁切参数
        self.crop_x = self.initializer.crop_x
        self.crop_y = self.initializer.crop_y
        self.crop_width = self.initializer.crop_width
        self.crop_height = self.initializer.crop_height
        
        # 初始化预处理器
        self.preprocessor = Preprocessor(self.K, self.dist)
        # 初始化边框检测器
        self.border_detector = BorderDetector()
        # 初始化形状检测器
        self.shape_detector = ShapeDetector()
        # 初始化距离计算器
        self.distance_calculator = DistanceCalculator(self.K)

    def button_pressed(self):
        if self.button_var.get():
            self.button_var.set(False)  # 重置以备下次按压
            return True
        return False

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("捕获帧失败")
        
        # 居中裁切
        cropped_frame = frame[self.crop_y:self.crop_y + self.crop_height, 
                            self.crop_x:self.crop_x + self.crop_width]
        
        return cropped_frame

    def run(self):
        def process_measurement():
            if self.button_pressed():
                t0 = time.time()
                frame = self.capture_frame()
                
                # 预裁剪
                cropped_frame, ok = self.preprocessor.pre_crop(frame)
                if not ok:
                    self.log_error("预裁剪失败，无法检测闭合轮廓")
                    return
                
                # 预处理（去畸变和边缘检测，使用裁剪后的帧）
                edges = self.preprocessor.preprocess(cropped_frame)
                
                # 检测A4纸边框并获取角点
                # corners顺序为: 左上, 右上, 右下, 左下
                ok, corners = self.border_detector.detect_border(edges, cropped_frame)
                if not ok:
                    self.log_error("无法检测A4边框")
                    return
                
                # 基于A4边框进行后裁切，避免边框干扰形状检测
                post_cropped_frame, adjusted_corners = self.border_detector.post_crop(cropped_frame, corners, inset_pixels=5)
                
                # 使用PnP计算距离D（使用原始角点）
                D = self.distance_calculator.calculate_D(corners)
                if D is None:
                    self.log_error("PnP求解失败")
                    return
                
                # 检测形状和像素尺寸（使用后裁切的图像）
                shape, x_pix = self.shape_detector.detect_shape(post_cropped_frame)
                if not shape:
                    self.log_error("无法检测形状")
                    return
                
                # 计算实际尺寸x，传入调整后的A4纸角点信息
                x = self.shape_detector.calculate_X(x_pix, D, self.K, adjusted_corners)
                
                self.show_result(D, x)
                self.save_log(D, x, t0)
        
        def update():
            try:
                process_measurement()
                # 安排下次更新
                self.root.after(100, update)  # 100ms后再次检查
            except Exception as e:
                self.log_error(f"运行时错误: {e}")
                self.cap.release()
                self.root.quit()
        
        # 开始更新循环
        self.root.after(100, update)
        
        # 运行tkinter主循环
        self.root.mainloop()
        
        # 清理
        self.cap.release()

    def show_result(self, D, x):
        messagebox.showinfo("测量结果", f"距离 D: {D:.2f} cm\n尺寸 x: {x:.2f} cm")

    def log_error(self, msg):
        print(f"错误: {msg}")

    def save_log(self, D, x, t0):
        elapsed = time.time() - t0
        print(f"日志: D={D:.2f} cm, x={x:.2f} cm, 时间={elapsed:.2f} s")

if __name__ == "__main__":
    system = MeasurementSystem()
    system.run()