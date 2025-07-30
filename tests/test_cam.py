import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import glob
from system_initializer import MeasurementSystem

class CameraApp:
    def __init__(self):
        # 初始化系统
        self.system = MeasurementSystem("calib.yaml", crop_width=500)
        self.current_frame = None
        
        # 确保保存目录存在
        self.save_dir = "images/dataset"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建tkinter窗口
        self.root = tk.Tk()
        self.root.title("相机拍照系统")
        self.root.geometry("200x100")
        
        # 创建拍照按钮
        self.capture_btn = tk.Button(
            self.root, 
            text="拍照", 
            command=self.take_photo,
            font=("Arial", 16),
            bg="lightblue",
            width=10,
            height=2
        )
        self.capture_btn.pack(pady=20)
    
    def get_next_filename(self):
        """获取下一个文件名，从0001开始编号"""
        pattern = os.path.join(self.save_dir, "*.png")
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            return os.path.join(self.save_dir, "0001.png")
        
        # 提取所有数字编号
        numbers = []
        for file in existing_files:
            basename = os.path.basename(file)
            name_without_ext = os.path.splitext(basename)[0]
            if name_without_ext.isdigit():
                numbers.append(int(name_without_ext))
        
        # 获取最大编号并加1
        max_num = max(numbers) if numbers else 0
        next_num = max_num + 1
        
        return os.path.join(self.save_dir, f"{next_num:04d}.png")
    
    def take_photo(self):
        """拍照并保存"""
        if self.current_frame is not None:
            filename = self.get_next_filename()
            cv2.imwrite(filename, self.current_frame)
            print(f"照片已保存: {filename}")
            messagebox.showinfo("成功", f"照片已保存: {os.path.basename(filename)}")
        else:
            messagebox.showwarning("警告", "没有可用的图像帧")
    
    def run(self):
        """运行主循环"""
        def update_frame():
            frame = self.system.capture_frame()
            if frame is not None:
                self.current_frame = frame
                print(frame.shape)
                cv2.imshow("Original Frame", frame)
            
            # 检查是否按下'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                return
            
            # 继续更新帧
            self.root.after(10, update_frame)
        
        # 开始更新帧
        update_frame()
        
        # 启动tkinter主循环
        self.root.mainloop()
    
    def cleanup(self):
        """清理资源"""
        self.system.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()

# 创建并运行应用
if __name__ == "__main__":
    app = CameraApp()
    app.run()