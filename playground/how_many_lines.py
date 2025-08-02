import os
import glob

def count_non_empty_lines(file_path):
    """统计单个文件的非空行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            non_empty_lines = [line for line in lines if line.strip()]
            return len(non_empty_lines)
    except (UnicodeDecodeError, PermissionError):
        # 如果文件编码问题或权限问题，尝试其他编码或跳过
        try:
            with open(file_path, 'r', encoding='gbk') as file:
                lines = file.readlines()
                non_empty_lines = [line for line in lines if line.strip()]
                return len(non_empty_lines)
        except:
            print(f"警告: 无法读取文件 {file_path}")
            return 0

def count_all_py_files():
    """统计当前目录下的.py文件行数"""
    total_lines = 0
    total_files = 0
    
    # 使用glob查找当前目录下的.py文件
    py_files = glob.glob('*.py')
    
    print("Python文件统计结果:")
    print("-" * 50)
    
    for file_path in py_files:
        line_count = count_non_empty_lines(file_path)
        total_lines += line_count
        total_files += 1
        print(f"{file_path}: {line_count} 行")
    
    print("-" * 50)
    print(f"总计: {total_files} 个文件, {total_lines} 行代码（不含空行）")

if __name__ == "__main__":
    count_all_py_files()