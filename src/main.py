import os
import requests
# 从模块中导入三个不同的实验函数
from spec_image_experiment.my_module import (
    real_data_experiment, 
    image_noise_experiment, 
    full_noise_experiment
)

def main():
    print(">>> [配置阶段] 正在初始化路径...")

    # ==========================================
    # 在这里修改你的文件地址
    # ==========================================
    # 根目录 (修改这里即可影响子文件夹)
    PATH_ROOT = r"D:\word文档\博士生项目\图谱融合\spec-image-experiment\spec-image-experiment\DATA"
    
    # 具体的子路径 (根据根目录自动生成，也可以手动修改成其他绝对路径)
    my_content_file = os.path.join(PATH_ROOT, "样品含量表.xls")
    my_image_dir = os.path.join(PATH_ROOT, "picture")
    my_spec_dir = os.path.join(PATH_ROOT, "spec")

    # ==========================================
    
    print(f" -> 含量表: {my_content_file}")
    print(f" -> 图像库: {my_image_dir}")
    print(f" -> 光谱库: {my_spec_dir}")

# ==========================================
    # 实验选择区域 (想跑哪个实验，就取消哪一行的注释)
    # ==========================================
    
    # --- 实验 1: 真实数据 (image-spe-test-1.py) ---
    print("\n>>> 启动实验 1: 真实数据定标")
    real_data_experiment(my_content_file, my_spec_dir, my_image_dir)

    # --- 实验 2: 光谱噪声 + 真实图像 (noise-test.py) ---
    #print("\n>>> 启动实验 2: 光谱噪声实验")
    #image_noise_experiment(my_content_file, my_spec_dir, my_image_dir)

    # --- 实验 3: 全噪声模拟 (noise_noise_test.py) ---
    #print("\n>>> 启动实验 3: 全噪声模拟实验")
    #full_noise_experiment(my_content_file, my_spec_dir, my_image_dir)
    
    print("\n>>> 所有选定实验运行结束。")

if __name__ == "__main__":
    main()