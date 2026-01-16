import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==========================================
# 0. 内部参数配置 (这些通常不需要经常改，保留在内部)
# ==========================================
BATCH_SIZE = 100        
SAMPLE_COUNT = 5        
TARGET_ELEMENT = 'Nb'   
TARGET_WAVELENGTH = 407.98  
SEARCH_BORDER = 3       

# ==========================================
# 1. 辅助工具 (原样保留)
# ==========================================
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def load_concentrations(filepath, target_col):
    df = pd.read_excel(filepath)
    df = df.sort_values(by='文件编号', ascending=True).reset_index(drop=True)
    return df[target_col].astype(float).values

def load_spectra_data(spec_dir, sample_count, batch_size):
    print(f"正在读取光谱数据 (路径: {spec_dir})...")
    files = [f for f in os.listdir(spec_dir) if f.startswith('spec_') and f.endswith('.csv')]
    files.sort(key=natural_keys)
    all_wavelengths = None
    all_intensities = []
    for i in range(sample_count):
        df = pd.read_csv(os.path.join(spec_dir, files[i]), header=None)
        if all_wavelengths is None: all_wavelengths = df.values[:, 0]
        all_intensities.append(df.values[:, 1:1+batch_size].T)
    return all_wavelengths, np.vstack(all_intensities)

def load_image_data(image_dir, sample_count, batch_size):
    print(f"正在读取图像数据 (路径: {image_dir})...")
    folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    folders.sort(key=natural_keys)
    all_images = []
    for folder_name in folders[:sample_count]:
        folder_path = os.path.join(image_dir, folder_name)
        img_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        img_files.sort(key=natural_keys)
        sample_imgs = [pd.read_csv(os.path.join(folder_path, f), header=None).values.flatten() for f in tqdm(img_files[:batch_size], desc=f"加载样品{folder_name}")]
        all_images.append(np.array(sample_imgs, dtype=np.float32))
    return np.vstack(all_images)


# 全噪声实验用的图片加载函数 (只读维度，省内存)
def load_image_data_placeholder(image_dir, sample_count, batch_size):
    print("正在读取图像数据结构 (仅获取维度)...")
    folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    folders.sort(key=natural_keys)
    all_images = []
    for folder_name in folders[:sample_count]:
        folder_path = os.path.join(image_dir, folder_name)
        img_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        img_files.sort(key=natural_keys)
        # 仅加载第一张图片以获取展平后的维度
        temp_img = pd.read_csv(os.path.join(folder_path, img_files[0]), header=None).values.flatten()
        img_dim = temp_img.shape[0]
        # 创建一个空矩阵占据该样品的空间
        sample_imgs = np.zeros((batch_size, img_dim), dtype=np.float32)
        all_images.append(sample_imgs)
    return np.vstack(all_images)


# 全真实数据定标 
# ==========================================
def real_data_experiment(content_file, spec_dir, image_dir):
    """逻辑：真实光谱 + 真实图像"""
    content_vals = load_concentrations(content_file, TARGET_ELEMENT)
    content_expanded = np.repeat(content_vals, BATCH_SIZE)
    wavelengths, X_spec = load_spectra_data(spec_dir, SAMPLE_COUNT, BATCH_SIZE)
    X_image = load_image_data(image_dir, SAMPLE_COUNT, BATCH_SIZE)
    
    # 核心算法 (原样保留)
    test_idx = 2
    is_test = np.zeros(len(content_expanded), dtype=bool)
    is_test[test_idx*BATCH_SIZE : (test_idx+1)*BATCH_SIZE] = True
    
    print("\n执行真实数据定标...")
    w_idx = (np.abs(wavelengths - TARGET_WAVELENGTH)).argmin()
    start_idx, end_idx = max(0, w_idx - SEARCH_BORDER), min(X_spec.shape[1], w_idx + SEARCH_BORDER + 1)
    peak_intensities = np.max(X_spec[:, start_idx:end_idx], axis=1)
    
    itrain, itest = peak_intensities[~is_test], peak_intensities[is_test]
    ctrain, ctest = content_expanded[~is_test], content_expanded[is_test]
    p_init = np.polyfit(ctrain, itrain, 1)
    etrain = itrain - np.polyval(p_init, ctrain)
    
    pca_s = PCA(n_components=20).fit_transform(X_spec)
    pca_i = PCA(n_components=10).fit_transform(X_image)
    X_pca = np.hstack([pca_s, pca_i])
    
    scaler = StandardScaler()
    xtrain_s = scaler.fit_transform(X_pca[~is_test])
    xtest_s = scaler.transform(X_pca[is_test])
    net = MLPRegressor(hidden_layer_sizes=(15,), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
    net.fit(xtrain_s, etrain)
    
    itrain_cor = itrain - net.predict(xtrain_s)
    itest_cor = itest - net.predict(xtest_s)
    
 #  结果聚合与 R2 计算 
    train_samples_indices = [0, 1, 3, 4]
    final_train = []
    for i, s_idx in enumerate(train_samples_indices):
        seg = itrain_cor[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        final_train.append([content_vals[s_idx], np.mean(seg), np.std(seg, ddof=1)])
    
    train_arr = np.array(final_train)
    test_arr = np.array([[content_vals[test_idx], np.mean(itest_cor), np.std(itest_cor, ddof=1)]])
    
    # 归一化（便于绘图）
    n_min, n_max = train_arr[:,1].min(), train_arr[:,1].max()
    train_arr[:,1] = (train_arr[:,1] - n_min) / (n_max - n_min)
    test_arr[:,1] = (test_arr[:,1] - n_min) / (n_max - n_min)
    norm_factor = n_max - n_min

    #计算 R2
    # 使用训练集样品的平均值点进行拟合
    slope, intercept = np.polyfit(train_arr[:,0], train_arr[:,1], 1)
    y_pred_train = slope * train_arr[:,0] + intercept
    r2_val = r2_score(train_arr[:,1], y_pred_train)
    
    print(f"\n定标完成！训练集 R^2 = {r2_val:.4f}")

    #  绘图
    plt.figure(figsize=(8, 6), dpi=100)
    
    # 绘制训练集点
    plt.errorbar(train_arr[:,0], train_arr[:,1], yerr=train_arr[:,2]/norm_factor, 
                 fmt='bo', markersize=8, capsize=5, label=f'Train Samples (R$^2$={r2_val:.4f})', fillstyle='none', markeredgewidth=1.5)
    
    # 绘制测试集点
    plt.errorbar(test_arr[:,0], test_arr[:,1], yerr=test_arr[:,2]/norm_factor, 
                 fmt='rs', markersize=8, capsize=5, label='Test Sample', fillstyle='none', markeredgewidth=1.5)
    
    # 绘制拟合直线
    x_range = np.array([content_vals.min()*0.9, content_vals.max()*1.1])
    plt.plot(x_range, slope * x_range + intercept, 'k--', alpha=0.6, label='Linear Fit')
    
    plt.xlabel(f'{TARGET_ELEMENT} Concentration (wt.%)', fontsize=12)
    plt.ylabel('Normalized Intensity (a.u.)', fontsize=12)
    plt.title(f'Calibration Curve for {TARGET_ELEMENT} (Fusion Model)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 在图中直接标注 R2 文本
    plt.text(0.05, 0.95, f'$R^2 = {r2_val:.4f}$', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 核心功能函数 (接收路径作为参数)
# ==========================================
def image_noise_experiment(content_file_path, spec_dir_path, image_dir_path):
    """
    执行噪声测试实验的主逻辑。
    参数:
        content_file_path: 样品含量表.xls 的完整路径
        spec_dir_path: 光谱文件夹 (spec) 的路径
        image_dir_path: 图像文件夹 (picture) 的路径
    """
    
    #  数据加载 
    # 使用传入的参数进行加载
    content_vals = load_concentrations(content_file_path, TARGET_ELEMENT)
    content_expanded = np.repeat(content_vals, BATCH_SIZE)
    
    # 加载真实光谱仅为了获取维度和波长信息
    wavelengths, X_spec_real = load_spectra_data(spec_dir_path, SAMPLE_COUNT, BATCH_SIZE)
    X_image = load_image_data(image_dir_path, SAMPLE_COUNT, BATCH_SIZE)
    
    # 【修改部分】：将光谱数据替换为相同形状的随机噪声
    print("\n--- 实验：正在将光谱数据替换为随机噪声 ---")
    X_spec = np.random.normal(loc=0.0, scale=1.0, size=X_spec_real.shape) 
    
    #  数据划分
    test_idx = 2  # 样品3作为测试
    is_test = np.zeros(len(content_expanded), dtype=bool)
    is_test[test_idx*BATCH_SIZE : (test_idx+1)*BATCH_SIZE] = True
    
    #   基础定标
    print("执行基础定标（基于噪声数据）...")
    w_idx = (np.abs(wavelengths - TARGET_WAVELENGTH)).argmin()
    start_idx, end_idx = max(0, w_idx - SEARCH_BORDER), min(X_spec.shape[1], w_idx + SEARCH_BORDER + 1)
    peak_intensities = np.max(X_spec[:, start_idx:end_idx], axis=1)
    
    itrain, itest = peak_intensities[~is_test], peak_intensities[is_test]
    ctrain, ctest = content_expanded[~is_test], content_expanded[is_test]
    
    p_init = np.polyfit(ctrain, itrain, 1)
    etrain = itrain - np.polyval(p_init, ctrain)
    
    #  神经网络修正 
    print("执行 PCA 与 NN 修正 (Input: Noise + Real Images)...")
    pca_s = PCA(n_components=20).fit_transform(X_spec)
    pca_i = PCA(n_components=10).fit_transform(X_image)
    X_pca = np.hstack([pca_s, pca_i])
    
    scaler = StandardScaler()
    xtrain_s = scaler.fit_transform(X_pca[~is_test])
    xtest_s = scaler.transform(X_pca[is_test])
    
    net = MLPRegressor(hidden_layer_sizes=(15,), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
    net.fit(xtrain_s, etrain)
    
    itrain_cor = itrain - net.predict(xtrain_s)
    itest_cor = itest - net.predict(xtest_s)
    
    #结果聚合与 R2 计算
    train_samples_indices = [0, 1, 3, 4]
    final_train = []
    for i, s_idx in enumerate(train_samples_indices):
        seg = itrain_cor[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        final_train.append([content_vals[s_idx], np.mean(seg), np.std(seg, ddof=1)])
    
    train_arr = np.array(final_train)
    test_arr = np.array([[content_vals[test_idx], np.mean(itest_cor), np.std(itest_cor, ddof=1)]])
    
    # 归一化
    n_min, n_max = train_arr[:,1].min(), train_arr[:,1].max()
    train_arr[:,1] = (train_arr[:,1] - n_min) / (n_max - n_min)
    test_arr[:,1] = (test_arr[:,1] - n_min) / (n_max - n_min)
    norm_factor = n_max - n_min

    #  计算 R2 
    slope, intercept = np.polyfit(train_arr[:,0], train_arr[:,1], 1)
    y_pred_train = slope * train_arr[:,0] + intercept
    r2_val = r2_score(train_arr[:,1], y_pred_train)
    
    print(f"\n实验完成！(光谱=噪声, 图像=真实)")
    print(f"训练集 R^2 = {r2_val:.4f}")

    #  绘图 
    plt.figure(figsize=(8, 6), dpi=100)
    plt.errorbar(train_arr[:,0], train_arr[:,1], yerr=train_arr[:,2]/norm_factor, 
                 fmt='bo', markersize=8, capsize=5, label=f'Train (Noise Spectrum) R$^2$={r2_val:.4f}', fillstyle='none', markeredgewidth=1.5)
    plt.errorbar(test_arr[:,0], test_arr[:,1], yerr=test_arr[:,2]/norm_factor, 
                 fmt='rs', markersize=8, capsize=5, label='Test Sample', fillstyle='none', markeredgewidth=1.5)
    
    x_range = np.array([content_vals.min()*0.9, content_vals.max()*1.1])
    plt.plot(x_range, slope * x_range + intercept, 'k--', alpha=0.6, label='Linear Fit')
    
    plt.xlabel(f'{TARGET_ELEMENT} Concentration (wt.%)', fontsize=12)
    plt.ylabel('Normalized Intensity (a.u.)', fontsize=12)
    plt.title(f'Calibration with NOISE Spectrum & Real Images', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.text(0.05, 0.95, f'$R^2 = {r2_val:.4f}$', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def full_noise_experiment(content_file_path, spec_dir_path, image_dir_path):
    #  数据加载
    content_vals = load_concentrations(content_file_path, TARGET_ELEMENT)
    content_expanded = np.repeat(content_vals, BATCH_SIZE)
    
    # 获取光谱和图像的原始维度
    wavelengths, X_spec_real = load_spectra_data(spec_dir_path, SAMPLE_COUNT, BATCH_SIZE)
    # 注意：这里使用 placeholder 加载器，因为我们不需要真实像素值
    X_image_real = load_image_data_placeholder(image_dir_path, SAMPLE_COUNT, BATCH_SIZE)
    
    # 【修改部分】：全部替换为随机噪声
    print("\n--- 光谱与图像数据均替换为噪声 ---")
    # 生成光谱噪声 (形状同 X_spec_real)
    X_spec = np.random.normal(loc=0.0, scale=1.0, size=X_spec_real.shape) 
    
    # 生成图像噪声 (形状同 X_image_real)
    X_image = np.random.normal(loc=0.0, scale=1.0, size=X_image_real.shape)
    
    #  数据划分
    test_idx = 2  
    is_test = np.zeros(len(content_expanded), dtype=bool)
    is_test[test_idx*BATCH_SIZE : (test_idx+1)*BATCH_SIZE] = True
    
    #   基础定标 
    print("执行基础定标（基于噪声光谱）...")
    w_idx = (np.abs(wavelengths - TARGET_WAVELENGTH)).argmin()
    start_idx, end_idx = max(0, w_idx - SEARCH_BORDER), min(X_spec.shape[1], w_idx + SEARCH_BORDER + 1)
    peak_intensities = np.max(X_spec[:, start_idx:end_idx], axis=1)
    
    itrain, itest = peak_intensities[~is_test], peak_intensities[is_test]
    ctrain, ctest = content_expanded[~is_test], content_expanded[is_test]
    
    p_init = np.polyfit(ctrain, itrain, 1)
    etrain = itrain - np.polyval(p_init, ctrain)
    
    #  神经网络修正 
    print("执行 PCA 与 NN 修正 (Input: Total Noise)...")
    pca_s = PCA(n_components=20).fit_transform(X_spec)
    pca_i = PCA(n_components=10).fit_transform(X_image)
    X_pca = np.hstack([pca_s, pca_i])
    
    scaler = StandardScaler()
    xtrain_s = scaler.fit_transform(X_pca[~is_test])
    xtest_s = scaler.transform(X_pca[is_test])
    
    # 保持参数一致
    net = MLPRegressor(hidden_layer_sizes=(15,), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
    net.fit(xtrain_s, etrain)
    
    itrain_cor = itrain - net.predict(xtrain_s)
    itest_cor = itest - net.predict(xtest_s)
    
    # 结果聚合与 R2 计算 
    train_samples_indices = [0, 1, 3, 4]
    final_train = []
    for i, s_idx in enumerate(train_samples_indices):
        seg = itrain_cor[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        final_train.append([content_vals[s_idx], np.mean(seg), np.std(seg, ddof=1)])
    
    train_arr = np.array(final_train)
    test_arr = np.array([[content_vals[test_idx], np.mean(itest_cor), np.std(itest_cor, ddof=1)]])
    
    # 归一化
    n_min, n_max = train_arr[:,1].min(), train_arr[:,1].max()
    train_arr[:,1] = (train_arr[:,1] - n_min) / (n_max - n_min)
    test_arr[:,1] = (test_arr[:,1] - n_min) / (n_max - n_min)
    norm_factor = n_max - n_min

    #计算 R2
    slope, intercept = np.polyfit(train_arr[:,0], train_arr[:,1], 1)
    y_pred_train = slope * train_arr[:,0] + intercept
    r2_val = r2_score(train_arr[:,1], y_pred_train)
    
    print(f"\n实验完成！(全噪声模拟)")
    print(f"训练集 R^2 = {r2_val:.4f}")

    #  绘图 
    plt.figure(figsize=(8, 6), dpi=100)
    plt.errorbar(train_arr[:,0], train_arr[:,1], yerr=train_arr[:,2]/norm_factor, 
                 fmt='go', markersize=8, capsize=5, label=f'Train (All Noise) R$^2$={r2_val:.4f}', fillstyle='none', markeredgewidth=1.5)
    plt.errorbar(test_arr[:,0], test_arr[:,1], yerr=test_arr[:,2]/norm_factor, 
                 fmt='rs', markersize=8, capsize=5, label='Test Sample', fillstyle='none', markeredgewidth=1.5)
    
    x_range = np.array([content_vals.min()*0.9, content_vals.max()*1.1])
    plt.plot(x_range, slope * x_range + intercept, 'k--', alpha=0.6, label='Linear Fit')
    
    plt.xlabel(f'{TARGET_ELEMENT} Concentration (wt.%)', fontsize=12)
    plt.ylabel('Normalized Intensity (a.u.)', fontsize=12)
    plt.title(f'Calibration with COMPLETE NOISE (Spec + Image)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.text(0.05, 0.95, f'$R^2 = {r2_val:.4f}$', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()