# -*- coding: utf-8 -*-
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_features(filepath):
    """
    从文本文件中加载特征数据。
    文件格式应为：'序号 feat1 feat2 ...'
    函数会跳过第一列（序号）。
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件未找到 -> {filepath}", file=sys.stderr)
        return None
    try:
        # np.loadtxt 默认以空格为分隔符
        # 我们只加载从第二列开始的数据 (usecols=range(1, ...))
        # 但更简单的方法是加载所有数据后进行切片
        data = np.loadtxt(filepath)
        if data.ndim == 1:  # 如果只有一行数据
            data = data.reshape(1, -1)
        return data[:, 1:]  # 返回除第一列外的所有数据
    except Exception as e:
        print(f"错误: 加载文件 {filepath} 时出错. {e}", file=sys.stderr)
        return None


def main(file_path1, file_path2):
    """主函数，用于加载、比较、可视化和打印特征"""
    print(f"正在比较文件:\n  1: {file_path1}\n  2: {file_path2}\n")

    # 1. 加载特征数据
    feats1 = load_features(file_path1)
    feats2 = load_features(file_path2)

    if feats1 is None or feats2 is None:
        sys.exit(1)  # 如果加载失败则退出

    # 检查维度是否一致
    if feats1.shape != feats2.shape:
        print("错误: 两个文件的特征矩阵维度不匹配!", file=sys.stderr)
        print(f"  - {os.path.basename(file_path1)} 的维度: {feats1.shape}", file=sys.stderr)
        print(f"  - {os.path.basename(file_path2)} 的维度: {feats2.shape}", file=sys.stderr)
        sys.exit(1)

    num_vectors, feat_dim = feats1.shape
    print(f"加载成功! 共 {num_vectors} 个特征向量，每个向量维度为 {feat_dim}。")
    print("-" * 50)

    # 2. 计算相似度/差异度指标

    # --- 指标A: 逐个特征向量进行比较 (更关注ReID性能) ---
    # 计算每对特征向量的余弦相似度
    per_image_cosine_sim = np.array([1 - cosine(v1, v2) for v1, v2 in zip(feats1, feats2)])
    # 计算每对特征向量的绝对差值总和
    per_image_abs_diff = np.sum(np.abs(feats1 - feats2), axis=1)

    # --- 指标B: 将所有特征视为一个大向量进行比较 (更关注数值精度) ---
    flat1 = feats1.flatten()
    flat2 = feats2.flatten()

    mse = mean_squared_error(flat1, flat2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(flat1, flat2)

    # 检查标准差是否为零，避免correlation计算返回NaN
    if np.std(flat1) == 0 or np.std(flat2) == 0:
        correlation = np.nan
    else:
        correlation = np.corrcoef(flat1, flat2)[0, 1]

    # 平均相对误差
    relative_error = np.mean(np.abs(flat1 - flat2) / (np.abs(flat1) + 1e-12)) * 100

    # 3. 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'特征比较: {os.path.basename(file_path1)} vs {os.path.basename(file_path2)}', fontsize=16)

    # 图1: 显示每个图像特征的余弦相似度
    ax1.plot(per_image_cosine_sim, marker='.', linestyle='-', markersize=4, label='Per-Image Cosine Similarity')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Similarity (1.0)')
    # 动态调整Y轴范围，以便更好地观察高相似度区域
    min_sim_val = np.min(per_image_cosine_sim)
    ax1.set_ylim(bottom=min(0.98, min_sim_val - 0.001), top=1.001)
    ax1.set_title('Per-Image Feature Cosine Similarity')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Cosine Similarity')
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 图2: 显示每个图像特征的绝对差异总和
    ax2.plot(per_image_abs_diff, color='orangered', marker='.', linestyle='-', markersize=4,
             label='Sum of Absolute Differences')
    ax2.set_title('Per-Image Absolute Feature Difference (L1 Distance)')
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Sum(|feat1 - feat2|)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 在图上添加一个包含所有关键指标的文本框
    mean_cosine_sim = np.mean(per_image_cosine_sim)
    min_cosine_sim_val = np.min(per_image_cosine_sim)

    metrics_text = (
        f"--- Per-Image Cosine Sim. ---\n"
        f"Mean: {mean_cosine_sim:.6f}\n"
        f"Min:  {min_cosine_sim_val:.6f} (at index {np.argmin(per_image_cosine_sim)})\n\n"
        f"--- Global Flat-Vector Metrics ---\n"
        f"Correlation: {correlation:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"MAE: {mae:.6f}\n"
        f"Mean Relative Error: {relative_error:.4f}%"
    )
    ax1.text(0.99, 0.98, metrics_text,
             fontsize=10, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应主标题
    plt.show()

    # 4. 打印详细统计报告
    print("=== 详细统计报告 ===")
    print("\n[逐向量余弦相似度统计]")
    print(f"  均值:   {mean_cosine_sim:.8f}")
    print(f"  标准差: {np.std(per_image_cosine_sim):.8f}")
    print(f"  最小值: {min_cosine_sim_val:.8f} (在第 {np.argmin(per_image_cosine_sim)} 个向量处)")
    print(f"  最大值: {np.max(per_image_cosine_sim):.8f} (在第 {np.argmax(per_image_cosine_sim)} 个向量处)")

    print("\n[全局数值精度指标 (基于所有特征值)]")
    print(f"  皮尔逊相关系数:    {correlation:.8f}")
    print(f"  均方根误差 (RMSE): {rmse:.8f}")
    print(f"  平均绝对误差 (MAE):{mae:.8f}")
    print(f"  平均相对误差:      {relative_error:.4f}%")
    print("=" * 28)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="比较两个 feature.txt 文件，并可视化相似度。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--file1',
        default='/home/manu/tmp/features_org.txt',
    )
    parser.add_argument(
        '--file2',
        default='/home/manu/tmp/features_cpp_onnx.txt',
    )
    args = parser.parse_args()

    main(args.file1, args.file2)
