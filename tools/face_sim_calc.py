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
    从特征文件加载数据，自动跳过第一列（可能是字符串ID）。
    支持逗号、空格、Tab分隔。
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件未找到 -> {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
        delimiter = ',' if ',' in first_line else None

        # 用 genfromtxt 可以解析字符串，并跳过第一列
        data = np.genfromtxt(filepath, delimiter=delimiter, dtype=str)  # 全部作为str读取
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # 从第二列开始转成 float
        feats = data[:, 1:].astype(np.float64)
        return feats
    except Exception as e:
        print(f"错误: 加载文件 {filepath} 时出错: {e}", file=sys.stderr)
        return None


def main(file_path1, file_path2):
    """主函数，用于加载、比较、可视化和打印特征"""
    print(f"正在比较文件:\n  1: {file_path1}\n  2: {file_path2}\n")

    # 1. 加载特征数据
    feats1 = load_features(file_path1)
    feats2 = load_features(file_path2)

    if feats1 is None or feats2 is None:
        sys.exit(1)  # 如果加载失败则退出

    # 检查维度
    if feats1.shape != feats2.shape:
        print("错误: 两个文件的特征矩阵维度不匹配!", file=sys.stderr)
        print(f"  - {os.path.basename(file_path1)} 的维度: {feats1.shape}", file=sys.stderr)
        print(f"  - {os.path.basename(file_path2)} 的维度: {feats2.shape}", file=sys.stderr)
        sys.exit(1)

    num_vectors, feat_dim = feats1.shape
    print(f"加载成功! 共 {num_vectors} 个特征向量，每个向量维度为 {feat_dim}。")
    print("-" * 50)

    # 2. 计算逐向量指标
    per_image_cosine_sim = np.array([1 - cosine(v1, v2) for v1, v2 in zip(feats1, feats2)])
    per_image_abs_diff = np.sum(np.abs(feats1 - feats2), axis=1)

    # 打印前10行对比详情
    print("\n=== 每个向量对比结果（前10个） ===")
    for i, (cos_sim, l1_diff) in enumerate(zip(per_image_cosine_sim, per_image_abs_diff)):
        if i < 10:
            print(f"Index {i:03d}: CosSim={cos_sim:.6f}, L1_Diff={l1_diff:.6f}")
    if num_vectors > 10:
        print("...")

    # 3. 全局向量指标
    flat1 = feats1.flatten()
    flat2 = feats2.flatten()

    mse = mean_squared_error(flat1, flat2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(flat1, flat2)
    correlation = np.nan if np.std(flat1) == 0 or np.std(flat2) == 0 else np.corrcoef(flat1, flat2)[0, 1]
    relative_error = np.mean(np.abs(flat1 - flat2) / (np.abs(flat1) + 1e-12)) * 100

    # 4. 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'特征比较: {os.path.basename(file_path1)} vs {os.path.basename(file_path2)}', fontsize=16)

    ax1.plot(per_image_cosine_sim, marker='.', linestyle='-', markersize=4, label='Per-Image Cosine Similarity')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect (1.0)')
    min_sim_val = np.min(per_image_cosine_sim)
    ax1.set_ylim(bottom=min(0.98, min_sim_val - 0.001), top=1.001)
    ax1.set_title('Per-Image Feature Cosine Similarity')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Cosine Similarity')
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(per_image_abs_diff, color='orangered', marker='.', linestyle='-', markersize=4,
             label='Sum of Absolute Differences')
    ax2.set_title('Per-Image Absolute Feature Difference (L1 Distance)')
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Sum(|feat1 - feat2|)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    metrics_text = (
        f"--- Per-Image Cosine Sim. ---\n"
        f"Mean: {np.mean(per_image_cosine_sim):.6f}\n"
        f"Min:  {np.min(per_image_cosine_sim):.6f} (idx {np.argmin(per_image_cosine_sim)})\n\n"
        f"--- Global Metrics ---\n"
        f"Correlation: {correlation:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"MAE: {mae:.6f}\n"
        f"Mean RelErr: {relative_error:.4f}%"
    )
    ax1.text(0.99, 0.98, metrics_text,
             fontsize=10, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 5. 打印统计报告
    print("\n=== 详细统计报告 ===")
    print(f"[逐向量余弦相似度] 均值: {np.mean(per_image_cosine_sim):.8f}  "
          f"标准差: {np.std(per_image_cosine_sim):.8f}  "
          f"最小值: {np.min(per_image_cosine_sim):.8f} (idx {np.argmin(per_image_cosine_sim)})  "
          f"最大值: {np.max(per_image_cosine_sim):.8f} (idx {np.argmax(per_image_cosine_sim)})")
    print(f"[全局指标] 相关系数: {correlation:.8f}  RMSE: {rmse:.8f}  "
          f"MAE: {mae:.8f}  平均相对误差: {relative_error:.4f}%")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="比较两个 feature.txt 文件，并可视化相似度。",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--file1', default='/home/manu/tmp/embeddings_py.txt')
    parser.add_argument('--file2', default='/home/manu/nfs/embeddings_cpp_from_aligned_bmps.txt')
    args = parser.parse_args()

    main(args.file1, args.file2)
