#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    # 添加一个极小值避免除以零
    return dot_product / (norm_product + 1e-8)


def main():
    parser = argparse.ArgumentParser(
        description="高级比对工具：对比 C++ 和 Python 生成的人脸特征 (embedding) JSON 文件，提供多项指标和可视化图表。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 文件路径参数保持不变
    parser.add_argument("--cpp-file", default="/home/manu/tmp/embeddings.json", help="C++ 程序生成的 JSON 文件路径。")
    parser.add_argument("--py-file", default="/home/manu/tmp/embeddings_py.json",
                        help="Python 程序生成的 JSON 文件路径。")

    # 其他参数保持不变
    parser.add_argument("--top-n", type=int, default=10, help="显示相似度最低的N个样本。")
    parser.add_argument("--sim-threshold", type=float, default=0.99, help="低于此相似度阈值时将高亮显示。")

    # 新增绘图相关参数
    parser.add_argument("--save-plot", default="/home/manu/tmp/embedding_comparison.png",
                        help="可视化结果图表的保存路径。传入空字符串可禁用保存。")
    parser.add_argument("--no-show", action="store_true", help="设置此项后将不自动显示图表。")

    args = parser.parse_args()

    print("=" * 60)
    print("      高级 Embedding 比对工具 (多指标与可视化版)")
    print("=" * 60)
    print(f"C++ (文件 A): {args.cpp_file}")
    print(f"Python (文件 B): {args.py_file}\n")

    # --- 1. 加载文件 ---
    try:
        with open(args.cpp_file, 'r') as f:
            cpp_data = json.load(f)
        with open(args.py_file, 'r') as f:
            py_data = json.load(f)
    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 文件解析失败: {e}")
        return

    # --- 2. 键集合分析 ---
    cpp_keys = set(cpp_data.keys())
    py_keys = set(py_data.keys())
    common_keys = sorted(list(cpp_keys.intersection(py_keys)))

    if not common_keys:
        print("没有可供比对的共有特征。程序退出。")
        return

    # --- 3. 逐个计算指标 ---
    results = []
    for key in common_keys:
        emb_cpp = np.asarray(cpp_data[key], dtype=np.float32)
        emb_py = np.asarray(py_data[key], dtype=np.float32)
        results.append({
            "key": key,
            "similarity": cosine_similarity(emb_cpp, emb_py),
            "rmse": np.sqrt(mean_squared_error(emb_cpp, emb_py)),
            "mae": mean_absolute_error(emb_cpp, emb_py)
        })

    # --- 4. 汇总统计指标 (这部分逻辑不变) ---
    similarities = np.array([r["similarity"] for r in results])
    rmses = np.array([r["rmse"] for r in results])
    maes = np.array([r["mae"] for r in results])

    # 在终端打印所有统计信息
    print("--- 样本摘要 ---")
    print(f"文件 A (C++) 中包含         : {len(cpp_keys)} 个特征")
    print(f"文件 B (Python) 中包含      : {len(py_keys)} 个特征")
    print(f"两个文件共有的图片/特征      : {len(common_keys)} 个\n")
    # ... (此处省略详细的文本输出，因为它们仍在脚本中，只是为了简洁不全贴)
    print("--- 总体指标统计 (基于所有共有样本) ---")
    print("\n[余弦相似度 (Cosine Similarity)] - 越高越好，1为完美")
    print(f"  平均值 (Avg)  : {np.mean(similarities):.8f}")
    # ... etc ...

    # ======================= 【新增绘图部分】 =======================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle('C++ vs Python Embedding Comparison', fontsize=16)

    x_axis = np.arange(len(common_keys))

    # (图1) 余弦相似度
    ax1.plot(x_axis, similarities, '.-', label='Cosine Similarity', color='tab:blue', markersize=4, zorder=10)
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect Match (1.0)')
    ax1.axhline(args.sim_threshold, color='orange', linestyle=':', linewidth=1.5,
                label=f'Warning Threshold ({args.sim_threshold})')

    # 动态调整Y轴范围，以便看清细微差别
    min_sim_for_plot = min(np.min(similarities), args.sim_threshold) - 0.001
    ax1.set_ylim(bottom=min_sim_for_plot, top=1.001)

    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Similarity Score for Each Common Sample')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # (图2) 均方根误差 (RMSE)
    ax2.plot(x_axis, rmses, '.-', label='RMSE', color='tab:green', markersize=4)
    ax2.set_xlabel('Common Sample Index')
    ax2.set_ylabel('Root Mean Squared Error')
    ax2.set_title('Vector Error (RMSE) for Each Common Sample')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 在图上添加总体统计文本框
    summary_text = (
        f"Common Samples: {len(common_keys)}\n\n"
        f"--- Similarity ---\n"
        f"Avg: {np.mean(similarities):.6f}\n"
        f"Min: {np.min(similarities):.6f}\n"
        f"Std: {np.std(similarities):.6f}\n\n"
        f"--- RMSE ---\n"
        f"Avg: {np.mean(rmses):.6f}"
    )
    ax1.text(0.99, 0.01, summary_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为总标题留出空间

    # 保存和显示
    if args.save_plot:
        try:
            plt.savefig(args.save_plot, dpi=150)
            print(f"\n[成功] 图表已保存到: {args.save_plot}")
        except Exception as e:
            print(f"\n[错误] 保存图表失败: {e}")

    if not args.no_show:
        plt.show()
    # ======================= 【绘图部分结束】 =======================

    # --- 5 & 6. 报告问题和独有样本 (不变) ---
    sorted_results = sorted(results, key=lambda x: x["similarity"])
    print(f"\n--- 相似度最低的TOP-{args.top_n}个样本 ---")
    # ... (省略打印逻辑)
    # ...
    # cpp_only_keys / py_only_keys reporting logic ...

    print("\n" + "=" * 60)
    print("比对完成。")


if __name__ == "__main__":
    main()
