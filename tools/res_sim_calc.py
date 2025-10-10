#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两个 output_result.txt

格式（含表头）:
frame_id,cam_id,tid,gid,score,n_tid

以 (frame_id, cam_id, tid) 为主键进行对齐：
1. 统计双方共有样本中 GID 是否一致
2. 计算 score/n_tid 的误差指标
3. 列出各类差异
4. 可视化

Author: your_name
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_txt(path):
    """
    读取 output_result.txt
    返回 dict: key -> dict 内容
    key = (frame_id, cam_id, tid)
    """
    res = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        name2idx = {name: i for i, name in enumerate(header)}

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            frame_id = int(parts[name2idx["frame_id"]])
            cam_id = parts[name2idx["cam_id"]]
            tid = int(parts[name2idx["tid"]])
            gid = parts[name2idx["gid"]]
            score = float(parts[name2idx["score"]])
            n_tid = int(parts[name2idx["n_tid"]])

            res[(frame_id, cam_id, tid)] = {
                "gid": gid,
                "score": score,
                "n_tid": n_tid
            }
    return res


def compute_metrics(vals1, vals2, key):
    """score / n_tid 的误差指标"""
    v1 = np.array([d[key] for d in vals1])
    v2 = np.array([d[key] for d in vals2])

    mse = mean_squared_error(v1, v2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(v1, v2)
    corr = np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else np.nan
    return dict(MSE=mse, RMSE=rmse, MAE=mae, Corr=corr)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two output_result.txt files")
    parser.add_argument("--file_a", default="/home/manu/tmp/output_result_py_org_v2.txt")
    parser.add_argument("--file_b", default="/home/manu/tmp/output_result_py.txt")
    # parser.add_argument("--file_b", default="/home/manu/nfs/output_result_cpp_load.txt")
    parser.add_argument("--show", default=True)
    parser.add_argument("--out_png", default="/home/manu/tmp/compare_result.png")
    args = parser.parse_args()

    if not os.path.isfile(args.file_a) or not os.path.isfile(args.file_b):
        raise FileNotFoundError("输入文件不存在!")

    print("Loading files …")
    res_a = load_txt(args.file_a)
    res_b = load_txt(args.file_b)

    keys_a = set(res_a.keys())
    keys_b = set(res_b.keys())
    common_keys = sorted(list(keys_a & keys_b))
    only_a = sorted(list(keys_a - keys_b))
    only_b = sorted(list(keys_b - keys_a))

    print(f"共有样本: {len(common_keys)}")
    print(f"A 独有样本: {len(only_a)}")
    print(f"B 独有样本: {len(only_b)}")

    # --------------------------------------
    # GID 一致性
    # --------------------------------------
    gid_match = []
    score_pairs = []
    n_pairs = []
    mismatch_gid_keys = []

    for k in common_keys:
        a, b = res_a[k], res_b[k]
        gid_match.append(int(a["gid"] == b["gid"]))
        score_pairs.append((a["score"], b["score"]))
        n_pairs.append((a["n_tid"], b["n_tid"]))
        if a["gid"] != b["gid"]:
            mismatch_gid_keys.append((k, a["gid"], b["gid"]))

    gid_accuracy = np.mean(gid_match) if gid_match else np.nan
    metrics_score = compute_metrics(
        [dict(score=s) for s in score_pairs], [dict(score=t) for t in score_pairs], "score")
    metrics_n = compute_metrics(
        [dict(n=s) for s in n_pairs], [dict(n=t) for t in n_pairs], "n")

    print("\n=== 对齐样本统计 ===")
    print(f"GID 一致率: {gid_accuracy * 100:.2f}% "
          f" ({sum(gid_match)}/{len(gid_match)})")

    print("\nScore 误差:")
    for k, v in metrics_score.items():
        print(f"  {k}: {v:.6f}")

    print("\nn_tid 误差:")
    for k, v in metrics_n.items():
        print(f"  {k}: {v:.6f}")

    # --------------------------------------
    # 可视化
    # --------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    idx = np.arange(len(common_keys))

    # (1) GID match / mismatch 可视化为 0-1
    ax1.step(idx, gid_match, where='mid', label='GID match (1=一致)', color='green')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_ylabel("GID Match")
    ax1.set_title("GID Match over common keys")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # (2) score difference
    score_diff = np.array([a - b for a, b in score_pairs])
    ax2.plot(idx, score_diff, label='score diff (A - B)', color='red')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax2.set_ylabel("Score Difference")
    ax2.set_xlabel("Aligned sample index")
    ax2.set_title("Score difference")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # 在图上写主要指标
    text = (f"GID  Accuracy: {gid_accuracy * 100:.2f}%\n"
            f"Score RMSE: {metrics_score['RMSE']:.6f}\n"
            f"Score MAE : {metrics_score['MAE']:.6f}\n"
            f"Score Corr: {metrics_score['Corr']:.4f}")
    ax1.text(0.01, 0.02, text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"\n图像已保存到 {args.out_png}")
    if args.show:
        plt.show()

    # --------------------------------------
    # 输出差异明细
    # --------------------------------------
    if only_a:
        print("\n*** 仅存在于 A 的样本 (最多显示 10 条) ***")
        for k in only_a[:10]:
            print(k)
    if only_b:
        print("\n*** 仅存在于 B 的样本 (最多显示 10 条) ***")
        for k in only_b[:10]:
            print(k)
    if mismatch_gid_keys:
        print("\n*** GID 不一致的样本 (最多显示 10 条) ***")
        for item in mismatch_gid_keys[:10]:
            k, gid_a, gid_b = item
            print(f"{k} | A:{gid_a}  B:{gid_b}")

    print("\n完成！")


if __name__ == "__main__":
    main()
