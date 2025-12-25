#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较两个 gid 目录的人脸相似度
注意：假设输入图片已经是裁剪对齐好的人脸（如 112x112），跳过检测步骤直接提取特征。
"""

import os
import sys
from glob import glob
from typing import List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# =========================================================
#              在这里改要比对的两个 gid
# =========================================================
GID_A = "G00105"
GID_B = "G00036"
# =========================================================

SAVE_DIR = "/home/manu/mnt/perimeter_201/perimeter_cpp/"  # 图片存储根目录


def norm(v: np.ndarray) -> np.ndarray:
    """归一化向量"""
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def get_aligned_embedding(app: FaceAnalysis, img: np.ndarray) -> Optional[np.ndarray]:
    """
    从已对齐的人脸图像中直接提取特征，跳过检测步骤。
    """
    # 1. 获取识别模型 (ArcFace)
    rec_model = app.models.get('recognition', None)
    if rec_model is None:
        # 如果字典键名不对，尝试遍历查找
        for model in app.models.values():
            if 'recognition' in str(type(model)).lower():
                rec_model = model
                break

    if rec_model is None:
        print("[Error] 未找到识别模型 (Recognition Model)")
        return None

    # 2. 直接提取特征 (跳过对齐)
    try:
        # 因为图片已经是 112x112 的对齐人脸，直接调用 get_feat 即可
        # rec_model.get() 内部会尝试做 warp/crop，如果没有 kps 会报错
        emb = rec_model.get_feat(img)
        if emb is not None:
            return norm(emb.flatten())
    except Exception as e:
        print(f"[Warn] 特征提取失败: {e}")

    return None


def load_face_feats(paths: List[str], app: FaceAnalysis) -> List[np.ndarray]:
    """读取图片列表并提取特征"""
    feats = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue

        # 确保图片尺寸符合模型输入 (标准 ArcFace 输入通常为 112x112)
        # 如果你的对齐图片尺寸不一致，这里强制 resize，否则模型可能会报错或效果变差
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112))

        emb = get_aligned_embedding(app, img)
        if emb is not None:
            feats.append(emb)

    return feats


def avg(feats: List[np.ndarray]) -> Optional[np.ndarray]:
    """计算特征中心（平均值）并归一化"""
    if not feats:
        return None
    # axis=0 表示在数量维度求平均
    return norm(np.mean(np.stack(feats, axis=0), axis=0))


def compare(gid1: str, gid2: str):
    d1 = os.path.join(SAVE_DIR, gid1)
    d2 = os.path.join(SAVE_DIR, gid2)

    if not (os.path.isdir(d1) and os.path.isdir(d2)):
        sys.exit(f"gid 目录不存在: {d1} 或 {d2}")

    # -------- 1. 初始化模型 --------
    # providers: ['CUDAExecutionProvider'] for GPU, ['CPUExecutionProvider'] for CPU
    print("正在初始化 InsightFace 模型...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    # ctx_id=0 使用第一个 GPU，det_size 在这里不重要因为我们跳过了检测
    app.prepare(ctx_id=0, det_size=(640, 640))

    # -------- 2. 读取文件列表 --------
    # 假设人脸图在 gid/faces/ 目录下
    f1 = sorted(glob(os.path.join(d1, "faces", "*.jpg")))
    f2 = sorted(glob(os.path.join(d2, "faces", "*.jpg")))

    if not f1:
        print(f"[Warn] {gid1} 下没有找到 .jpg 图片")
    if not f2:
        print(f"[Warn] {gid2} 下没有找到 .jpg 图片")

    # -------- 3. 提取特征 --------
    print(f"正在提取 {gid1} ({len(f1)} 张)...")
    feats1 = load_face_feats(f1, app)

    print(f"正在提取 {gid2} ({len(f2)} 张)...")
    feats2 = load_face_feats(f2, app)

    # -------- 4. 计算平均特征 --------
    rep1 = avg(feats1)
    rep2 = avg(feats2)

    if rep1 is None or rep2 is None:
        sys.exit("无法比较：其中一个 GID 未提取到有效特征。")

    # -------- 5. 计算相似度 --------
    # 两个归一化向量的点积即为余弦相似度
    sim_score = float(np.dot(rep1, rep2))

    print("\n" + "=" * 40)
    print(f"比对结果: {gid1}  vs  {gid2}")
    print(f"人脸相似度 (Cosine): {sim_score:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    compare(GID_A, GID_B)
