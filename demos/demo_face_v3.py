#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from cores.faceSearcher import FaceSearcher  # 你已有的类


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)


def remove_outliers(embeddings: np.ndarray, thresh: float = 2.0):
    """
    用标准差方法去掉特征异常值
    返回: (clean_embeddings, keep_mask)
    """
    mean_vec = embeddings.mean(axis=0)
    dist = np.linalg.norm(embeddings - mean_vec, axis=1)
    z_scores = (dist - dist.mean()) / (dist.std() + 1e-8)
    keep_mask = np.abs(z_scores) < thresh
    return embeddings[keep_mask], keep_mask


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--folder", default="/home/manu/tmp/perimeter/G00010/faces/")
    pa.add_argument("--provider", default="CUDAExecutionProvider",
                    choices=["CUDAExecutionProvider", "CPUExecutionProvider"])
    pa.add_argument("--det-size", type=int, default=640, help="SCRFD 输入尺寸")
    pa.add_argument("--outlier-thresh", type=float, default=1.2, help="异常值阈值（标准差倍数）")
    args = pa.parse_args()

    # 初始化模型
    fs = FaceSearcher(provider=args.provider)
    face_app = fs.app
    ctx_id = 0 if args.provider.startswith("CUDA") else -1
    face_app.prepare(ctx_id=ctx_id,
                     det_thresh=0.5,
                     det_size=(args.det_size, args.det_size))
    logger.info(f"Face detector ready  (provider={args.provider}, ctx_id={ctx_id})")

    folder = Path(args.folder)
    if not folder.exists():
        logger.error(f"文件夹不存在: {folder}")
        return

    embeddings = []
    paths_list = []  # 保存每个embedding对应的图片路径
    image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"读取失败: {img_path}")
            continue

        faces = face_app.get(img)
        if len(faces) == 0:
            logger.warning(f"未检测到人脸: {img_path}")
            continue

        for f in faces:
            if hasattr(f, "embedding") and f.embedding is not None:
                emb = f.embedding
                emb = emb / (np.linalg.norm(emb) + 1e-8)  # L2归一化
                embeddings.append(emb)
                paths_list.append(img_path.name)  # 保存文件名，而不是全路径
            else:
                logger.warning(f"该检测结果无embedding信息: {img_path}")

    if not embeddings:
        logger.error("未获取到任何人脸特征")
        return

    embeddings = np.array(embeddings)
    logger.info(f"共获取 {len(embeddings)} 张人脸特征")

    # 去掉异常值
    clean_embeddings, keep_mask = remove_outliers(embeddings, args.outlier_thresh)
    kept_paths = np.array(paths_list)[keep_mask].tolist()
    removed_paths = np.array(paths_list)[~keep_mask].tolist()  # 被剔除的图片

    logger.info(f"去除异常值后剩余 {len(clean_embeddings)} 张 (剔除 {len(embeddings) - len(clean_embeddings)})")
    if removed_paths:
        logger.info("被剔除的图片有：")
        for name in removed_paths:
            logger.info(f" - {name}")

    # 计算均值向量
    mean_vec = clean_embeddings.mean(axis=0)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)

    # 计算每张人脸与均值的相似度
    logger.info("与均值向量相似度:")
    for name, emb in zip(kept_paths, clean_embeddings):
        sim = cosine_similarity(emb, mean_vec)
        logger.info(f"{name}  相似度: {sim:.4f}")

    logger.info("计算完成")


if __name__ == "__main__":
    main()
