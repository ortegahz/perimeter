#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.utils import face_align
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
    pa.add_argument("--folder", default="/home/manu/tmp/perimeter_v1/G00002/faces/")
    pa.add_argument("--provider", default="CUDAExecutionProvider",
                    choices=["CUDAExecutionProvider", "CPUExecutionProvider"])
    pa.add_argument("--det-size", type=int, default=640, help="SCRFD 输入尺寸")
    pa.add_argument("--outlier-thresh", type=float, default=1.2, help="异常值阈值（标准差倍数）")
    pa.add_argument("--output-json", default="/home/manu/tmp/embeddings_py.json", help="输出的人脸特征JSON文件路径")
    pa.add_argument("--output-aligned-dir", default="/home/manu/tmp/face_aligned_py")
    pa.add_argument("--output-detection-txt", default="/home/manu/tmp/detections_py.txt")
    # ======================= 【修改的部分在此】 =======================
    pa.add_argument("--output-embedding-txt", default="/home/manu/tmp/embeddings_py.txt",
                    help="输出的人脸特征TXT文件路径")
    # ======================= 【修改结束】 =======================
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

    # 准备保存对齐人脸的输出目录
    if args.output_aligned_dir:
        aligned_dir = Path(args.output_aligned_dir)
        aligned_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"对齐后的人脸将保存到: {aligned_dir}")
    else:
        aligned_dir = None

    # 准备保存检测结果的TXT文件
    detection_txt_file = None
    if args.output_detection_txt:
        try:
            detection_txt_file = open(args.output_detection_txt, 'w', encoding='utf-8')
            header = "filename,bbox_x,bbox_y,bbox_width,bbox_height,score,kps0_x,kps0_y,kps1_x,kps1_y,kps2_x,kps2_y,kps3_x,kps3_y,kps4_x,kps4_y\n"
            detection_txt_file.write(header)
            logger.info(f"人脸检测结果将保存到: {args.output_detection_txt}")
        except Exception as e:
            logger.error(f"无法打开或写入检测结果TXT文件: {e}")
            detection_txt_file = None

    # ======================= 【修改的部分在此】 =======================
    # 准备保存特征向量的TXT文件
    embedding_txt_file = None
    if args.output_embedding_txt:
        try:
            embedding_txt_file = open(args.output_embedding_txt, 'w', encoding='utf-8')
            logger.info(f"人脸特征向量将保存到: {args.output_embedding_txt}")
        except Exception as e:
            logger.error(f"无法打开或写入特征向量TXT文件: {e}")
            embedding_txt_file = None
    # ======================= 【修改结束】 =======================

    embeddings = []
    paths_list = []
    # ======================= 【修改的部分在此】 =======================
    # 对图像路径进行排序，确保处理顺序一致
    image_paths = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
    # ======================= 【修改结束】 =======================

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"读取失败: {img_path}")
            continue

        faces = face_app.get(img)
        if len(faces) == 0:
            logger.warning(f"未检测到人脸: {img_path}")
            continue

        for i, f in enumerate(faces):
            # ... (保存对齐图和检测结果的代码保持不变) ...
            if aligned_dir and hasattr(f, 'kps') and f.kps is not None:
                try:
                    aligned_img = face_align.norm_crop(img, landmark=f.kps)
                    arcface_template = np.array([
                        [38.2946, 51.6963], [73.5318, 51.5014],
                        [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]
                    ], dtype=np.float32)
                    for pt in arcface_template:
                        cv2.circle(aligned_img, tuple(pt.astype(int)), 2, (0, 255, 0), -1)
                    stem = img_path.stem
                    save_path = aligned_dir / f"{stem}_face{i}.jpg"
                    cv2.imwrite(str(save_path), aligned_img)
                except Exception as e:
                    logger.warning(f"保存对齐人脸失败 {img_path.name}: {e}")

            if detection_txt_file:
                bbox = f.bbox
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                kps_flat = f.kps.flatten()
                data_row = [
                               img_path.name,
                               f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}",
                               f"{f.det_score:.6f}",
                           ] + [f"{val:.4f}" for val in kps_flat]
                detection_txt_file.write(",".join(data_row) + "\n")

            if hasattr(f, "embedding") and f.embedding is not None:
                emb = f.embedding
                emb = emb / (np.linalg.norm(emb) + 1e-8)  # L2归一化
                embeddings.append(emb)
                paths_list.append(img_path.name)

                # ======================= 【修改的部分在此】 =======================
                # --- 保存特征向量到TXT文件 ---
                if embedding_txt_file:
                    face_id = f"{img_path.stem}_face{i}"
                    # 为了比对，使用足够高的精度
                    emb_str_list = [f"{val:.8f}" for val in emb]
                    line_to_write = f"{face_id},{','.join(emb_str_list)}\n"
                    embedding_txt_file.write(line_to_write)
                # ======================= 【修改结束】 =======================

            else:
                logger.warning(f"该检测结果无embedding信息: {img_path}")

    # 关闭文件
    if detection_txt_file:
        detection_txt_file.close()
    # ======================= 【修改的部分在此】 =======================
    if embedding_txt_file:
        embedding_txt_file.close()
    # ======================= 【修改结束】 =======================

    if not embeddings:
        logger.error("未获取到任何人脸特征")
        # 如果只做检测，没有特征也正常退出
        if args.output_detection_txt and not embeddings:
            logger.info("仅执行了人脸检测/特征提取，结果已保存。")
            return
        return

    # --- (后续的JSON保存和离群点分析代码保持不变) ---
    output_data = {path: emb.tolist() for path, emb in zip(paths_list, embeddings)}
    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"所有提取到的人脸特征已保存到: {args.output_json}")
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")

    embeddings = np.array(embeddings)
    logger.info(f"共获取 {len(embeddings)} 张人脸特征")

    # ... (后续代码不变) ...
    clean_embeddings, keep_mask = remove_outliers(embeddings, args.outlier_thresh)
    kept_paths = np.array(paths_list)[keep_mask].tolist()
    removed_paths = np.array(paths_list)[~keep_mask].tolist()

    logger.info(f"去除异常值后剩余 {len(clean_embeddings)} 张 (剔除 {len(embeddings) - len(clean_embeddings)})")
    if removed_paths:
        logger.info("被剔除的图片有：")
        for name in removed_paths:
            logger.info(f" - {name}")

    if not clean_embeddings.any():  # 检查数组是否为空或全为0
        logger.error("去除异常值后无剩余特征。")
        return

    mean_vec = clean_embeddings.mean(axis=0)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)

    logger.info("与均值向量相似度:")
    for name, emb in zip(kept_paths, clean_embeddings):
        sim = cosine_similarity(emb, mean_vec)
        logger.info(f"{name}  相似度: {sim:.4f}")

    logger.info("计算完成")


if __name__ == "__main__":
    main()
