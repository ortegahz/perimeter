#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据行人 ReID+Face 特征，把 SAVE_DIR 下的 GID 做离线聚类，
找出“同一人被分配到多个 gid”的情况。

示例：
    python tools/cluster_gid.py --save-dir /home/manu/tmp/perimeter \
                                --thr 0.25           # 余弦距离阈值(1-相似度)
    python tools/cluster_gid.py --dry-run            # 只打印建议
    python tools/cluster_gid.py --merge              # 自动把同簇 gid 合并
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from loguru import logger
from sklearn.cluster import AgglomerativeClustering

# ---------- 项目内模块 ----------
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.macros import DIR_REID_MODEL

# ---------------------------------

# ———————————————————————————— 常 量 ————————————————————————————
W_FACE, W_BODY = 0.6, 0.4  # 与主流程保持一致
FACE_MIN_DET_SCORE = 0.60  # 人脸置信筛选
FACE_BLUR_VAR = 100  # 人脸清晰度阈值
IMG_SIZE_REID = (128, 256)  # ReID 网络输入


# ————————————————————————————————————————————————————————————

@torch.inference_mode()
def face_embedding(searcher, img: np.ndarray) -> np.ndarray | None:
    """直接送入搜索器，不需要再次做人脸检测(目录里已是裁剪好的脸)。"""
    res = searcher.get(img)
    if len(res) != 1:
        return None
    f = res[0]
    if getattr(f, "det_score", 1.0) < FACE_MIN_DET_SCORE:
        return None
    if cv2.Laplacian(img, cv2.CV_64F).var() < FACE_BLUR_VAR:
        return None
    emb = f.embedding.astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


@torch.inference_mode()
def body_embedding(reid_model, img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, IMG_SIZE_REID)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(reid_model.device)
    feat = reid_model.model(tensor)
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def fuse_feat(face_f: np.ndarray | None, body_f: np.ndarray | None) -> np.ndarray:
    """face/body 均可能缺失，用零向量补齐后再加权归一化."""
    if face_f is None and body_f is None:
        raise RuntimeError("Both face & body feat are None")
    face_f = np.zeros(512, np.float32) if face_f is None else face_f * W_FACE
    body_f = np.zeros(2048, np.float32) if body_f is None else body_f * W_BODY
    combo = np.concatenate([face_f, body_f]).astype(np.float32)
    return combo / (np.linalg.norm(combo) + 1e-9)


def read_gid_features(gid_dir: Path,
                      face_app,
                      reid_model) -> tuple[np.ndarray | None, np.ndarray | None]:
    """读取某个 gid 目录下所有原型图片，返回平均人脸特征、人身特征."""
    face_feats, body_feats = [], []

    # -- faces --
    for img_path in sorted((gid_dir / "faces").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        f = face_embedding(face_app, img)
        if f is not None:
            face_feats.append(f)

    # -- bodies --
    for img_path in sorted((gid_dir / "bodies").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        b = body_embedding(reid_model, img)
        body_feats.append(b)

    face_vec = None if not face_feats else np.mean(np.stack(face_feats), 0)
    body_vec = None if not body_feats else np.mean(np.stack(body_feats), 0)
    if face_vec is not None:
        face_vec /= (np.linalg.norm(face_vec) + 1e-9)
    if body_vec is not None:
        body_vec /= (np.linalg.norm(body_vec) + 1e-9)
    return face_vec, body_vec


def cluster_gids(all_gids: List[str],
                 gid2feat: Dict[str, np.ndarray],
                 thr: float) -> Dict[int, List[str]]:
    """层次聚类，thr=1-cosine_sim."""
    feats = np.stack([gid2feat[g] for g in all_gids])
    # sklearn 的 AgglomerativeClustering 支持 precomputed 距离也支持直接做向量
    clr = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=thr,
        metric="cosine",
        linkage="average"
    )
    labels = clr.fit_predict(feats)
    clusters: Dict[int, List[str]] = {}
    for gid, lb in zip(all_gids, labels):
        clusters.setdefault(lb, []).append(gid)
    return clusters


def merge_gid_dirs(cluster: List[str], save_dir: Path, dry_run: bool):
    """把同一簇的 gid 目录合并到第一个 gid 中."""
    master = cluster[0]
    master_dir = save_dir / master
    for slave in cluster[1:]:
        slave_dir = save_dir / slave
        if not slave_dir.exists():
            continue
        logger.info(f"Merging {slave} -> {master}")
        for sub in ("faces", "bodies"):
            dst_sub = master_dir / sub
            dst_sub.mkdir(parents=True, exist_ok=True)
            for img in slave_dir.joinpath(sub).glob("*.jpg"):
                shutil.move(str(img), str(dst_sub / f"{slave}_{img.name}"))
        # 最后删空目录
        if not dry_run:
            shutil.rmtree(slave_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="/home/manu/tmp/perimeter_v0")
    parser.add_argument("--thr", type=float, default=0.6, help="聚类距离阈值(1-余弦相似度)")
    parser.add_argument("--dry-run", default=True, help="只打印建议，不做磁盘移动")
    parser.add_argument("--merge", action="store_true", help="执行目录合并(危险操作，先确保备份或加 --dry-run)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir).expanduser()
    if not save_dir.exists():
        logger.error(f"{save_dir} not found")
        sys.exit(1)

    # ---------------- Init 模型 ----------------
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    reid_model = PersonReid(DIR_REID_MODEL, which_epoch="last", gpu="0" if torch.cuda.is_available() else "")
    reid_model.model.eval()

    # ---------------- 收集特征 ------------------
    gid2feat = {}
    for gid_dir in sorted(save_dir.iterdir()):
        if not gid_dir.is_dir() or not gid_dir.name.startswith("G"):
            continue
        try:
            face_f, body_f = read_gid_features(gid_dir, face_app, reid_model)
            fused = fuse_feat(face_f, body_f)
            gid2feat[gid_dir.name] = fused
        except Exception as e:
            logger.warning(f"Skip {gid_dir.name}: {e}")

    all_gids = list(gid2feat.keys())
    if len(all_gids) < 2:
        logger.info("<=1 gid, nothing to cluster")
        return

    # ---------------- 聚类 ----------------------
    clusters = cluster_gids(all_gids, gid2feat, args.thr)
    logger.info(f"Total {len(all_gids)} gid → {len(clusters)} clusters (thr={args.thr})")

    # ---------------- 输出结果 ------------------
    for cid, gids in clusters.items():
        if len(gids) == 1:
            continue
        logger.info(f"[Cluster-{cid}] {', '.join(gids)}")

        if args.merge:
            if args.dry_run:
                logger.info("(dry-run) would merge above gids")
            else:
                merge_gid_dirs(gids, save_dir, dry_run=False)

    if args.merge and not args.dry_run:
        logger.info("Merge finished.")


if __name__ == "__main__":
    main()
