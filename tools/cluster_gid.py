#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据行人 ReID+Face 特征，把 SAVE_DIR 下的 GID 做离线聚类，
将同簇 GID 的数据复制到 OUT_DIR，原目录不动。

python tools/cluster_gid.py --save-dir /home/manu/tmp/perimeter \
                            --thr 0.25  \
                            --out-dir /home/manu/tmp/perimeter_merged \
                            --dry-run          # 只打印
python tools/cluster_gid.py --merge            # 真正写出
"""
from __future__ import annotations

import argparse
import cv2
import numpy as np
import shutil
import sys
import torch
from pathlib import Path
from typing import Dict, List

from loguru import logger
from sklearn.cluster import AgglomerativeClustering

# ---------- 项目内模块 ----------
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.macros import DIR_REID_MODEL

# ---------------------------------

# 常量
W_FACE, W_BODY = 0.6, 0.4
FACE_MIN_DET_SCORE, FACE_BLUR_VAR = 0.60, 100
IMG_SIZE_REID = (128, 256)


@torch.inference_mode()
def face_embedding(searcher, img: np.ndarray) -> np.ndarray | None:
    res = searcher.get(img)
    if len(res) != 1: return None
    f = res[0]
    if getattr(f, "det_score", 1.0) < FACE_MIN_DET_SCORE: return None
    if cv2.Laplacian(img, cv2.CV_64F).var() < FACE_BLUR_VAR: return None
    emb = f.embedding.astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


@torch.inference_mode()
def body_embedding(reid_model, img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, IMG_SIZE_REID)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(reid_model.device)
    feat = reid_model.model(tensor)
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def fuse_feat(face_f, body_f):
    if face_f is None and body_f is None:
        raise RuntimeError("Both face & body feat are None")
    face_f = np.zeros(512, np.float32) if face_f is None else face_f * W_FACE
    body_f = np.zeros(2048, np.float32) if body_f is None else body_f * W_BODY
    combo = np.concatenate([face_f, body_f]).astype(np.float32)
    return combo / (np.linalg.norm(combo) + 1e-9)


def read_gid_features(gid_dir: Path, face_app, reid_model):
    face_feats, body_feats = [], []
    for p in (gid_dir / "faces").glob("*.jpg"):
        f = face_embedding(face_app, cv2.imread(str(p)))
        if f is not None: face_feats.append(f)
    for p in (gid_dir / "bodies").glob("*.jpg"):
        body_feats.append(body_embedding(reid_model, cv2.imread(str(p))))
    face_vec = None if not face_feats else np.mean(np.stack(face_feats), 0)
    body_vec = None if not body_feats else np.mean(np.stack(body_feats), 0)
    if face_vec is not None: face_vec /= (np.linalg.norm(face_vec) + 1e-9)
    if body_vec is not None: body_vec /= (np.linalg.norm(body_vec) + 1e-9)
    return face_vec, body_vec


def cluster_gids(all_gids: List[str], gid2feat: Dict[str, np.ndarray], thr: float):
    feats = np.stack([gid2feat[g] for g in all_gids])
    clr = AgglomerativeClustering(
        n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average"
    )
    labels = clr.fit_predict(feats)
    clusters: Dict[int, List[str]] = {}
    for gid, lb in zip(all_gids, labels):
        clusters.setdefault(lb, []).append(gid)
    return clusters


def dump_cluster(cluster: List[str], save_dir: Path, out_dir: Path, dry_run: bool):
    master = cluster[0]  # 以第一个 gid 命名
    dst_master = out_dir / master
    for gid in cluster:
        for sub in ("faces", "bodies"):
            for img in (save_dir / gid / sub).glob("*.jpg"):
                dst = dst_master / sub / f"{gid}_{img.name}"
                if dry_run:
                    logger.info(f"(dry) copy {img}  ->  {dst}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="/home/manu/tmp/perimeter/")
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--out-dir", help="输出目录(默认 save_dir+'_merged')")
    ap.add_argument("--dry-run", default=False)
    ap.add_argument("--merge", default=True)
    args = ap.parse_args()

    save_dir = Path(args.save_dir).expanduser()
    if not save_dir.exists(): logger.error(f"{save_dir} not found"); sys.exit(1)

    out_dir = Path(args.out_dir or (str(save_dir) + "_merged")).expanduser()
    if not args.dry_run: out_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    reid_model = PersonReid(DIR_REID_MODEL, which_epoch="last", gpu="0" if torch.cuda.is_available() else "")
    reid_model.model.eval()

    # 收集特征
    gid2feat = {}
    for d in sorted(save_dir.iterdir()):
        if d.is_dir() and d.name.startswith("G"):
            try:
                fused = fuse_feat(*read_gid_features(d, face_app, reid_model))
                gid2feat[d.name] = fused
            except Exception as e:
                logger.warning(f"Skip {d}: {e}")
    all_gids = list(gid2feat.keys())
    if len(all_gids) < 2:
        logger.info("<=1 gid, nothing to cluster");
        return

    # 聚类
    clusters = cluster_gids(all_gids, gid2feat, args.thr)
    logger.info(f"Total {len(all_gids)} gid → {len(clusters)} clusters (thr={args.thr})")

    # 输出
    for cid, gids in clusters.items():
        if len(gids) == 1: continue
        logger.info(f"[Cluster-{cid}] {gids}")
        if args.merge:
            dump_cluster(gids, save_dir, out_dir, args.dry_run)

    if args.merge and not args.dry_run:
        logger.info(f"Finished. merged dir -> {out_dir}")


if __name__ == "__main__":
    main()
