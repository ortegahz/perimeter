#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较两个 gid 目录的相似度（写死 gid，不接收命令行参数）
"""

import os
import sys
from glob import glob
from typing import List, Optional

import cv2
import numpy as np
import torch

from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.macros import DIR_REID_MODEL  # 根据实际情况修改

# =========================================================
#              在这里改要比对的两个 gid
# =========================================================
GID_A = "G00004"
GID_B = "G00016"
# =========================================================

SAVE_DIR = "/home/manu/tmp/perimeter"  # 与主程序一致
W_FACE, W_BODY = 0.6, 0.4  # 加权方式保持一致


def norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------- 兼容不同版本 insightface ----------
def safe_face_get(face_app, img):
    try:
        return face_app.get(img, max_size=640, detect=False)
    except TypeError:
        try:
            return face_app.get(img, max_size=640)
        except TypeError:
            return face_app.get(img)


# --------------------------------------------

def load_face_feats(paths: List[str], face_app) -> List[np.ndarray]:
    feats = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        faces = safe_face_get(face_app, img)
        if not faces:
            continue
        feats.append(norm(faces[0].embedding))
    return feats


@torch.inference_mode()
def load_body_feats(paths: List[str], reid) -> List[np.ndarray]:
    feats = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        im = cv2.resize(img, (128, 256))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        mean = np.array([.485, .456, .406], dtype=np.float32)
        std = np.array([.229, .224, .225], dtype=np.float32)
        im = ((im - mean) / std)[None].transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(im).float().to(reid.device)
        feat = reid.model(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1).cpu().numpy()[0]
        feats.append(feat)
    return feats


def avg(feats: List[np.ndarray]) -> Optional[np.ndarray]:
    if not feats:
        return None
    return norm(np.mean(np.stack(feats, 0), 0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def compare(gid1: str, gid2: str):
    d1, d2 = os.path.join(SAVE_DIR, gid1), os.path.join(SAVE_DIR, gid2)
    if not (os.path.isdir(d1) and os.path.isdir(d2)):
        sys.exit("gid 目录不存在")

    # -------- 模型 --------
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    reid = PersonReid(DIR_REID_MODEL, which_epoch="last", gpu="0")

    # -------- 读取 patch --
    f1 = sorted(glob(os.path.join(d1, "faces", "*.jpg")))
    b1 = sorted(glob(os.path.join(d1, "bodies", "*.jpg")))
    f2 = sorted(glob(os.path.join(d2, "faces", "*.jpg")))
    b2 = sorted(glob(os.path.join(d2, "bodies", "*.jpg")))

    face1 = load_face_feats(f1, face_app)
    face2 = load_face_feats(f2, face_app)
    body1 = load_body_feats(b1, reid)
    body2 = load_body_feats(b2, reid)

    rep_f1, rep_f2 = avg(face1), avg(face2)
    rep_b1, rep_b2 = avg(body1), avg(body2)

    # 把旧的这一行
    # if None in (rep_f1, rep_f2, rep_b1, rep_b2):

    # 改成（任选其一）
    if any(x is None for x in (rep_f1, rep_f2, rep_b1, rep_b2)):
        sys.exit("两个 gid 的人脸 / 人体 patch 不够，无法比较")

    sim_face = cosine(rep_f1, rep_f2)
    sim_body = cosine(rep_b1, rep_b2)
    score = W_FACE * sim_face + W_BODY * sim_body

    print(f"gid {gid1}  vs  {gid2}")
    print(f"   face_sim  = {sim_face:.3f}")
    print(f"   body_sim  = {sim_body:.3f}")
    print(f" => final_score = {score:.3f}")


if __name__ == "__main__":
    compare(GID_A, GID_B)
