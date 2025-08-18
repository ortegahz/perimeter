from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from loguru import logger

# ------------ 内部模块 ------------
from utils_peri.general_funcs import make_dirs

# ---------------------------------


SHOW_SCALE = 0.5
SENTINEL = None
MIN_HW_RATIO = 1.5
MIN_BODY4GID = 8
MIN_FACE4GID = 8
W_FACE, W_BODY = 0.6, 0.4
MATCH_THR = 0.5
THR_NEW_GID = 0.3
FACE_DET_MIN_SCORE = 0.60
SAVE_DIR = "/home/manu/tmp/perimeter"
os.makedirs(SAVE_DIR, exist_ok=True)
make_dirs(SAVE_DIR, reset=True)

UPDATE_THR = 0.65
FACE_THR_STRICT = 0.5
BODY_THR_STRICT = 0.4
NEW_GID_MIN_FRAMES = 3
NEW_GID_TIME_WINDOW = 50
BIND_LOCK_FRAMES = 15
CANDIDATE_FRAMES = 2
MAX_TID_GAP = 256

GID_MAX_IDLE = 25 / 2 * 60 * 5
WAIT_FRAMES_AMBIGUOUS = 10


def is_long_patch(patch: np.ndarray, thr: float = MIN_HW_RATIO) -> bool:
    if patch is None or patch.size == 0: return False
    h, w = patch.shape[:2]
    return h / (w + 1e-9) >= thr


class TrackAgg:
    """
    聚合当前 track 在多帧内的 body/face 特征，并提供一致性检测。
    - body: 存 (feat, score, patch)
    - face: 存 (feat, patch)
    """

    def __init__(self, max_body=MIN_BODY4GID, max_face=MIN_BODY4GID):
        self.body: deque = deque(maxlen=max_body)
        self.face: deque = deque(maxlen=max_face)
        self.last_fid = -1

    def _main_representation(self, feats, patches, outlier_thr=1.5):
        if len(feats) == 0: return None, None
        arr = np.stack(feats)
        mean_f = arr.mean(axis=0)
        mean_f /= (np.linalg.norm(mean_f) + 1e-9)
        dists = np.linalg.norm(arr - mean_f, axis=1)
        keep = dists < (dists.mean() + outlier_thr * dists.std())
        kept_arr = arr[keep]
        kept_patches = [p for k, p in zip(keep, patches) if k]
        if kept_arr.shape[0] == 0:
            kept_arr = arr
            kept_patches = patches
        mean_f = kept_arr.mean(axis=0)
        mean_f /= (np.linalg.norm(mean_f) + 1e-9)
        sims = kept_arr @ mean_f
        idx = int(np.argmax(sims))
        return kept_arr[idx], kept_patches[idx]

    def main_body_feat_and_patch(self):
        if not self.body: return None, None
        feats, scores, patches = zip(*self.body)
        if not self._check_consistency(list(feats), thr=0.5):
            return None, None
        return self._main_representation(feats, patches, outlier_thr=1.5)

    def main_face_feat_and_patch(self):
        if not self.face: return None, None
        feats, patches = zip(*self.face)
        if not self._check_consistency(list(feats), thr=0.5):
            return None, None
        return self._main_representation(feats, patches, outlier_thr=1.5)

    def add_body(self, feat, scr, fid, patch):
        """新增一帧 body 特征"""
        if feat is None:
            return
        self.body.append((np.asarray(feat, dtype=np.float32), scr, patch))
        self.last_fid = fid

    def add_face(self, feat, fid, patch):
        """新增一帧 face 特征"""
        if feat is None:
            return
        self.face.append((np.asarray(feat, dtype=np.float32), patch))
        self.last_fid = fid

    @staticmethod
    def _check_consistency(feats, thr=0.35):
        """
        检查一组特征的一致性。
        如果平均 pairwise(1 - cos_sim) 大于 thr，则认为不一致。
        feats 必须是 L2-normalized。
        """
        if len(feats) < 2:
            return True  # 样本太少，默认通过
        sims = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                sims.append(float(feats[i] @ feats[j]))
        mean_diff = 1.0 - np.mean(sims)  # 平均 1 - cos_sim
        return mean_diff <= thr

    def body_feat(self):
        """
        返回聚合后的 body 特征; 如果一致性不过关或数量不足返回 None
        """
        if not self.body:
            return None
        feats, scores, _ = zip(*self.body)
        feats = list(feats)

        # 一致性检测
        if not self._check_consistency(feats, thr=0.5):
            return None

        # 置信度加权平均
        w = np.clip(np.float32(scores), 1e-2, None)
        w /= w.sum()
        rep = (np.stack(feats) * w[:, None]).sum(0)
        norm = np.linalg.norm(rep)
        if norm < 1e-9:  # 避免除 0
            return None
        return rep / norm

    def face_feat(self):
        """
        返回聚合后的 face 特征; 如果一致性不过关或数量不足返回 None
        """
        if not self.face:
            return None
        feats, _ = zip(*self.face)
        feats = list(feats)

        # 一致性检测（更严格）
        if not self._check_consistency(feats, thr=0.5):
            return None

        rep = np.mean(np.stack(feats, axis=0), axis=0)
        norm = np.linalg.norm(rep)
        if norm < 1e-9:
            return None
        return rep / norm

    def body_patches(self):
        """返回所有缓存的 body patch 图像"""
        return [p for *_r, p in self.body]

    def face_patches(self):
        """返回所有缓存的 face patch 图像"""
        return [p for _f, p in self.face]


@torch.inference_mode()
def prep_patch(patch: np.ndarray) -> torch.Tensor:
    im = cv2.resize(patch, (128, 256))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean, std = np.array([.485, .456, .406], dtype=np.float32), np.array([.229, .224, .225], dtype=np.float32)
    im = (im - mean) / std
    return torch.from_numpy(im.transpose(2, 0, 1))


def normv(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class GlobalID:
    def __init__(self, max_proto=8, w_face=0.6, w_body=0.4, thr=0.5, outlier_thresh=3.0):
        """
        max_proto: 每类特征最多保留的原型数量
        w_face, w_body: face/body 匹配权重
        thr: 匹配阈值
        outlier_thresh: 异常值剔除的Z分数阈值
        """
        self.max_proto = max_proto
        self.w_face = w_face
        self.w_body = w_body
        self.thr = thr
        self.outlier_thresh = outlier_thresh
        self.bank: Dict[str, Dict[str, List[np.ndarray]]] = {}  # {gid: {"faces": [], "bodies": []}}
        self.tid_hist: Dict[str, List[str]] = {}  # {gid: [tid1, tid2...]}
        self.last_update: Dict[str, int] = {}  # ==== 新增，记录每个gid的最近更新时间
        self.gid_next = 1  # <<< 新增唯一自增编号变量

    @staticmethod
    def _sim(a, b):
        return float(a @ b)

    @staticmethod
    def _avg(feats):
        rep = np.mean(np.stack(feats, axis=0), 0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    @staticmethod
    def remove_outliers(embeddings: list[np.ndarray], thresh: float = 1.2):
        if len(embeddings) < 3:
            return embeddings, [True] * len(embeddings)
        arr = np.stack(embeddings, axis=0)
        mean_vec = arr.mean(axis=0)
        dist = np.linalg.norm(arr - mean_vec, axis=1)
        z_scores = (dist - dist.mean()) / (dist.std() + 1e-8)
        keep_mask = np.abs(z_scores) < thresh
        new_list = [embeddings[i] for i in range(len(embeddings)) if keep_mask[i]]
        return new_list, keep_mask.tolist()

    def can_update_proto(self, gid, face_feat, body_feat):
        pool = self.bank[gid]
        if pool['faces'] and self._sim(face_feat, self._avg(pool['faces'])) < FACE_THR_STRICT:
            return -1
        if pool['bodies'] and self._sim(body_feat, self._avg(pool['bodies'])) < BODY_THR_STRICT:
            return -2
        return 0

    def _add(self, lst, feat, patch, dir_path):
        if feat is None or patch is None:
            return
        if lst and max(self._sim(feat, x) for x in lst) < UPDATE_THR:
            return
        if len(lst) < self.max_proto:
            idx = len(lst)
            lst.append(feat)
        else:
            idx = int(np.argmax([self._sim(feat, x) for x in lst]))
            lst[idx] = 0.7 * lst[idx] + 0.3 * feat
            lst[idx] /= np.linalg.norm(lst[idx]) + 1e-9

        # 保存对应的 patch 图片
        cv2.imwrite(os.path.join(dir_path, f"{idx:02d}.jpg"), patch)

        # 新增：做一次异常值剔除
        new_lst, keep_mask = self.remove_outliers(lst, self.outlier_thresh)
        if len(new_lst) != len(lst):
            logger.info(f"[GlobalID] Outlier removed: {len(lst) - len(new_lst)} from {dir_path}")
            lst.clear()
            lst.extend(new_lst)
            img_files = sorted(Path(dir_path).glob("*.jpg"))
            for i, img_path in enumerate(img_files):
                if i >= len(keep_mask) or not keep_mask[i]:
                    img_path.unlink(missing_ok=True)

    def _best_match(self, face, body):
        gid_best, score_best = None, -1.0
        for gid, pool in self.bank.items():
            if not pool['faces'] or not pool['bodies']:
                continue
            score = self.w_face * self._sim(face, self._avg(pool['faces'])) + \
                    self.w_body * self._sim(body, self._avg(pool['bodies']))
            if score > score_best:
                gid_best, score_best = gid, score
        return gid_best, score_best

    def probe(self, face, body):
        return self._best_match(face, body)

    def bind(self, gid, face, body, agg=None, tid=None, current_fid=None):
        root = os.path.join(SAVE_DIR, gid)
        if agg is not None:
            f_feat, f_patch = agg.main_face_feat_and_patch()
            b_feat, b_patch = agg.main_body_feat_and_patch()
        else:
            f_feat, f_patch = face, None
            b_feat, b_patch = body, None
        self._add(self.bank[gid]['faces'],
                  f_feat,
                  f_patch,
                  os.path.join(root, "faces"))
        self._add(self.bank[gid]['bodies'],
                  b_feat,
                  b_patch,
                  os.path.join(root, "bodies"))

        if tid:
            self.tid_hist.setdefault(gid, [])
            if tid not in self.tid_hist[gid]:
                self.tid_hist[gid].append(tid)

        if current_fid is not None:
            self.last_update[gid] = current_fid

    def new_gid(self):
        gid = f"G{self.gid_next:05d}"  # <<< 改这里，永远用gid_next，不重复
        self.gid_next += 1  # <<< 新建立即+1
        self.bank[gid] = dict(faces=[], bodies=[])
        self.tid_hist[gid] = []
        os.makedirs(os.path.join(SAVE_DIR, gid, "faces"), exist_ok=True)
        os.makedirs(os.path.join(SAVE_DIR, gid, "bodies"), exist_ok=True)
        logger.info(f"[GlobalID] new {gid}")
        return gid
