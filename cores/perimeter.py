#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周界安全 Demo
-------------------------------------------------
1. ByteTrackPipeline            : 目标检测 + 跟踪
2. PersonReid                   : 行人 ReID
3. FaceSearcher.FaceAnalysis    : 人脸特征
4. GlobalIDManager              : 人脸优先、ReID 兜底的身份归并
5. matplotlib 实时可视化        : 代替 cv2.imshow
"""

from __future__ import annotations

import os
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
plt.ion()

import cv2
# ---------- matplotlib 设置 ----------
import numpy as np
import torch
from loguru import logger
from yolox.utils.visualize import plot_tracking

# ---------- 你的内部模块 ----------
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.general_funcs import imshow_plt
from utils_peri.macros import DIR_PERSON_REID

# --------------------------------------------------
# 全局常量
# --------------------------------------------------
DIR_REID_MODEL = os.path.join(DIR_PERSON_REID, "model/ft_ResNet50")


# --------------------------------------------------
# TrackFeatureAggregator
# --------------------------------------------------
class TrackFeatureAggregator:
    def __init__(self, maxlen: int = 10):
        self.body_feats: deque = deque(maxlen=maxlen)  # [(feat, score)]
        self.face_feats: deque = deque(maxlen=3)
        self.last_update_frame: int = -1

    def add_body(self, feat: np.ndarray, q: float, frame_id: int):
        self.body_feats.append((feat, q))
        self.last_update_frame = frame_id

    def add_face(self, feat: np.ndarray, frame_id: int):
        self.face_feats.append(feat)
        self.last_update_frame = frame_id

    def get_body_feature(self) -> Optional[np.ndarray]:
        if not self.body_feats:
            return None
        feats, scores = zip(*self.body_feats)
        w = np.clip(np.asarray(scores, np.float32), 1e-2, None)
        w /= w.sum()
        rep = (np.stack(feats) * w[:, None]).sum(0)
        rep /= np.linalg.norm(rep) + 1e-9
        return rep.astype(np.float32)

    def get_face_feature(self) -> Optional[np.ndarray]:
        if not self.face_feats:
            return None
        rep = np.mean(np.stack(self.face_feats), axis=0)
        rep /= np.linalg.norm(rep) + 1e-9
        return rep.astype(np.float32)

    def is_stale(self, cur_frame: int, max_gap: int = 60) -> bool:
        return (cur_frame - self.last_update_frame) >= max_gap


# --------------------------------------------------
# GlobalIDManager
# --------------------------------------------------
class GlobalIDManager:
    def __init__(self, face_thr: float = .45, body_thr: float = .27, max_proto: int = 5):
        self.face_thr, self.body_thr, self.max_proto = face_thr, body_thr, max_proto
        self._store: Dict[str, Dict] = {}

    def match(self,
              face_feat: Optional[np.ndarray],
              body_feat: Optional[np.ndarray]) -> str:
        """
        返回匹配到的 global_id；如都匹配不到则新建
        """
        # --------- 1. 无有效特征直接新建 ----------
        if face_feat is None and body_feat is None:
            return self._new(None, None)

        # --------- 2. 先试人脸 ----------
        if face_feat is not None:
            gid = self._match(face_feat, "faces", self.face_thr)
            if gid:
                self._update(gid, face_feat, body_feat)
                return gid

        # --------- 3. 再试身体 ----------
        if body_feat is not None:
            gid = self._match(body_feat, "bodies", self.body_thr)
            if gid:
                self._update(gid, face_feat, body_feat)
                return gid

        # --------- 4. 都没命中 → 新建 ----------
        return self._new(face_feat, body_feat)

    # ---------- 内部 ----------
    def _match(self, feat: np.ndarray, key: str, thr: float) -> Optional[str]:
        best_gid, best_sim = None, -1
        for gid, item in self._store.items():
            for p in item[key]:
                sim = float(feat @ p)
                if sim > best_sim:
                    best_gid, best_sim = gid, sim
        return best_gid if best_sim >= thr else None

    def _update(self, gid: str,
                face_feat: Optional[np.ndarray],
                body_feat: Optional[np.ndarray]):
        entry = self._store[gid]
        if face_feat is not None:
            self._add_proto(entry["faces"], face_feat)
        if body_feat is not None:
            self._add_proto(entry["bodies"], body_feat)
        entry["last_seen"] = time.time()

    def _new(self,
             face_feat: Optional[np.ndarray],
             body_feat: Optional[np.ndarray]) -> str:
        gid = uuid.uuid4().hex[:16]
        self._store[gid] = dict(faces=[], bodies=[], first_seen=time.time(), last_seen=time.time())
        self._update(gid, face_feat, body_feat)
        logger.info(f"[NEW] GlobalID={gid[:8]} created")
        return gid

    def _add_proto(self, protos: List[np.ndarray], feat: np.ndarray):
        if len(protos) < self.max_proto:
            protos.append(feat)
        else:
            sims = [float(feat @ p) for p in protos]
            idx = int(np.argmax(sims))
            protos[idx] = .7 * protos[idx] + .3 * feat
            protos[idx] /= np.linalg.norm(protos[idx]) + 1e-9


# --------------------------------------------------
# PerimeterExecutor
# --------------------------------------------------
class PerimeterExecutor:
    def __init__(self,
                 bytetrack_pipe: ByteTrackPipeline,
                 reid_model_dir: str,
                 face_cache: str):

        self.bt_pipe = bytetrack_pipe
        self.reid = PersonReid(reid_model_dir, which_epoch="last", gpu="0")
        self.face_app = FaceSearcher(provider="CPUExecutionProvider",
                                     cache_path=face_cache).app
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gid_mgr = GlobalIDManager()
        self.trk_pool: Dict[int, TrackFeatureAggregator] = {}
        self.trk_meta = {}
        self.loiter_thr = 90

        self._plt_handle = None  # matplotlib 句柄

    # -------- 行人 ReID 预处理 + 前向 ----------
    def _get_reid_feat(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr.size == 0:
            return None
        img = cv2.resize(crop_bgr, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.reid.extract_feat(tensor)
        feat = feat.squeeze(0).cpu().numpy()
        feat /= np.linalg.norm(feat) + 1e-9
        return feat.astype(np.float32)

    def _inside_roi(self, x: float, y: float) -> bool:
        return True  # TODO: 根据实际 ROI 修改

    # 其他 import 与类成员保持不变 …

    # ------------------ 主循环 ------------------
    def run(self, video_src: str):
        cap = cv2.VideoCapture(video_src)
        frame_id = 0
        n_skip = 8

        while True:
            t_loop0 = time.perf_counter()  # ===== 整帧起点 =====

            # ---------- 0. 读帧 ----------
            t_read0 = time.perf_counter()
            ok, frame = cap.read()
            t_read = time.perf_counter() - t_read0  # 读帧耗时
            if not ok:
                break

            frame_id += 1
            if frame_id % n_skip != 0:
                continue

            # ---------- 1. ByteTrack ----------
            t_bt0 = time.perf_counter()
            results = self.bt_pipe.update(frame)
            t_bt = time.perf_counter() - t_bt0

            # ---------- 2. 特征抽取 ----------
            body_time = 0.0
            face_time = 0.0
            for det in results:
                tid = det["id"]
                x, y, w, h = map(int, det["tlwh"])
                agg = self.trk_pool.setdefault(tid, TrackFeatureAggregator())

                # 2-A 行人特征
                t0 = time.perf_counter()
                crop = frame[y:y + h, x:x + w]
                body_feat = self._get_reid_feat(crop)
                body_time += time.perf_counter() - t0
                if body_feat is None:
                    continue
                agg.add_body(body_feat, det["score"], frame_id)

                # 2-B 人脸特征
                t0 = time.perf_counter()
                faces = self.face_app.get(crop)
                face_time += time.perf_counter() - t0
                if faces:
                    f = faces[0].embedding.astype(np.float32)
                    f /= np.linalg.norm(f) + 1e-9
                    agg.add_face(f, frame_id)

                if tid not in self.trk_meta and self._inside_roi(x + w / 2, y + h / 2):
                    self.trk_meta[tid] = dict(enter_ts=time.time())

            # ---------- 3. Track 回收 ----------
            t_flush0 = time.perf_counter()
            self._flush(frame_id)
            t_flush = time.perf_counter() - t_flush0

            # ---------- 4. 可视化 ----------
            t_vis0 = time.perf_counter()
            vis = plot_tracking(frame,
                                [tuple(r["tlwh"]) for r in results],
                                [r["id"] for r in results],
                                frame_id=frame_id)
            self._plt_handle, quit_flag = imshow_plt(vis, self._plt_handle)
            t_vis = time.perf_counter() - t_vis0
            if quit_flag:
                break

            # ---------- 5. 打印耗时 ----------
            t_total = time.perf_counter() - t_loop0
            logger.info(
                f"[Frm {frame_id:>5}] "
                f"Read={t_read * 1e3:6.1f}ms | "
                f"Bt={t_bt * 1e3:6.1f}ms | "
                f"BodyFeat={body_time * 1e3:6.1f}ms | "
                f"FaceFeat={face_time * 1e3:6.1f}ms | "
                f"Flush={t_flush * 1e3:6.1f}ms | "
                f"Vis={t_vis * 1e3:6.1f}ms | "
                f"Tot={t_total * 1e3:6.1f}ms"
            )

        cap.release()
        plt.close("all")

    # ---------- Track 结束时注册 / 徘徊 ----------
    def _flush(self, cur_frame: int):
        ended = [tid for tid, agg in self.trk_pool.items() if agg.is_stale(cur_frame)]
        for tid in ended:
            agg = self.trk_pool.pop(tid)
            gid = self.gid_mgr.match(agg.get_face_feature(), agg.get_body_feature())
            logger.info(f"Track {tid}  →  GlobalID {gid[:8]}")

            if tid in self.trk_meta:
                stay = time.time() - self.trk_meta[tid]["enter_ts"]
                if stay >= self.loiter_thr:
                    logger.warning(f"[ALERT] GlobalID {gid[:8]} loitering {stay:.1f}s")
                self.trk_meta.pop(tid)
