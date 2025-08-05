#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Process 周界安全 Demo
-------------------------------------------------
P0 (Main)             : 负责创建队列、事件并拉起子进程
P1 Decoder            : 视频解码                       → q_dec2det
P2 Detector & Tracker : YOLO-11 + BYTETracker         → q_det2feat
P3 Feature & Fusion   : ReID / Face / 全局身份归并      → q_feat2disp
P4 Displayer          : OpenCV 实时显示（按 q/ESC 退出）
-------------------------------------------------
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import time
from typing import Dict, List, Optional

import cv2
import torch
from yolox.utils.visualize import plot_tracking

# -------------------------------------------------- #
# 你的内部模块（保持与原 demo 中的目录一致）
# -------------------------------------------------- #
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.macros import DIR_PERSON_REID


# -------------------------------------------------- #
# ↓↓↓ 从原 demo 中拷贝的两个辅助类
# -------------------------------------------------- #
class TrackFeatureAggregator:
    def __init__(self, maxlen: int = 10):
        from collections import deque
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


class GlobalIDManager:
    def __init__(self, face_thr: float = .45, body_thr: float = .27, max_proto: int = 5):
        self.face_thr, self.body_thr, self.max_proto = face_thr, body_thr, max_proto
        self._store: Dict[str, Dict] = {}

    # ------------ 对外 API ------------
    def match(self,
              face_feat: Optional[np.ndarray],
              body_feat: Optional[np.ndarray]) -> str:
        # 1. 都为空
        if face_feat is None and body_feat is None:
            return self._new(None, None)

        # 2. 先人脸，再身体
        if face_feat is not None:
            gid = self._match(face_feat, "faces", self.face_thr)
            if gid:
                self._update(gid, face_feat, body_feat)
                return gid
        if body_feat is not None:
            gid = self._match(body_feat, "bodies", self.body_thr)
            if gid:
                self._update(gid, face_feat, body_feat)
                return gid

        # 3. 新建
        return self._new(face_feat, body_feat)

    # ------------ 内部 ------------
    def _match(self, feat: np.ndarray, key: str, thr: float) -> Optional[str]:
        best_gid, best_sim = None, -1.0
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
        import uuid
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


# -------------------------------------------------- #
# 进程定义
# -------------------------------------------------- #
SENTINEL = None  # 队列停止标记
QUEUE_SIZE = 5  # 每级队列长度


def decoder_proc(src: str, q_out: mp.Queue, stop_evt: mp.Event,
                 skip: int = 1):
    cap = cv2.VideoCapture(src)
    fid = 0
    logger.info("[Decoder] started.")
    while not stop_evt.is_set():
        ok, frm = cap.read()
        if not ok:
            break
        fid += 1
        if fid % skip:
            continue
        q_out.put((fid, frm))
    cap.release()
    q_out.put(SENTINEL)  # 通知下游
    logger.info("[Decoder] finished.")


def det_proc(q_in: mp.Queue, q_out: mp.Queue, stop_evt: mp.Event):
    bt = ByteTrackPipeline(device="cuda")
    logger.info("[Detector] model ready.")
    while not stop_evt.is_set():
        item = q_in.get()
        if item is SENTINEL:
            break
        fid, frm = item
        dets = bt.update(frm, debug=False)  # List[dict]
        q_out.put((fid, frm, dets))
    q_out.put(SENTINEL)
    logger.info("[Detector] finished.")


def feature_proc(q_in: mp.Queue, q_out: mp.Queue, stop_evt: mp.Event,
                 reid_root: str, face_cache: str = "face_db_cache.pkl"):
    # ---------------- 初始化各种模型 ----------------
    reid = PersonReid(reid_root, which_epoch="last", gpu="0")
    face_app = FaceSearcher(provider="CPUExecutionProvider",
                            cache_path=face_cache).app
    gid_mgr = GlobalIDManager()
    trk_pool: Dict[int, TrackFeatureAggregator] = {}
    trk_meta = {}
    loiter_thr = 90  # 秒
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("[Feature] models ready.")

    def get_reid_feat(crop_bgr):
        if crop_bgr.size == 0:
            return None
        img = cv2.resize(crop_bgr, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        mean = np.array([.485, .456, .406], np.float32)
        std = np.array([.229, .224, .225], np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = reid.extract_feat(tensor)
        feat = feat.squeeze(0).cpu().numpy()
        feat /= np.linalg.norm(feat) + 1e-9
        return feat.astype(np.float32)

    frame_id = 0
    while not stop_evt.is_set():
        pack = q_in.get()
        if pack is SENTINEL:
            break
        fid, frame, results = pack
        frame_id = fid

        # ---------- 行人 ReID / 人脸 ----------
        for det in results:
            tid = det["id"]
            x, y, w, h = map(int, det["tlwh"])
            crop = frame[y:y + h, x:x + w]

            agg = trk_pool.setdefault(tid, TrackFeatureAggregator())

            body_feat = get_reid_feat(crop)
            if body_feat is not None:
                agg.add_body(body_feat, det["score"], frame_id)

            faces = face_app.get(crop)
            if faces:
                f = faces[0].embedding.astype(np.float32)
                f /= np.linalg.norm(f) + 1e-9
                agg.add_face(f, frame_id)

            if tid not in trk_meta:
                trk_meta[tid] = dict(enter_ts=time.time())

        # ---------- Track 回收 ----------
        ended = [t for t, ag in trk_pool.items() if ag.is_stale(frame_id)]
        for tid in ended:
            agg = trk_pool.pop(tid)
            gid = gid_mgr.match(agg.get_face_feature(), agg.get_body_feature())
            if tid in trk_meta:
                stay = time.time() - trk_meta[tid]["enter_ts"]
                if stay >= loiter_thr:
                    logger.warning(f"[ALERT] GlobalID {gid[:8]} loitering {stay:.1f}s")
                trk_meta.pop(tid)

        # ---------- 叠加可视化 ----------
        vis = plot_tracking(frame,
                            [tuple(r["tlwh"]) for r in results],
                            [r["id"] for r in results],
                            frame_id=fid)
        fps_txt = f"Fid:{fid}"
        cv2.putText(vis, fps_txt, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
        q_out.put((fid, vis))

    q_out.put(SENTINEL)
    logger.info("[Feature] finished.")


# ------------------------------------------------------------
# matplotlib 实时显示版本
# ------------------------------------------------------------
import matplotlib

matplotlib.use("TkAgg")  # 也可换成适合你平台的后端
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from loguru import logger

SENTINEL = None  # 与主程序保持一致


def display_proc(q_in: mp.Queue, stop_evt: mp.Event):
    """
    使用 matplotlib 实时显示，从队列 q_in 读取 (frame_id, bgr_img)。
    按 'q' / 'ESC' 或关闭窗体即可停止。
    """
    plt.ion()  # 打开交互模式

    fig, ax, im = None, None, None

    # ---------- 内部回调：按键 / 关闭 ----------
    def _on_key(event):
        if event.key in ('q', 'escape'):
            stop_evt.set()

    def _on_close(event):
        stop_evt.set()

    logger.info("[Display] started (matplotlib)...")

    while not stop_evt.is_set():
        itm = q_in.get()
        if itm is SENTINEL or itm is None:
            break

        fid, img_bgr = itm  # BGR → RGB
        img_rgb = img_bgr[..., ::-1]

        if fig is None:  # 第一次建立窗口
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(img_rgb)
            ax.set_title(f"Perimeter  |  Frame {fid}")
            ax.axis('off')
            fig.canvas.mpl_connect('key_press_event', _on_key)
            fig.canvas.mpl_connect('close_event', _on_close)
        else:  # 更新
            im.set_data(img_rgb)
            ax.set_title(f"Perimeter  |  Frame {fid}")

        fig.canvas.draw_idle()
        plt.pause(0.001)  # 必须要，有刷新作用

    plt.close('all')
    logger.info("[Display] finished.")


# -------------------------------------------------- #
# 主程序
# -------------------------------------------------- #
def main():
    mp.set_start_method('spawn', force=True)  # 跨平台安全

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",
                        default="rtsp://admin:1QAZ2wsx@172.20.20.64",
                        help="video path / rtsp url / webcam id")
    parser.add_argument("--reid_root",
                        default=os.path.join(DIR_PERSON_REID, "model/ft_ResNet50"))
    args = parser.parse_args()

    # --------- 队列 / 事件 ----------
    q_dec2det = mp.Queue(maxsize=QUEUE_SIZE)
    q_det2feat = mp.Queue(maxsize=QUEUE_SIZE)
    q_feat2disp = mp.Queue(maxsize=QUEUE_SIZE)
    stop_evt = mp.Event()

    # --------- 进程 ----------
    procs = [
        mp.Process(target=decoder_proc,
                   args=(args.video, q_dec2det, stop_evt),
                   name="Decoder"),
        mp.Process(target=det_proc,
                   args=(q_dec2det, q_det2feat, stop_evt),
                   name="Detector"),
        mp.Process(target=feature_proc,
                   args=(q_det2feat, q_feat2disp, stop_evt,
                         args.reid_root),
                   name="Feature"),
        mp.Process(target=display_proc,
                   args=(q_feat2disp, stop_evt),
                   name="Display"),
    ]
    for p in procs:
        p.start()

    # --------- Ctrl-C / SIGINT 处理 ----------
    def _sigint_handler(sig, frame):
        logger.warning("SIGINT received, stopping ...")
        stop_evt.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # --------- 等待 ----------
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(0.5)
    finally:
        stop_evt.set()
        # 主动往所有队列塞 SENTINEL，防止下游阻塞
        for q in (q_dec2det, q_det2feat, q_feat2disp):
            try:
                q.put_nowait(SENTINEL)
            except Exception:
                pass
        for p in procs:
            p.join()

    logger.info("All done.")


if __name__ == "__main__":
    main()
