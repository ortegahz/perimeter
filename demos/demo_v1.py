#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Process 周界安全 Demo
-------------------------------------------------
P0 (Main)             : 创建队列/事件并拉起子进程
P1 Decoder            : 视频解码                       → q_dec2det
P2 Detector & Tracker : YOLO-X + BYTETracker           → q_det2feat
P3 Feature & Fusion   : ReID / Face / 全局身份归并      → q_feat2disp
P4 Streamer           : GStreamer UDP 推流
-------------------------------------------------
本版本嵌入详细日志，方便定位瓶颈：

    • 每进程打印 FPS / 单帧耗时
    • 打印上下游队列长度
    • put_nowait 失败会报“queue full”
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import subprocess
from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger
from yolox.utils.visualize import plot_tracking

# -------------------------------------------------- #
# 内部模块
# -------------------------------------------------- #
from cores.byteTrackPipeline import ByteTrackPipeline


# -------------------------------------------------- #
# ↓↓↓ 与原 demo 相同的辅助类
# -------------------------------------------------- #
class TrackFeatureAggregator:
    def __init__(self, maxlen: int = 10):
        from collections import deque
        self.body_feats: deque = deque(maxlen=maxlen)
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
    def __init__(self, face_thr=.45, body_thr=.27, max_proto=5):
        self.face_thr, self.body_thr, self.max_proto = face_thr, body_thr, max_proto
        self._store: Dict[str, Dict] = {}

    # 外部调用
    def match(self, face_feat, body_feat) -> str:
        if face_feat is None and body_feat is None:
            return self._new(None, None)
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
        return self._new(face_feat, body_feat)

    # ---------- 内部 ----------
    def _match(self, feat, key, thr):
        best_gid, best_sim = None, -1.0
        for gid, item in self._store.items():
            for p in item[key]:
                sim = float(feat @ p)
                if sim > best_sim:
                    best_gid, best_sim = gid, sim
        return best_gid if best_sim >= thr else None

    def _update(self, gid, face_feat, body_feat):
        entry = self._store[gid]
        if face_feat is not None:
            self._add_proto(entry["faces"], face_feat)
        if body_feat is not None:
            self._add_proto(entry["bodies"], body_feat)
        entry["last_seen"] = time.time()

    def _new(self, face_feat, body_feat) -> str:
        import uuid
        gid = uuid.uuid4().hex[:16]
        self._store[gid] = dict(faces=[], bodies=[], first_seen=time.time(), last_seen=time.time())
        self._update(gid, face_feat, body_feat)
        logger.info(f"[NEW] GlobalID={gid[:8]} created")
        return gid

    def _add_proto(self, protos, feat):
        if len(protos) < self.max_proto:
            protos.append(feat)
        else:
            sims = [float(feat @ p) for p in protos]
            idx = int(np.argmax(sims))
            protos[idx] = .7 * protos[idx] + .3 * feat
            protos[idx] /= np.linalg.norm(protos[idx]) + 1e-9


# -------------------------------------------------- #
# 公共参数
# -------------------------------------------------- #
SENTINEL = None  # 队列停止标记
QUEUE_SIZE = 32  # 每级队列长度


def safe_qsize(q: mp.Queue):
    try:
        return q.qsize()
    except (AttributeError, NotImplementedError):
        return "NA"


# -------------------------------------------------- #
# 进程 1 : Decoder
# -------------------------------------------------- #
def decoder_proc(src: str, q_out: mp.Queue, stop_evt: mp.Event,
                 skip: int = 1):
    cap = cv2.VideoCapture(src)
    fid, last = 0, time.perf_counter()
    logger.info("[Decoder] started.")
    while not stop_evt.is_set():
        ok, frm = cap.read()
        if not ok:
            break
        fid += 1
        if fid % skip:  # 主动丢帧
            continue

        try:
            q_out.put_nowait((fid, frm))
        except queue.Full:
            logger.warning("[Decoder] q_out full, dropping a frame")
            continue

        # 日志
        fps = 1.0 / (time.perf_counter() - last)
        last = time.perf_counter()
        logger.debug(f"[Decoder] Fid={fid:<6d}  {fps:5.1f} fps  "
                     f"q_out={safe_qsize(q_out)}")
    cap.release()
    q_out.put(SENTINEL)
    logger.info("[Decoder] finished.")


# -------------------------------------------------- #
# 进程 2 : Detector & Tracker
# -------------------------------------------------- #
import queue
import time


def det_proc(q_in: mp.Queue, q_out: mp.Queue, stop_evt: mp.Event):
    bt = ByteTrackPipeline(device="cuda")
    logger.info("[Detector] model ready.")

    while not stop_evt.is_set():
        loop_t0 = time.perf_counter()  # ———— 整个循环起点

        # ---------- 1. 等待上游帧 ----------
        t0 = time.perf_counter()
        item = q_in.get()
        t_wait_get = time.perf_counter() - t0

        if item is SENTINEL:
            break
        fid, frm = item

        # ---------- 2. 算法 ----------
        t1 = time.perf_counter()
        dets = bt.update(frm, debug=False)  # 内部已打印 det/track 细分
        t_alg = time.perf_counter() - t1

        # ---------- 3. 送下游 ----------
        t2 = time.perf_counter()
        try:
            q_out.put_nowait((fid, frm, dets))
            t_wait_put = time.perf_counter() - t2
        except queue.Full:
            t_wait_put = time.perf_counter() - t2
            logger.warning("[Detector] q_out full, dropping a frame")
            continue

        # ---------- 4. 统计 & 打印 ----------
        total = time.perf_counter() - loop_t0
        fps = 1.0 / total if total > 0 else float("inf")

        logger.debug(
            f"[Detector] Fid={fid:<6d} | "
            f"wait_get={t_wait_get * 1e3:5.1f} ms  "
            f"alg={t_alg * 1e3:5.1f} ms  "
            f"wait_put={t_wait_put * 1e3:5.1f} ms  "
            f"total={total * 1e3:5.1f} ms  "
            f"{fps:5.1f} fps | "
            f"q_in={safe_qsize(q_in)}  q_out={safe_qsize(q_out)}"
        )

    q_out.put(SENTINEL)
    logger.info("[Detector] finished.")


# -------------------------------------------------- #
# 进程 3 : Feature & Fusion   （此处只可视化，不做 ReID/FACE）
# -------------------------------------------------- #
def feature_proc(q_in: mp.Queue, q_out: mp.Queue, stop_evt: mp.Event):
    last = time.perf_counter()
    logger.info("[Feature] started.")

    while not stop_evt.is_set():
        pack = q_in.get()
        if pack is SENTINEL:
            break

        fid, frame, results = pack
        vis = plot_tracking(frame,
                            [tuple(r["tlwh"]) for r in results],
                            [r["id"] for r in results],
                            frame_id=fid)

        try:
            q_out.put_nowait((fid, vis))
        except queue.Full:
            logger.warning("[Feature] q_out full, dropping a frame")
            continue

        fps = 1.0 / (time.perf_counter() - last)
        last = time.perf_counter()
        logger.debug(f"[Feature]  Fid={fid:<6d}  {fps:5.1f} fps  "
                     f"q_in={safe_qsize(q_in)}  q_out={safe_qsize(q_out)}")

    q_out.put(SENTINEL)
    logger.info("[Feature] finished.")


# ------------------------------------------------------------
# 进程 4 : Display -> GStreamer UDP
# ------------------------------------------------------------
def init_gst(w, h, fps, host="127.0.0.1", port=5000):
    cmd = [
        "gst-launch-1.0", "-q",
        "fdsrc", f"blocksize={w * h * 3}", "!",
        "videoparse", f"width={w}", f"height={h}",
        "format=bgr", f"framerate={int(fps)}/1", "!",
        "videoconvert", "!",
        "queue", "leaky=downstream",
        "max-size-buffers=0", "max-size-bytes=0", "max-size-time=0", "!",
        "x264enc",
        "tune=zerolatency", "speed-preset=ultrafast",
        f"key-int-max={int(fps)}", "bframes=0", "byte-stream=true",
        "threads=1", "!",
        "h264parse", "!",
        "mpegtsmux", "alignment=7", "latency=0", "!",
        "udpsink", f"host={host}", f"port={port}",
        "sync=false", "async=false", "qos=false"
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def display_proc(q_in: mp.Queue, stop_evt: mp.Event,
                 host="127.0.0.1", port=5000, exp_fps=25.0):
    logger.info(f"[Display] started, push → udp://{host}:{port}")
    gst_proc, first = None, True
    last = time.perf_counter()

    while not stop_evt.is_set():
        itm = q_in.get()
        if itm is SENTINEL or itm is None:
            break
        fid, frame = itm

        if first:
            H, W = frame.shape[:2]
            gst_proc = init_gst(W, H, exp_fps, host, port)
            logger.info(f"[Display] GST ready  ({W}x{H}@{exp_fps})")
            first = False

        try:
            if gst_proc and gst_proc.poll() is None:
                gst_proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError):
            logger.warning("[Display] GST pipe closed")
            gst_proc = None

        fps = 1.0 / (time.perf_counter() - last)
        last = time.perf_counter()
        logger.debug(f"[Display] Fid={fid:<6d}  {fps:5.1f} fps  "
                     f"q_in={safe_qsize(q_in)}")

    if gst_proc:
        gst_proc.stdin.close()
        gst_proc.wait()
    logger.info("[Display] finished.")


# -------------------------------------------------- #
# 主程序
# -------------------------------------------------- #
def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",
                        default="rtsp://admin:1QAZ2wsx@172.20.20.64")
    parser.add_argument("--udp_host", default="127.0.0.1")
    parser.add_argument("--udp_port", type=int, default=5000)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--skip", type=int, default=1,
                        help="Decoder 跳帧间隔，1=不跳")
    args = parser.parse_args()

    # 队列 / 事件
    q_dec2det = mp.Queue(maxsize=1)
    q_det2feat = mp.Queue(maxsize=QUEUE_SIZE)
    q_feat2disp = mp.Queue(maxsize=QUEUE_SIZE)
    stop_evt = mp.Event()

    procs = [
        mp.Process(target=decoder_proc,
                   args=(args.video, q_dec2det, stop_evt, args.skip),
                   name="Decoder"),
        mp.Process(target=det_proc,
                   args=(q_dec2det, q_det2feat, stop_evt),
                   name="Detector"),
        mp.Process(target=feature_proc,
                   args=(q_det2feat, q_feat2disp, stop_evt),
                   name="Feature"),
        mp.Process(target=display_proc,
                   args=(q_feat2disp, stop_evt,
                         args.udp_host, args.udp_port, args.fps),
                   name="Display"),
    ]
    [p.start() for p in procs]

    def _sigint(sig, frm):
        logger.warning("SIGINT received → stopping ...")
        stop_evt.set()

    signal.signal(signal.SIGINT, _sigint)

    try:
        while any(p.is_alive() for p in procs):
            time.sleep(0.5)
    finally:
        stop_evt.set()
        for q in (q_dec2det, q_det2feat, q_feat2disp):
            try:
                q.put_nowait(SENTINEL)
            except Exception:
                pass
        [p.join() for p in procs]

    logger.info("All done.")


if __name__ == "__main__":
    main()
