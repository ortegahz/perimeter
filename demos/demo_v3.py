#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
周界安全 Demo :  “快显慢算”  +  tid|gid 同显（无 gid 时 -1）
Display 端通过 GStreamer 推 UDP，默认 udp://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import signal
import subprocess
import time
from collections import defaultdict, deque
from typing import Dict, List

import cv2
import numpy as np
import torch
from loguru import logger

# ---------- 业务内部模块 ----------
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.macros import DIR_REID_MODEL

# -----------------------------------

# ---------- 全局常量 ----------
SHOW_SCALE = 0.5
MAX_TID_GAP = 60
QUEUE_SIZE = 8
SENTINEL = None


# ---------------------------------

def safe_qsize(q):
    try:
        return q.qsize()
    except:
        return "NA"


# ======================================================
#   1. 覆盖型队列
# ======================================================
class LatestQueue(mpq.Queue):
    def __init__(self, maxsize=1, *, ctx=None):
        super().__init__(maxsize, ctx=ctx or mp.get_context())

    def put(self, obj, block=True, timeout=None):
        try:
            while True: self.get_nowait()
        except queue.Empty:
            pass
        super().put(obj, block, timeout)


# ======================================================
#   2. TrackAgg / GlobalID
# ======================================================
class TrackAgg:
    def __init__(self, max_body=10, max_face=3):
        self.body, self.face = deque(maxlen=max_body), deque(maxlen=max_face)
        self.last_fid = -1

    def add_body(self, f, s, fid):
        self.body.append((f, s))
        self.last_fid = fid

    def add_face(self, f, fid):
        self.face.append(f)
        self.last_fid = fid

    def body_feat(self):
        if not self.body: return None
        feats, scores = zip(*self.body)
        w = np.clip(np.float32(scores), 1e-2, None)
        w /= w.sum()
        rep = (np.stack(feats) * w[:, None]).sum(0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    def face_feat(self):
        if not self.face: return None
        rep = np.mean(np.stack(self.face), 0)
        return rep / (np.linalg.norm(rep) + 1e-9)


class GlobalID:
    def __init__(self, face_thr=.65, body_thr=.5, max_proto=5):
        self.face_thr, self.body_thr, self.max_proto = face_thr, body_thr, max_proto
        self.bank: Dict[str, Dict[str, List[np.ndarray]]] = {}

    @staticmethod
    def _sim(a, b):
        return float(a @ b)

    def _match(self, feat, key, thr):
        gid, best = None, -1.
        for g, pool in self.bank.items():
            for p in pool[key]:
                s = self._sim(feat, p)
                if s > best: gid, best = g, s
        return gid if best >= thr else None

    def _add(self, lst, feat):
        if len(lst) < self.max_proto:
            lst.append(feat)
        else:
            sims = [self._sim(feat, x) for x in lst]
            idx = int(np.argmax(sims))
            lst[idx] = 0.7 * lst[idx] + 0.3 * feat
            lst[idx] /= np.linalg.norm(lst[idx]) + 1e-9

    def _update(self, gid, face, body):
        if face is not None: self._add(self.bank[gid]['faces'], face)
        if body is not None: self._add(self.bank[gid]['bodies'], body)

    def assign(self, face, body):
        if face is not None:
            g = self._match(face, 'faces', self.face_thr)
            if g: self._update(g, face, body); return g
        if body is not None:
            g = self._match(body, 'bodies', self.body_thr)
            if g: self._update(g, face, body); return g
        gid = f"G{len(self.bank) + 1:05d}"
        self.bank[gid] = dict(faces=[], bodies=[])
        self._update(gid, face, body)
        logger.info(f"[GlobalID] new {gid}")
        return gid


# ======================================================
#   3. ReID 预处理
# ======================================================
@torch.inference_mode()
def prep_patch(patch: np.ndarray) -> torch.Tensor:
    im = cv2.resize(patch, (128, 256))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    im = ((im - mean) / std).astype(np.float32)  # 保证 float32
    return torch.from_numpy(im.transpose(2, 0, 1))


def normv(v): return v / (np.linalg.norm(v) + 1e-9)


# ======================================================
#   4. 进程函数
# ======================================================
def dec_det_proc(src, q_det2feat, q_det2disp, stop_evt, skip):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("[DecDet] open failed")
        q_det2feat.put(SENTINEL)
        q_det2disp.put(SENTINEL)
        return
    bt = ByteTrackPipeline(device="cuda")
    logger.info("[DecDet] ready")
    fid = 0
    while not stop_evt.is_set():
        ok, frm = cap.read()
        if not ok: break
        fid += 1
        if fid % skip: continue
        dets = bt.update(frm, debug=False)

        small = cv2.resize(frm, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
        q_det2disp.put((fid, small, dets))

        patches = []
        H, W = frm.shape[:2]
        for d in dets:
            x, y, w, h = d["tlwh"]
            x1, y1 = max(int(x), 0), max(int(y), 0)
            x2, y2 = min(int(x + w), W - 1), min(int(y + h), H - 1)
            patches.append(frm[y1:y2, x1:x2].copy())
        q_det2feat.put((fid, patches, dets))
    cap.release()
    q_det2feat.put(SENTINEL)
    q_det2disp.put(SENTINEL)
    logger.info("[DecDet] finished")


# ---- P2 Feature  -------------------------------------
def feature_proc(q_det2feat, q_map2disp, stop_evt):
    """
    1. batch 前向 reid.model()  → 512-d 特征
    2. 人脸特征 + body 特征融合给 gid_mgr
    3. 有人脸 或 ≥3 个 body 特征就立刻分配 gid
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reid = PersonReid(DIR_REID_MODEL, which_epoch="last",
                      gpu="0" if dev.type == "cuda" else "")
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    gid_mgr = GlobalID()

    agg_pool: dict[int, TrackAgg] = {}
    last_seen: defaultdict[int, int] = defaultdict(int)
    tid2gid: dict[int, str] = {}

    while not stop_evt.is_set():
        pkt = q_det2feat.get()
        if pkt is SENTINEL:
            break
        fid, patches, dets = pkt

        # ---------- 1) Body ReID ----------
        tensors, metas = [], []
        for det, patch in zip(dets, patches):
            if patch.size == 0:
                continue
            tensors.append(prep_patch(patch))
            metas.append((det["id"], det["score"]))

        if tensors:
            batch = torch.stack(tensors).to(dev).float()  # ★ 确保 float32
            with torch.no_grad():
                outputs = reid.model(batch)  # 直接前向
                outputs = torch.nn.functional.normalize(outputs, dim=1)
            feats = outputs.cpu().numpy()  # (N,512)

            for (tid, score), feat in zip(metas, feats):
                agg = agg_pool.setdefault(tid, TrackAgg())
                agg.add_body(feat, score, fid)
                last_seen[tid] = fid

        # ---------- 2) Face ----------
        for det, patch in zip(dets, patches):
            faces = face_app.get(patch)
            if faces:
                f = normv(faces[0].embedding.astype(np.float32))
                agg = agg_pool.setdefault(det["id"], TrackAgg())
                agg.add_face(f, fid)
                last_seen[det["id"]] = fid

        # ---------- 3) 尝试立即分配 gid ----------
        for tid, agg in agg_pool.items():
            if tid in tid2gid:
                continue
            if agg.face or len(agg.body) >= 8:
                gid = gid_mgr.assign(agg.face_feat(), agg.body_feat())
                tid2gid[tid] = gid
                q_map2disp.put((tid, gid))

        # ---------- 4) flush 消失轨迹 ----------
        for tid in list(last_seen.keys()):
            if fid - last_seen[tid] >= MAX_TID_GAP:
                agg = agg_pool.pop(tid)
                last_seen.pop(tid)
                gid = tid2gid.get(tid)
                if not gid:  # 之前还没分配
                    gid = gid_mgr.assign(agg.face_feat(), agg.body_feat())
                    tid2gid[tid] = gid
                    q_map2disp.put((tid, gid))

    q_map2disp.put(SENTINEL)
    logger.info("[Feature] finished")


def init_gst(W, H, fps, host, port):
    cmd = ["gst-launch-1.0", "-q",
           "fdsrc", "!", "videoparse", f"width={W}", f"height={H}", "format=bgr",
           f"framerate={int(fps)}/1", "!", "videoconvert", "!",
           "x264enc", "tune=zerolatency", "speed-preset=ultrafast", "!",
           "h264parse", "!", "mpegtsmux", "!",
           "udpsink", f"host={host}", f"port={port}", "sync=false", "async=false"]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def display_proc(q_det2disp, q_map2disp, stop_evt, host, port, fps_exp):
    logger.info(f"[Display] push → udp://{host}:{port}")
    gst = None
    first = True
    tid2gid = {}
    while not stop_evt.is_set():
        try:
            while True:
                m = q_map2disp.get_nowait()
                if m is SENTINEL: q_map2disp.put(SENTINEL); break
                tid2gid[m[0]] = m[1]
        except queue.Empty:
            pass
        pkt = q_det2disp.get()
        if pkt is SENTINEL: break
        fid, frame, dets = pkt

        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            gid = tid2gid.get(d["id"], -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['id']}|{gid}", (x, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if first:
            H, W = frame.shape[:2]
            gst = init_gst(W, H, fps_exp, host, port)
            first = False
            logger.info(f"[Display] GST ready ({W}x{H}@{fps_exp})")
        try:
            if gst and gst.poll() is None:
                gst.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError):
            logger.warning("[Display] GST pipe closed")
            gst = None

    if gst: gst.stdin.close(); gst.wait()
    logger.info("[Display] finished")


# ======================================================
#   5. Main
# ======================================================
def main():
    mp.set_start_method("spawn", force=True)
    pa = argparse.ArgumentParser()
    pa.add_argument("--video", default="rtsp://admin:1QAZ2wsx@172.20.20.64")
    pa.add_argument("--skip", type=int, default=1)
    pa.add_argument("--udp_host", default="127.0.0.1")
    pa.add_argument("--udp_port", type=int, default=5000)
    pa.add_argument("--fps", type=float, default=25.0)
    args = pa.parse_args()

    q_det2feat = LatestQueue(1)
    q_det2disp = LatestQueue(1)
    q_map2disp = mp.Queue(QUEUE_SIZE)
    stop_evt = mp.Event()

    procs = [
        mp.Process(target=dec_det_proc,
                   args=(args.video, q_det2feat, q_det2disp, stop_evt, args.skip),
                   name="DecDet"),
        mp.Process(target=feature_proc,
                   args=(q_det2feat, q_map2disp, stop_evt),
                   name="Feature"),
        mp.Process(target=display_proc,
                   args=(q_det2disp, q_map2disp, stop_evt,
                         args.udp_host, args.udp_port, args.fps),
                   name="Display")
    ]
    [p.start() for p in procs]

    signal.signal(signal.SIGINT, lambda s, f: stop_evt.set())
    try:
        while any(p.is_alive() for p in procs): time.sleep(.5)
    finally:
        stop_evt.set()
        for q in (q_det2feat, q_det2disp, q_map2disp):
            try:
                q.put_nowait(SENTINEL)
            except:
                pass
        [p.join() for p in procs]
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
