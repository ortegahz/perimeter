#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
周界安全 Demo :  “快显慢算”  +  tid|gid 同显（无 gid 时 -1）

改动：
1) 只有同时具备人脸 + ≥16 张人体 patch 时才尝试比对 / 建立 GlobalID；
2) 全局比对方式： score = 0.6 * sim_face + 0.4 * sim_body，score ≥ 0.5 才视为命中。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import multiprocessing.queues as mpq
import os
import queue
import signal
import subprocess
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

# ------------ 业务内部模块 ------------
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.general_funcs import make_dirs
from utils_peri.macros import DIR_REID_MODEL

# -------------------------------------

# ------------- 常量 -------------------
SHOW_SCALE = 0.5
MAX_TID_GAP = 60
QUEUE_SIZE = 8
SENTINEL = None

MIN_HW_RATIO = 1.5  # 仅长条 patch 进入行人体管线
MIN_BODY4GID = 16  # 创建 / 比对前至少需要的 body patch 数

W_FACE, W_BODY = 0.6, 0.4  # 综合得分权重
MATCH_THR = 0.5  # 综合得分阈值

SAVE_DIR = "/home/manu/tmp/perimeter"
os.makedirs(SAVE_DIR, exist_ok=True)
make_dirs(SAVE_DIR, reset=True)


# --------------------------------------

def is_long_patch(patch: np.ndarray, thr: float = MIN_HW_RATIO) -> bool:
    if patch is None or patch.size == 0:
        return False
    h, w = patch.shape[:2]
    return h / (w + 1e-9) >= thr


def safe_qsize(q):
    try:
        return q.qsize()
    except Exception:
        return "NA"


# ======================================================
#   1. 覆盖型队列
# ======================================================
class LatestQueue(mpq.Queue):
    def __init__(self, maxsize=1, *, ctx=None):
        super().__init__(maxsize, ctx=ctx or mp.get_context())

    def put(self, obj, block=True, timeout=None):
        try:
            while True:
                self.get_nowait()
        except queue.Empty:
            pass
        super().put(obj, block, timeout)


# ======================================================
#   2. TrackAgg / GlobalID
# ======================================================
class TrackAgg:
    """
    聚合同一个 tid 的历史特征 & patch
    """

    def __init__(self, max_body=32, max_face=3):
        # (feat, score, patch)
        self.body: deque[Tuple[np.ndarray, float, np.ndarray]] = deque(maxlen=max_body)
        # (feat, patch)
        self.face: deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=max_face)
        self.last_fid = -1

    # --------- 添加 ----------
    def add_body(self, f: np.ndarray, s: float, fid: int, patch: np.ndarray):
        self.body.append((f, s, patch))
        self.last_fid = fid

    def add_face(self, f: np.ndarray, fid: int, patch: np.ndarray):
        self.face.append((f, patch))
        self.last_fid = fid

    # --------- 表征 ----------
    def body_feat(self):
        if not self.body:
            return None
        feats, scores, _ = zip(*self.body)
        w = np.clip(np.float32(scores), 1e-2, None)
        w /= w.sum()
        rep = (np.stack(feats) * w[:, None]).sum(0)
        rep /= np.linalg.norm(rep) + 1e-9
        return rep

    def face_feat(self):
        if not self.face:
            return None
        feats, _ = zip(*self.face)
        rep = np.mean(np.stack(feats, axis=0), 0)
        rep /= np.linalg.norm(rep) + 1e-9
        return rep

    # --------- Patch 列表 ----------
    def body_patches(self) -> List[np.ndarray]:
        return [p for *_r, p in self.body]

    def face_patches(self) -> List[np.ndarray]:
        return [p for _f, p in self.face]


# ======================================================
#   3. Patch 保存
# ======================================================
def save_patches(gid: str, agg: TrackAgg):
    gid_root = os.path.join(SAVE_DIR, gid)
    dir_body = os.path.join(gid_root, "bodies")
    dir_face = os.path.join(gid_root, "faces")
    os.makedirs(dir_body, exist_ok=True)
    os.makedirs(dir_face, exist_ok=True)

    idx_body = len(os.listdir(dir_body))
    idx_face = len(os.listdir(dir_face))

    for p in agg.body_patches():
        cv2.imwrite(os.path.join(dir_body, f"{idx_body:05d}.jpg"), p)
        idx_body += 1
    for p in agg.face_patches():
        cv2.imwrite(os.path.join(dir_face, f"{idx_face:05d}.jpg"), p)
        idx_face += 1


class GlobalID:
    """
    face + body 联合比对：score = 0.6 * sim_face + 0.4 * sim_body
    """

    def __init__(self,
                 max_proto=16,
                 w_face: float = W_FACE,
                 w_body: float = W_BODY,
                 thr: float = MATCH_THR):
        self.max_proto = max_proto
        self.w_face, self.w_body, self.thr = w_face, w_body, thr
        self.bank: Dict[str, Dict[str, List[np.ndarray]]] = {}

    # ----------- 工具 ------------
    @staticmethod
    def _sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b)

    def _avg(self, feats: List[np.ndarray]) -> np.ndarray:
        rep = np.mean(np.stack(feats, axis=0), axis=0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    def _add(self, lst: List[np.ndarray], feat: np.ndarray):
        """
        向原型池中添加新特征（类似 FIFO + 更新）
        """
        if len(lst) < self.max_proto:
            lst.append(feat)
        else:
            sims = [self._sim(feat, x) for x in lst]
            idx = int(np.argmax(sims))
            lst[idx] = 0.7 * lst[idx] + 0.3 * feat
            lst[idx] /= np.linalg.norm(lst[idx]) + 1e-9

    # ----------- 核心匹配 ------------
    def _match(self, face: np.ndarray, body: np.ndarray) -> Tuple[str | None, float]:
        gid_best, score_best = None, -1.0
        for gid, pool in self.bank.items():
            if not pool['faces'] or not pool['bodies']:
                continue  # 该 gid 两类特征不全，不比
            rep_face = self._avg(pool['faces'])
            rep_body = self._avg(pool['bodies'])
            sim_f = self._sim(face, rep_face)
            sim_b = self._sim(body, rep_body)

            score = self.w_face * sim_f + self.w_body * sim_b
            if score > score_best:
                gid_best, score_best = gid, score
        if score_best >= self.thr:
            return gid_best, score_best
        return None, score_best

    # ----------- 对外接口 ------------
    def assign(self,
               face: np.ndarray,
               body: np.ndarray,
               agg: TrackAgg | None = None) -> str:
        """
        face & body 均不为 None，才会被调用
        """
        gid, score = self._match(face, body)
        if gid:  # 命中
            self._update(gid, face, body)
            if agg:
                save_patches(gid, agg)
            logger.debug(f"[GlobalID] match {gid}  ({score:.3f})")
            return gid

        # 新建
        gid = f"G{len(self.bank) + 1:05d}"
        self.bank[gid] = dict(faces=[], bodies=[])
        self._update(gid, face, body)
        if agg:
            save_patches(gid, agg)
        logger.info(f"[GlobalID] new {gid}")
        return gid

    def _update(self, gid: str, face: np.ndarray | None, body: np.ndarray | None):
        if face is not None:
            self._add(self.bank[gid]['faces'], face)
        if body is not None:
            self._add(self.bank[gid]['bodies'], body)


# ======================================================
#   4. ReID 前处理
# ======================================================
@torch.inference_mode()
def prep_patch(patch: np.ndarray) -> torch.Tensor:
    im = cv2.resize(patch, (128, 256))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    im = ((im - mean) / std).astype(np.float32)
    return torch.from_numpy(im.transpose(2, 0, 1))


def normv(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ======================================================
#   5. 进程函数
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
        if not ok:
            break
        fid += 1
        if fid % skip:
            continue
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


# ------------ P2 Feature ------------------------------
def feature_proc(q_det2feat, q_map2disp, stop_evt):
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

        # 1) Body ReID ------------------------------------------------------
        tensors, metas, keep_patches = [], [], []
        for det, patch in zip(dets, patches):
            if not is_long_patch(patch):
                continue
            tensors.append(prep_patch(patch))
            metas.append((det["id"], det["score"]))
            keep_patches.append(patch)

        if tensors:
            batch = torch.stack(tensors).to(dev).float()
            with torch.no_grad():
                outputs = reid.model(batch)
                outputs = torch.nn.functional.normalize(outputs, dim=1)
            feats = outputs.cpu().numpy()

            for (tid, score), feat, img_patch in zip(metas, feats, keep_patches):
                agg = agg_pool.setdefault(tid, TrackAgg())
                agg.add_body(feat, score, fid, img_patch)
                last_seen[tid] = fid

        # 2) Face -----------------------------------------------------------
        for det, patch in zip(dets, patches):
            faces = face_app.get(patch)
            if not faces:
                continue
            f = normv(faces[0].embedding)
            tid = det["id"]
            agg = agg_pool.setdefault(tid, TrackAgg())
            agg.add_face(f, fid, patch)
            last_seen[tid] = fid

        # 3) 尝试分配 GlobalID ----------------------------------------------
        for tid, agg in list(agg_pool.items()):
            if tid in tid2gid:
                continue  # 已分配
            if len(agg.body) < MIN_BODY4GID:
                continue  # body 不足
            face_feat = agg.face_feat()
            if face_feat is None:
                continue  # 无人脸
            gid = gid_mgr.assign(face_feat, agg.body_feat(), agg)
            tid2gid[tid] = gid
            q_map2disp.put((tid, gid))

        # 4) Flush 过期轨迹 --------------------------------------------------
        for tid in list(last_seen.keys()):
            if fid - last_seen[tid] >= MAX_TID_GAP:
                agg = agg_pool.pop(tid)
                last_seen.pop(tid)
                if tid in tid2gid:  # 已分配，直接丢弃
                    continue
                if len(agg.body) < MIN_BODY4GID or agg.face_feat() is None:
                    continue
                gid = gid_mgr.assign(agg.face_feat(), agg.body_feat(), agg)
                tid2gid[tid] = gid
                q_map2disp.put((tid, gid))

    q_map2disp.put(SENTINEL)
    logger.info("[Feature] finished")


# ------------ P3 Display ------------------------------
def init_gst(W, H, fps, host, port):
    cmd = ["gst-launch-1.0", "-q",
           "fdsrc", "!", "videoparse",
           f"width={W}", f"height={H}", "format=bgr",
           f"framerate={int(fps)}/1", "!",
           "videoconvert", "!",
           "x264enc", "tune=zerolatency", "speed-preset=ultrafast", "!",
           "h264parse", "!", "mpegtsmux", "!",
           "udpsink", f"host={host}", f"port={port}",
           "sync=false", "async=false"]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def display_proc(q_det2disp, q_map2disp, stop_evt, host, port, fps_exp):
    logger.info(f"[Display] push → udp://{host}:{port}")
    gst = None
    first = True
    tid2gid = {}
    while not stop_evt.is_set():
        # 更新 tid → gid 映射
        try:
            while True:
                m = q_map2disp.get_nowait()
                if m is SENTINEL:
                    q_map2disp.put(SENTINEL)
                    break
                tid2gid[m[0]] = m[1]
        except queue.Empty:
            pass

        pkt = q_det2disp.get()
        if pkt is SENTINEL:
            break
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

    if gst:
        gst.stdin.close()
        gst.wait()
    logger.info("[Display] finished")


# ======================================================
#   6. Main
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
                   args=(args.video, q_det2feat, q_det2disp,
                         stop_evt, args.skip),
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
        while any(p.is_alive() for p in procs):
            time.sleep(.5)
    finally:
        stop_evt.set()
        for q in (q_det2feat, q_det2disp, q_map2disp):
            try:
                q.put_nowait(SENTINEL)
            except Exception:
                pass
        [p.join() for p in procs]
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
