#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双路视频 + 全局GID
- 使用严格候选检测、多帧确认、防抖&锁定机制
- 第一个GID：有人脸+body -> 立即创建
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
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

# ------------ 内部模块 ------------
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.general_funcs import make_dirs
from utils_peri.macros import DIR_REID_MODEL

# ---------------------------------

SHOW_SCALE = 0.5
SENTINEL = None
MIN_HW_RATIO = 1.5
MIN_BODY4GID = 8
W_FACE, W_BODY = 0.6, 0.4
MATCH_THR = 0.5
THR_NEW_GID = 0.3
FACE_DET_MIN_SCORE = 0.60
SAVE_DIR = "/home/manu/tmp/perimeter"
os.makedirs(SAVE_DIR, exist_ok=True)
make_dirs(SAVE_DIR, reset=True)

UPDATE_THR = 0.65
FACE_THR_STRICT = 0.6
BODY_THR_STRICT = 0.55
NEW_GID_MIN_FRAMES = 3
NEW_GID_TIME_WINDOW = 50
BIND_LOCK_FRAMES = 15
CANDIDATE_FRAMES = 2
MAX_TID_GAP = 60


def is_long_patch(patch: np.ndarray, thr: float = MIN_HW_RATIO) -> bool:
    if patch is None or patch.size == 0: return False
    h, w = patch.shape[:2]
    return h / (w + 1e-9) >= thr


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


class TrackAgg:
    def __init__(self, max_body=MIN_BODY4GID, max_face=MIN_BODY4GID):
        self.body: deque = deque(maxlen=max_body)
        self.face: deque = deque(maxlen=max_face)
        self.last_fid = -1

    def add_body(self, feat, scr, fid, patch):
        self.body.append((feat, scr, patch))
        self.last_fid = fid

    def add_face(self, feat, fid, patch):
        self.face.append((feat, patch))
        self.last_fid = fid

    def body_feat(self):
        if not self.body: return None
        feats, scores, _ = zip(*self.body)
        w = np.clip(np.float32(scores), 1e-2, None)
        w /= w.sum()
        rep = (np.stack(feats) * w[:, None]).sum(0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    def face_feat(self):
        if not self.face: return None
        feats, _ = zip(*self.face)
        rep = np.mean(np.stack(feats, axis=0), 0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    def body_patches(self):
        return [p for *_r, p in self.body]

    def face_patches(self):
        return [p for _f, p in self.face]


class GlobalID:
    def __init__(self, max_proto=8, w_face=W_FACE, w_body=W_BODY, thr=MATCH_THR):
        self.max_proto, self.w_face, self.w_body, self.thr = max_proto, w_face, w_body, thr
        self.bank: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.tid_hist: Dict[str, List[str]] = {}

    @staticmethod
    def _sim(a, b):
        return float(a @ b)

    def _avg(self, feats):
        rep = np.mean(np.stack(feats, axis=0), 0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    def can_update_proto(self, gid, face_feat, body_feat):
        pool = self.bank[gid]
        if pool['faces'] and self._sim(face_feat, self._avg(pool['faces'])) < FACE_THR_STRICT: return False
        if pool['bodies'] and self._sim(body_feat, self._avg(pool['bodies'])) < BODY_THR_STRICT: return False
        return True

    def _add(self, lst, feat, patch, dir_path):
        if feat is None or patch is None: return
        if lst and max(self._sim(feat, x) for x in lst) < UPDATE_THR: return
        if len(lst) < self.max_proto:
            idx = len(lst)
            lst.append(feat)
        else:
            idx = int(np.argmax([self._sim(feat, x) for x in lst]))
            lst[idx] = 0.7 * lst[idx] + 0.3 * feat
            lst[idx] /= np.linalg.norm(lst[idx]) + 1e-9
        cv2.imwrite(os.path.join(dir_path, f"{idx:02d}.jpg"), patch)

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

    def bind(self, gid, face, body, agg: TrackAgg | None = None, tid=None):
        root = os.path.join(SAVE_DIR, gid)
        self._add(self.bank[gid]['faces'],
                  face,
                  agg.face_patches()[-1] if agg and agg.face else None,
                  os.path.join(root, "faces"))
        self._add(self.bank[gid]['bodies'],
                  body,
                  agg.body_patches()[-1] if agg and agg.body else None,
                  os.path.join(root, "bodies"))

        if tid:
            self.tid_hist.setdefault(gid, [])
            if tid not in self.tid_hist[gid]:
                self.tid_hist[gid].append(tid)

    def new_gid(self):
        gid = f"G{len(self.bank) + 1:05d}"
        self.bank[gid] = dict(faces=[], bodies=[])
        self.tid_hist[gid] = []
        os.makedirs(os.path.join(SAVE_DIR, gid, "faces"), exist_ok=True)
        os.makedirs(os.path.join(SAVE_DIR, gid, "bodies"), exist_ok=True)
        logger.info(f"[GlobalID] new {gid}")
        return gid


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


def dec_det_proc(stream_id, src, q_det2feat, q_det2disp, stop_evt, skip):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"[{stream_id}] open failed")
        q_det2feat.put(SENTINEL)
        q_det2disp.put(SENTINEL)
        return
    bt = ByteTrackPipeline(device="cuda")
    logger.info(f"[{stream_id}] ready")
    fid = 0
    while not stop_evt.is_set():
        ok, frm = cap.read()
        if not ok: break
        fid += 1
        if fid % skip: continue
        dets = bt.update(frm, debug=False)
        small = cv2.resize(frm, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
        q_det2disp.put((stream_id, fid, small, dets))
        H, W = frm.shape[:2]
        patches = [
            frm[max(int(y), 0):min(int(y + h), H), max(int(x), 0):min(int(x + w), W)].copy()
            for x, y, w, h in (d["tlwh"] for d in dets)
        ]
        q_det2feat.put((stream_id, fid, patches, dets))
    cap.release()
    q_det2feat.put(SENTINEL)
    q_det2disp.put(SENTINEL)
    logger.info(f"[{stream_id}] finished")


def feature_proc(q_det2feat, q_map2disp, stop_evt):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reid = PersonReid(DIR_REID_MODEL, which_epoch="last", gpu="0" if dev.type == "cuda" else "")
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    gid_mgr = GlobalID()
    agg_pool: Dict[str, TrackAgg] = {}
    last_seen: Dict[str, int] = {}
    tid2gid: Dict[str, str] = {}
    candidate_state: Dict[str, dict] = {}
    new_gid_state: Dict[str, dict] = {}

    while not stop_evt.is_set():
        pkt = q_det2feat.get()
        if pkt is SENTINEL: break
        stream_id, fid, patches, dets = pkt

        tensors, metas, keep_patches = [], [], []
        for det, patch in zip(dets, patches):
            if not is_long_patch(patch): continue
            tensors.append(prep_patch(patch))
            metas.append((f"{stream_id}_{det['id']}", det["score"]))
            keep_patches.append(patch)

        if tensors:
            batch = torch.stack(tensors).to(dev).float()
            with torch.no_grad():
                outputs = reid.model(batch)
                outputs = torch.nn.functional.normalize(outputs, dim=1)
            feats = outputs.detach().cpu().numpy()
            for (tid, score), feat, img_patch in zip(metas, feats, keep_patches):
                agg = agg_pool.setdefault(tid, TrackAgg())
                agg.add_body(feat, score, fid, img_patch)
                last_seen[tid] = fid

        for det, patch in zip(dets, patches):
            faces = face_app.get(patch)
            if len(faces) != 1: continue
            face_obj = faces[0]
            if getattr(face_obj, "det_score", 1.0) < FACE_DET_MIN_SCORE: continue
            if patch.shape[0] < 120 or patch.shape[1] < 120: continue
            if cv2.Laplacian(patch, cv2.CV_64F).var() < 100: continue
            f_emb = normv(face_obj.embedding)
            tid = f"{stream_id}_{det['id']}"
            agg = agg_pool.setdefault(tid, TrackAgg())
            agg.add_face(f_emb, fid, patch)
            last_seen[tid] = fid

        realtime_map: Dict[str, Dict[int, Tuple[str, float, int]]] = {}
        for tid, agg in list(agg_pool.items()):
            if len(agg.body) < MIN_BODY4GID or len(agg.face) == 0:
                tid_stream, tid_num = tid.split("_", 1)
                realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-1", -1.0, 0)
                continue
            face_feat, body_feat = agg.face_feat(), agg.body_feat()
            cand_gid, cand_score = gid_mgr.probe(face_feat, body_feat)

            if tid in tid2gid:
                bound_gid = tid2gid[tid]
                lock_elapsed = fid - candidate_state.get(tid, {}).get("last_bind_fid", 0)
                if cand_gid != bound_gid and lock_elapsed < BIND_LOCK_FRAMES:
                    n_tid = len(gid_mgr.tid_hist[bound_gid])
                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (bound_gid, cand_score, n_tid)
                    continue

            state = candidate_state.setdefault(tid, {"cand_gid": None, "count": 0, "last_bind_fid": 0})
            if cand_gid and cand_score >= MATCH_THR:
                if state["cand_gid"] == cand_gid:
                    state["count"] += 1
                else:
                    state["cand_gid"] = cand_gid
                    state["count"] = 1
                if state["count"] >= CANDIDATE_FRAMES and gid_mgr.can_update_proto(cand_gid, face_feat, body_feat):
                    gid_mgr.bind(cand_gid, face_feat, body_feat, agg, tid=tid)
                    tid2gid[tid] = cand_gid
                    state["last_bind_fid"] = fid
                    n_tid = len(gid_mgr.tid_hist[cand_gid])
                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (cand_gid, cand_score, n_tid)
                else:
                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-1", -1.0, 0)

            else:
                time_since_last_new = fid - new_gid_state.get(tid, {}).get("last_new_fid", -1)
                ng_state = new_gid_state.setdefault(tid, {"count": 0, "last_new_fid": -NEW_GID_TIME_WINDOW})

                if len(gid_mgr.bank) < 1:
                    new_gid = gid_mgr.new_gid()
                    gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid)
                    tid2gid[tid] = new_gid
                    state["cand_gid"] = new_gid
                    state["count"] = CANDIDATE_FRAMES
                    state["last_bind_fid"] = fid
                    ng_state["last_new_fid"] = fid
                    n_tid = len(gid_mgr.tid_hist[new_gid])
                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (new_gid, cand_score, n_tid)

                elif cand_gid is None or cand_score < THR_NEW_GID:
                    if time_since_last_new >= NEW_GID_TIME_WINDOW:
                        ng_state["count"] += 1
                        if ng_state["count"] >= NEW_GID_MIN_FRAMES:
                            new_gid = gid_mgr.new_gid()
                            gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid)
                            tid2gid[tid] = new_gid
                            state["cand_gid"] = new_gid
                            state["count"] = CANDIDATE_FRAMES
                            state["last_bind_fid"] = fid
                            ng_state["last_new_fid"] = fid
                            ng_state["count"] = 0
                            n_tid = len(gid_mgr.tid_hist[new_gid])
                            tid_stream, tid_num = tid.split("_", 1)
                            realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (new_gid, cand_score, n_tid)
                        else:
                            tid_stream, tid_num = tid.split("_", 1)
                            realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-1", -1.0, 0)
                    else:
                        tid_stream, tid_num = tid.split("_", 1)
                        realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-1", -1.0, 0)
                else:
                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-1", -1.0, 0)

        for tid in list(last_seen.keys()):
            if fid - last_seen[tid] >= MAX_TID_GAP:
                last_seen.pop(tid)
                candidate_state.pop(tid, None)
                tid2gid.pop(tid, None)
                new_gid_state.pop(tid, None)

        q_map2disp.put(realtime_map)

    q_map2disp.put(SENTINEL)
    logger.info("[Feature] finished")


def init_gst(W, H, fps, host, port, use_nvenc=True):
    if use_nvenc:
        cmd = ["gst-launch-1.0", "-q", "fdsrc", "!", "videoparse", f"width={W}", f"height={H}",
               "format=bgr", f"framerate={int(fps)}/1", "!", "videoconvert", "!", "nvh264enc", "zerolatency=true", "!",
               "h264parse", "!", "mpegtsmux", "!", "udpsink", f"host={host}", f"port={port}", "sync=false",
               "async=false"]
    else:
        cmd = ["gst-launch-1.0", "-q", "fdsrc", "!", "videoparse", f"width={W}", f"height={H}",
               "format=bgr", f"framerate={int(fps)}/1", "!", "videoconvert", "!", "x264enc", "tune=zerolatency",
               "speed-preset=ultrafast", "!", "h264parse", "!", "mpegtsmux", "!", "udpsink", f"host={host}",
               f"port={port}",
               "sync=false", "async=false"]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def display_proc(my_stream_id, q_det2disp, q_map2disp, stop_evt, host, port, fps_exp):
    gst, first = None, True
    tid2info = {}
    while not stop_evt.is_set():
        try:
            m = q_map2disp.get_nowait()
            if m is SENTINEL:
                q_det2disp.put(SENTINEL)
                break
            tid2info = m.get(my_stream_id, {})
        except queue.Empty:
            pass
        pkt = q_det2disp.get()
        if pkt is SENTINEL: break
        stream_id, fid, frame, dets = pkt
        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            gid, score, n_tid = tid2info.get(d["id"], ("-1", -1.0, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255) if n_tid >= 2 else (0, 255, 0), 2)
            cv2.putText(frame, f"{d['id']}|{gid} n={n_tid} s={score:.2f}", (x, max(y - 3, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if first:
            H, W = frame.shape[:2]
            gst = init_gst(W, H, fps_exp, host, port, use_nvenc=False)
            first = False
        if gst and gst.poll() is None:
            try:
                gst.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                break
    if gst:
        gst.stdin.close()
        gst.wait()
    logger.info(f"[Display-{my_stream_id}] finished")


def main():
    mp.set_start_method("spawn", force=True)
    pa = argparse.ArgumentParser()
    pa.add_argument("--video1", default="rtsp://admin:1QAZ2wsx@172.20.20.64")
    pa.add_argument("--video2", default="rtsp://admin:1qaz2wsx@172.20.20.150")
    pa.add_argument("--skip", type=int, default=2)
    args = pa.parse_args()

    stop_evt = mp.Event()
    q_det2feat, q_map2disp = LatestQueue(1), LatestQueue(1)
    q_det2disp1, q_det2disp2 = LatestQueue(1), LatestQueue(1)
    procs = [
        mp.Process(target=dec_det_proc, args=("cam1", args.video1, q_det2feat, q_det2disp1, stop_evt, args.skip)),
        mp.Process(target=display_proc, args=("cam1", q_det2disp1, q_map2disp, stop_evt, "127.0.0.1", 5000, 25)),
        mp.Process(target=dec_det_proc, args=("cam2", args.video2, q_det2feat, q_det2disp2, stop_evt, args.skip)),
        mp.Process(target=display_proc, args=("cam2", q_det2disp2, q_map2disp, stop_evt, "127.0.0.1", 5001, 25)),
        mp.Process(target=feature_proc, args=(q_det2feat, q_map2disp, stop_evt))
    ]
    [p.start() for p in procs]
    signal.signal(signal.SIGINT, lambda s, f: stop_evt.set())
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(.5)
    finally:
        stop_evt.set()
        for q in (q_det2feat, q_map2disp):
            try:
                q.put_nowait(SENTINEL)
            except:
                pass
        [p.join() for p in procs]
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
