#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

import cv2
import torch
from loguru import logger

# ------------ 内部模块 ------------
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from demo_v6 import GlobalID, is_long_patch, prep_patch, TrackAgg, normv
from utils_peri.general_funcs import make_dirs
from utils_peri.macros import DIR_REID_MODEL

# ---------------------------------

SHOW_SCALE = 0.5
SENTINEL = None
MIN_BODY4GID = 8
MIN_FACE4GID = 8
MATCH_THR = 0.5
THR_NEW_GID = 0.3
FACE_DET_MIN_SCORE = 0.60
SAVE_DIR = "/home/manu/tmp/perimeter"
os.makedirs(SAVE_DIR, exist_ok=True)
make_dirs(SAVE_DIR, reset=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument(
        "--video1",
        default="/home/manu/tmp/vlc-record-2025-08-15-14h16m11s-rtsp___172.20.20.88_554_LiveMedia_ch1_Media1_trackID=1-.mp4"
    )
    pa.add_argument("--skip", type=int, default=2)
    pa.add_argument("--out", type=str, default="/home/manu/tmp/result.mp4", help="输出结果视频路径")
    args = pa.parse_args()

    cap = cv2.VideoCapture(args.video1)
    if not cap.isOpened():
        logger.error(f"[single] open failed: {args.video1}")
        return

    # 初始化模块
    bt = ByteTrackPipeline(device="cuda")
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    reid = PersonReid(DIR_REID_MODEL, which_epoch="last", gpu="0")
    gid_mgr = GlobalID()

    # 保存结果
    out_writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    agg_pool = {}
    last_seen = {}
    tid2gid = {}
    candidate_state = {}
    new_gid_state = {}
    fid = 0

    while True:
        ok, frm = cap.read()
        if not ok:
            break
        fid += 1
        if fid % args.skip != 0:
            continue

        # ==== dec_det_proc 部分 ====
        dets = bt.update(frm, debug=False)
        small = cv2.resize(frm, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
        H, W = frm.shape[:2]
        patches = [
            frm[max(int(y), 0):min(int(y + h), H), max(int(x), 0):min(int(x + w), W)].copy()
            for x, y, w, h in (d["tlwh"] for d in dets)
        ]
        faces_bboxes, faces_kpss = face_app.det_model.detect(small, max_num=0, metric='default')
        face_info = []
        if faces_bboxes is not None and faces_bboxes.shape[0] > 0:
            for i in range(faces_bboxes.shape[0]):
                bi = faces_bboxes[i, :4].astype(int)
                x1, y1, x2, y2 = [int(b / SHOW_SCALE) for b in bi]
                score = float(faces_bboxes[i, 4])
                kps = faces_kpss[i].astype(int).tolist() if faces_kpss is not None else None
                if kps is not None:
                    kps = [[int(kp[0] / SHOW_SCALE), int(kp[1] / SHOW_SCALE)] for kp in kps]
                face_info.append({"bbox": [x1, y1, x2, y2], "score": score, "kps": kps})

        # ==== feature_proc 部分（简化：只保留特征提取+聚合）
        tensors, metas, keep_patches = [], [], []
        for det, patch in zip(dets, patches):
            if not is_long_patch(patch): continue
            tensors.append(prep_patch(patch))
            metas.append((f"cam1_{det['id']}", det["score"]))
            keep_patches.append(patch)

        if tensors:
            batch = torch.stack(tensors).cuda().float()
            with torch.no_grad():
                outputs = reid.model(batch)
                outputs = torch.nn.functional.normalize(outputs, dim=1)
            feats = outputs.cpu().numpy()
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
            tid = f"cam1_{det['id']}"
            agg = agg_pool.setdefault(tid, TrackAgg())
            agg.add_face(f_emb, fid, patch)
            last_seen[tid] = fid

        # ==== 这里你可以插入 feature_proc 中的状态机匹配逻辑 ====
        # 简化，这里先空字典输出，实际可直接复制原 feature_proc 的GID匹配部分
        realtime_map = {}
        tid2info = realtime_map.get("cam1", {})

        # ==== display_proc 改为直接写视频 ====
        frame = small.copy()
        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            gid, score, n_tid = tid2info.get(d["id"], ("-1", -1.0, 0))
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255) if n_tid >= 2 else color, 2)
            cv2.putText(frame, f"{gid}", (x, max(y + 15, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"n={n_tid} s={score:.2f}", (x, max(y + 30, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for face in face_info:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{face['score']:.2f}", (x1, max(y1 - 2, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
            if "kps" in face:
                for kx, ky in face["kps"]:
                    kx = int(kx * SHOW_SCALE)
                    ky = int(ky * SHOW_SCALE)
                    cv2.circle(frame, (kx, ky), 1, (0, 0, 255), 2)

        # 初始化输出文件
        if out_writer is None:
            H_out, W_out = frame.shape[:2]
            out_writer = cv2.VideoWriter(args.out, fourcc, fps, (W_out, H_out))
            logger.info(f"保存结果视频到 {args.out}")

        out_writer.write(frame)

    cap.release()
    if out_writer:
        out_writer.release()
    logger.info(f"处理完成，结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
