#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单路离线 + 全局 GID  (Deterministic 版本)
----------------------------------------------------------
1. 全面设置随机种子，保证 ByteTrack / PyTorch 结果可复现
2. 所有 list / dict 输出都做显式排序
3. 可选缓存 process_packet 输入
----------------------------------------------------------
Author: your_name
"""

import json
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from cores.byteTrackPipeline import ByteTrackPipeline
from cores.featureProcessor import FaceSearcher, FeatureProcessor

# ------------------ 确定性设置 ------------------
SEED = 0  # 你可以改其他整数

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
# ------------------------------------------------

# ---------------- 可调参数 ----------------
VIDEO_PATH = "/home/manu/tmp/64.mp4"
OUTPUT_MP4 = "/home/manu/tmp/output_result.mp4"
OUTPUT_TXT = "/home/manu/tmp/output_result.txt"

SKIP = 50
SHOW_SCALE = 0.5
DEVICE = "cpu"
CAM_ID = "cam1"

# ---- 缓存相关 ----
SAVE_RAW = False
LOAD_RAW = True
RAW_DIR = "/home/manu/tmp/cache"
OVERWRITE = False


# -------------------------------------------------

# ---------------- 工具函数 ----------------
def save_packet(packet, root_dir=RAW_DIR, overwrite=OVERWRITE):
    cam_id, fid, patches, dets = packet
    frame_dir = Path(root_dir) / cam_id / f"{fid:06d}"
    if frame_dir.exists() and not overwrite:
        return
    frame_dir.mkdir(parents=True, exist_ok=True)

    patch_names = []
    for i, img in enumerate(patches):
        name = f"patch_{i:02d}.bmp"
        cv2.imwrite(str(frame_dir / name), img)
        patch_names.append(name)

    dets_json = []
    for d in dets:
        dets_json.append(
            {
                "tlwh": [float(x) for x in d["tlwh"]],
                "score": float(d.get("score", 0)),
                "id": int(d.get("id", -1)),
            }
        )

    meta = {"cam_id": cam_id, "fid": fid, "patches": patch_names, "dets": dets_json}
    (frame_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_packet(cam_id, fid, root_dir=RAW_DIR):
    frame_dir = Path(root_dir) / cam_id / f"{fid:06d}"
    if not frame_dir.exists():
        raise FileNotFoundError(f"未找到缓存: {frame_dir}")

    meta = json.loads((frame_dir / "meta.json").read_text("utf-8"))
    patches = [cv2.imread(str(frame_dir / p)) for p in meta["patches"]]
    dets = meta["dets"]
    return cam_id, fid, patches, dets


def compare_packet(p1, p2, eps=1.0):
    cam1, fid1, patches1, dets1 = p1
    cam2, fid2, patches2, dets2 = p2

    if cam1 != cam2 or fid1 != fid2:
        return False, "cam/fid 不一致"

    if len(patches1) != len(patches2):
        return False, "patch 数量不一致"

    for i, (im1, im2) in enumerate(zip(patches1, patches2)):
        if im1.shape != im2.shape:
            return False, f"patch[{i}] shape 不一致"

    if len(dets1) != len(dets2):
        return False, "dets 数量不一致"

    for d1, d2 in zip(dets1, dets2):
        if any(abs(a - b) > eps for a, b in zip(d1["tlwh"], d2["tlwh"])):
            return False, "tlwh 不一致"
        if abs(d1["score"] - d2["score"]) > 1e-3 or d1["id"] != d2["id"]:
            return False, "score/id 不一致"
    return True, ""


def sort_dets_and_patches(dets, patches):
    """按照 dets 的 id / score / tlwh 排序，保持与 patches 对齐"""
    idx = list(range(len(dets)))
    idx.sort(key=lambda i: (dets[i]["id"], -dets[i].get("score", 0), *dets[i]["tlwh"]))
    dets_sorted = [dets[i] for i in idx]
    patches_sorted = [patches[i] for i in idx]
    return dets_sorted, patches_sorted


# --------------- 可视化配色 ---------------
_COMMON_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (64, 64, 255),
    (255, 64, 64),
    (64, 255, 64),
]


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
    )
    logger = logging.getLogger("DeterministicVideo")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ori_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vis_W, vis_H = int(ori_W * SHOW_SCALE), int(ori_H * SHOW_SCALE)

    Path(OUTPUT_MP4).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_TXT).parent.mkdir(parents=True, exist_ok=True)

    video_writer = cv2.VideoWriter(
        OUTPUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis_W, vis_H)
    )

    logger.info(
        f"输出视频: {OUTPUT_MP4}  size=({vis_W},{vis_H})  fps={fps}  总帧:{total_frames}"
    )

    tracker = None if LOAD_RAW else ByteTrackPipeline(device=DEVICE)
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    processor = FeatureProcessor(
        device=DEVICE,
        use_fid_time=True,
        mode='load',  # load or realtime
        cache_path='/home/manu/tmp/features_cache.json'
    )

    f_res = open(OUTPUT_TXT, "w", encoding="utf-8")
    f_res.write("frame_id,cam_id,tid,gid,score,n_tid\n")

    fid = 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            pbar.update(1)

            if fid % SKIP != 0:
                continue

            # ------------- 获得 packet -------------
            if LOAD_RAW:
                packet = load_packet(CAM_ID, fid, RAW_DIR)
                patches, dets = packet[2:]
            else:
                dets = tracker.update(frame, debug=False)
                # 先按 score 降序排，保证进入 Kalman 顺序一致
                dets.sort(key=lambda d: -d.get("score", 0))

                H, W = frame.shape[:2]
                patches = [
                    frame[
                    max(int(y), 0): min(int(y + h), H),
                    max(int(x), 0): min(int(x + w), W),
                    ].copy()
                    for x, y, w, h in (d["tlwh"] for d in dets)
                ]

                # 再按 id 排序，写盘 / 写 txt / 画图都统一
                dets, patches = sort_dets_and_patches(dets, patches)
                packet = (CAM_ID, fid, patches, dets)
                if SAVE_RAW:
                    save_packet(packet, RAW_DIR, overwrite=OVERWRITE)

            # ------------- 人脸检测 -------------
            small = cv2.resize(frame, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
            faces_bboxes, faces_kpss = face_app.det_model.detect(
                small, max_num=0, metric="default"
            )

            face_info = []
            if faces_bboxes is not None and faces_bboxes.shape[0] > 0:
                for i in range(faces_bboxes.shape[0]):
                    bi = faces_bboxes[i, :4].astype(int)
                    x1, y1, x2, y2 = [int(b / SHOW_SCALE) for b in bi]
                    sc = float(faces_bboxes[i, 4])
                    kps = (
                        faces_kpss[i].astype(int).tolist()
                        if faces_kpss is not None
                        else None
                    )
                    if kps:
                        kps = [
                            [int(k[0] / SHOW_SCALE), int(k[1] / SHOW_SCALE)]
                            for k in kps
                        ]
                    face_info.append({"bbox": [x1, y1, x2, y2], "score": sc, "kps": kps})

            # ------------- 特征处理 -------------
            realtime_map = processor.process_packet(packet)
            cam_map = realtime_map.get(CAM_ID, {})

            # 按 tid 升序写 txt
            for tid in sorted(cam_map.keys()):
                gid, score, n_tid = cam_map[tid]
                f_res.write(f"{fid},{CAM_ID},{tid},{gid},{score:.4f},{n_tid}\n")

            # ------------- 可视化 -------------
            vis = cv2.resize(frame, (vis_W, vis_H))

            for d in dets:  # dets 已按 id 排序
                tid = d["id"]
                x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
                gid, score, n_tid = cam_map.get(tid, ("-1", -1.0, 0))
                color = (0, 255, 0) if n_tid < 2 else (0, 0, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    vis,
                    f"G:{gid}",
                    (x, max(y + 15, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                cv2.putText(
                    vis,
                    f"n={n_tid} s={score:.2f}",
                    (x, max(y + 30, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

            for face in face_info:
                x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    vis,
                    f"{face['score']:.2f}",
                    (x1, max(y1 - 2, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 128, 0),
                    1,
                )
                if face["kps"]:
                    for kx, ky in face["kps"]:
                        kx = int(kx * SHOW_SCALE)
                        ky = int(ky * SHOW_SCALE)
                        cv2.circle(vis, (kx, ky), 1, (0, 0, 255), 2)

            video_writer.write(vis)

    finally:
        cap.release()
        video_writer.release()
        f_res.close()
        pbar.close()
        logger.info(f"处理完成，结果保存至: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
