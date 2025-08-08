#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-/Video Face-Detection Demo  (InsightFace + onnxruntime)
改进版：
    1. 支持 warm-up
    2. 单张图片可重复多次推理做基准
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

# ---------------- 内部业务 -----------------
from cores.faceSearcher import FaceSearcher


# -------------------------------------------

def draw_faces(img: np.ndarray, faces):
    vis = img.copy()
    if not faces:
        return vis
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if f.kps is not None:
            for x, y in f.kps.astype(int):
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
    return vis


def is_cam(src: str) -> bool:
    return (len(src) == 1 and src.isdigit()) or src.lower() in {"cam", "camera"}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--src", default="/home/manu/图片/vlcsnap-2025-08-01-18h52m04s342.png")
    pa.add_argument("--provider", default="CUDAExecutionProvider",
                    choices=["CUDAExecutionProvider", "CPUExecutionProvider"])
    pa.add_argument("--pause", action="store_true",
                    help="静态图片检测完后停在窗口直到按键")
    pa.add_argument("--det-size", type=int, default=640, help="SCRFD 输入尺寸")
    pa.add_argument("--warmup", type=int, default=3, help="预热次数（不计入统计）")
    pa.add_argument("--repeat", type=int, default=100, help="正式计时次数（仅静态图）")
    args = pa.parse_args()

    logger.info(f"ORT providers: {ort.get_available_providers()}")

    # 1. 初始化模型 -----------------------------------------------------------------
    fs = FaceSearcher(provider=args.provider)
    face_app = fs.app
    ctx_id = 0 if args.provider.startswith("CUDA") else -1
    face_app.prepare(ctx_id=ctx_id,
                     det_thresh=0.5,
                     det_size=(args.det_size, args.det_size))
    logger.info(f"Face detector ready  (provider={args.provider}, ctx_id={ctx_id})")

    # 2. 打开数据源 -----------------------------------------------------------------
    src = args.src
    if is_cam(src):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else 0)
        if not cap.isOpened():
            logger.error("无法打开摄像头")
            return
        logger.info("Press [q] to quit")
        img = None
    else:
        p = Path(src)
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(p))
            if img is None:
                logger.error("读取图片失败")
                return
            cap = None
        else:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                logger.error("无法打开视频/流")
                return
            img = None

    # -------------------- Warm-up --------------------
    warm_frame = img if img is not None else (np.zeros((args.det_size, args.det_size, 3), np.uint8))
    for i in range(args.warmup):
        _ = face_app.get(warm_frame)
    logger.info(f"Finished {args.warmup} warm-up runs")

    # 3. 主循环 / 计时 ---------------------------------
    tot_f, tot_t = 0, 0.0
    if img is not None:  # -------- 静态图片基准 --------
        for i in range(args.repeat):
            t0 = time.perf_counter()
            faces = face_app.get(img)
            t1 = time.perf_counter()

            det_ms = (t1 - t0) * 1e3
            tot_f += 1
            tot_t += (t1 - t0)

            # vis = draw_faces(img, faces)
            # cv2.putText(vis, f"faces:{len(faces)} det:{det_ms:.1f}ms",
            #             (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            #             (0, 255, 255), 2)
            # cv2.imshow("FaceDetection", vis)
            # key = cv2.waitKey(1 if not args.pause else 0)
            # if key in (ord('q'), 27):
            #     break
        if args.pause:
            cv2.waitKey(0)

    else:  # -------- 摄像头 / 视频流 --------
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            faces = face_app.get(frame)
            t1 = time.perf_counter()

            det_ms = (t1 - t0) * 1e3
            tot_f += 1
            tot_t += (t1 - t0)

            vis = draw_faces(frame, faces)
            cv2.putText(vis, f"faces:{len(faces)} det:{det_ms:.1f}ms",
                        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2)
            cv2.imshow("FaceDetection", vis)
            key = cv2.waitKey(1)
            if key in (ord('q'), 27):
                break

    # 4. 汇总 -----------------------------------------
    if tot_f:
        logger.info(f"Total {tot_f} frames  "
                    f"Avg {tot_t / tot_f * 1e3:.2f} ms  "
                    f"FPS {tot_f / tot_t:.2f}")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
