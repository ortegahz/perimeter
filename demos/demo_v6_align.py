#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单路离线视频 + 全局 GID（FeatureProcessor 版）

1. 依然沿用 ByteTrack 做人体/多目标跟踪；
2. 依然调用 InsightFace 做人脸检测（如果你后续打算在
   FeatureProcessor 里统一做人脸也可以把这部分删除）；
3. 将 (cam_id, fid, patches, dets) 打包后送入 FeatureProcessor，
   得到 realtime_map → 里面维护了 tid → gid 等全局信息；
4. 把 realtime_map 同步写到 txt；同时把可视化结果写入 mp4。

与多路实时版本相比：
- 省去多进程、队列、事件等；逻辑简单。
"""

import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from cores.byteTrackPipeline import ByteTrackPipeline
from cores.featureProcessor import FaceSearcher, FeatureProcessor

# ---------------- 可调参数 ----------------
VIDEO_PATH = "/home/manu/tmp/64.mp4"  # 输入视频
OUTPUT_MP4 = "/home/manu/tmp/output_result.mp4"
OUTPUT_TXT = "/home/manu/tmp/output_result.txt"
SKIP = 5  # 抽帧间隔
SHOW_SCALE = 0.5  # 画面缩放
DEVICE = "cuda"  # 推理设备
CAM_ID = "cam1"  # 给 FeatureProcessor 的相机名
# -----------------------------------------

# 颜色
_COMMON_COLORS = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 0),
    (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (64, 64, 255),
    (255, 64, 64), (64, 255, 64)
]


def get_tid_color(tid, table, cmap=_COMMON_COLORS):
    if tid not in table:
        table[tid] = cmap[len(table) % len(cmap)]
    return table[tid]


def main():
    # ---------- 日志 ----------
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                        level=logging.INFO)
    logger = logging.getLogger("SingleVideo")

    # ---------- 打开视频 ----------
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

    vw_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_MP4, vw_fourcc, fps, (vis_W, vis_H))
    logger.info(f"输出视频: {OUTPUT_MP4} size=({vis_W},{vis_H}) fps={fps} 总帧:{total_frames}")

    # ---------- 模型 ----------
    tracker = ByteTrackPipeline(device=DEVICE)
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    processor = FeatureProcessor(device=DEVICE)

    # ---------- 颜色映射 ----------
    tid2color = {}

    # ---------- 结果 txt ----------
    f_res = open(OUTPUT_TXT, "w", encoding="utf-8")
    f_res.write("frame_id,cam_id,tid,gid,score,n_tid\n")

    # ---------- 主循环 ----------
    fid = 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            pbar.update(1)

            # 抽帧
            if fid % SKIP != 0:
                continue

            # ① 检测&跟踪 --------------------------------------------------
            dets = tracker.update(frame, debug=False)

            # ② 裁剪出每个 bbox 的 patch（送特征）--------------------------
            H0, W0 = frame.shape[:2]
            patches = [
                frame[max(int(y), 0):min(int(y + h), H0),
                max(int(x), 0):min(int(x + w), W0)].copy()
                for x, y, w, h in (d["tlwh"] for d in dets)
            ]

            # ③ 人脸检测（缩放到 SHOW_SCALE，和实时脚本一致）---------------
            small = cv2.resize(frame, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
            faces_bboxes, faces_kpss = face_app.det_model.detect(
                small, max_num=0, metric="default")

            face_info = []
            if faces_bboxes is not None and faces_bboxes.shape[0] > 0:
                for i in range(faces_bboxes.shape[0]):
                    bi = faces_bboxes[i, :4].astype(int)
                    x1, y1, x2, y2 = [int(b / SHOW_SCALE) for b in bi]
                    sc = float(faces_bboxes[i, 4])
                    kps = faces_kpss[i].astype(int).tolist() if faces_kpss is not None else None
                    if kps:
                        kps = [[int(k[0] / SHOW_SCALE), int(k[1] / SHOW_SCALE)] for k in kps]
                    face_info.append({"bbox": [x1, y1, x2, y2], "score": sc, "kps": kps})

            # ④ 特征处理 (得到全局 gid 映射) -------------------------------
            realtime_map = processor.process_packet((CAM_ID, fid, patches, dets))

            # 把结果写文件
            cam_map = realtime_map.get(CAM_ID, {})
            for tid, (gid, score, n_tid) in cam_map.items():
                f_res.write(f"{fid},{CAM_ID},{tid},{gid},{score:.4f},{n_tid}\n")

            # ⑤ 可视化 ------------------------------------------------------
            vis = cv2.resize(frame, (vis_W, vis_H))

            for d in dets:
                tid = d["id"]
                x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
                gid, score, n_tid = cam_map.get(tid, ("-1", -1.0, 0))
                color = (0, 255, 0) if n_tid < 2 else (0, 0, 255)

                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis, f"G:{gid}", (x, max(y + 15, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis, f"n={n_tid} s={score:.2f}",
                            (x, max(y + 30, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1)

            # 人脸
            for face in face_info:
                x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis, f"{face['score']:.2f}",
                            (x1, max(y1 - 2, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 128, 0), 1)
                if face["kps"]:
                    for kx, ky in face["kps"]:
                        kx = int(kx * SHOW_SCALE);
                        ky = int(ky * SHOW_SCALE)
                        cv2.circle(vis, (kx, ky), 1, (0, 0, 255), 2)

            video_writer.write(vis)

    finally:
        # ---------- 资源释放 ----------
        cap.release()
        video_writer.release()
        f_res.close()
        pbar.close()
        logger.info(f"处理完成，结果保存至: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
