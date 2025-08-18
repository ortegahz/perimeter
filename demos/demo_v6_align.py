#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import cv2
from tqdm import tqdm  # 进度条

from cores.byteTrackPipeline import ByteTrackPipeline
from cores.featureProcessor import FaceSearcher
from cores.featureProcessor import FeatureProcessor

# 颜色
COMMON_COLORS = [
    (255, 0, 0), (0, 255, 0),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (64, 64, 255), (255, 64, 64), (64, 255, 64)
]

SHOW_SCALE = 0.5  # 显示缩放比例


def get_tid_color(tid, tid2color, cmap=COMMON_COLORS):
    if tid not in tid2color:
        color = cmap[len(tid2color) % len(cmap)]
        tid2color[tid] = color
    return tid2color[tid]


def main():
    # ===== 配置部分 =====
    video_path = "/home/manu/tmp/64.mp4"
    output_path = "/home/manu/tmp/output_result.mp4"
    results_txt = "/home/manu/tmp/output_result.txt"  # <-- 变化: 输出txt路径
    skip = 5  # 抽帧间隔

    logger = logging.getLogger("SingleVideo")
    logging.basicConfig(level=logging.INFO)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SHOW_SCALE)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SHOW_SCALE)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    logger.info(f"输出视频: {output_path} size=({W},{H}) fps={fps} 总帧:{total_frames}")

    bt = ByteTrackPipeline(device="cuda")
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app
    processor = FeatureProcessor(device="cuda")
    tid2color = {}

    fid = 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    # <-- 变化: 打开结果txt文件
    f_results = open(results_txt, "w", encoding="utf-8")
    f_results.write("frame_id,cam_id,tid,gid,score,n_tid\n")  # 表头

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        pbar.update(1)

        if fid % skip != 0:
            continue

        dets = bt.update(frame, debug=False)

        small = cv2.resize(frame, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
        H0, W0 = frame.shape[:2]
        patches = [
            frame[max(int(y), 0):min(int(y + h), H0), max(int(x), 0):min(int(x + w), W0)].copy()
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

        # 特征处理
        realtime_map = processor.process_packet(("cam1", fid, patches, dets))

        # <-- 变化: 保存 realtime_map 到 txt
        for cam_id, tid_map in realtime_map.items():
            for tid, (gid, score, n_tid) in tid_map.items():
                f_results.write(f"{fid},{cam_id},{tid},{gid},{score:.4f},{n_tid}\n")

        # 绘制检测框
        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            gid, score, n_tid = realtime_map.get("cam1", {}).get(d["id"], ("-1", -1.0, 0))
            color = (0, 255, 0) if n_tid < 2 else (0, 0, 255)
            cv2.rectangle(small, (x, y), (x + w, y + h), color, 2)
            cv2.putText(small, f"GID:{gid}", (x, max(y + 15, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(small, f"n={n_tid} s={score:.2f}", (x, max(y + 30, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for face in face_info:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            cv2.rectangle(small, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(small, f"{face['score']:.2f}", (x1, max(y1 - 2, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
            if "kps" in face:
                for kx, ky in face["kps"]:
                    kx = int(kx * SHOW_SCALE)
                    ky = int(ky * SHOW_SCALE)
                    cv2.circle(small, (kx, ky), 1, (0, 0, 255), 2)

        video_writer.write(small)

    cap.release()
    video_writer.release()
    f_results.close()  # <-- 变化: 关闭结果txt
    pbar.close()
    logger.info(f"处理完成，结果已保存到 {results_txt}")


if __name__ == "__main__":
    main()
