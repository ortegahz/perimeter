#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周界安全 Demo
-------------------------------------------------
1. ByteTrackPipeline            : 目标检测 + 跟踪
2. PersonReid                   : 行人 ReID
3. FaceSearcher.FaceAnalysis    : 人脸特征
4. GlobalIDManager              : 人脸优先、ReID 兜底的身份归并
5. matplotlib 实时可视化        : 代替 cv2.imshow
"""

from __future__ import annotations

import argparse
import os

from cores.byteTrackPipeline import ByteTrackPipeline
from cores.perimeter import PerimeterExecutor
from utils_peri.macros import DIR_BYTE_TRACK, DIR_REID_MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",
                        default="rtsp://admin:1QAZ2wsx@172.20.20.64",
                        help="video path / rtsp url / webcam id")
    parser.add_argument("--exp_file",
                        default=os.path.join(DIR_BYTE_TRACK, "exps/example/mot/yolox_s_mix_det.py"))
    parser.add_argument("--ckpt",
                        default=os.path.join(DIR_BYTE_TRACK, "pretrained/bytetrack_s_mot17.pth.tar"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 1) ByteTrackPipeline
    bt_pipe = ByteTrackPipeline()

    # 2) 周界执行器
    executor = PerimeterExecutor(
        bytetrack_pipe=bt_pipe,
        reid_model_dir=DIR_REID_MODEL,
        face_cache="face_db_cache.pkl"
    )
    executor.run(args.video)
