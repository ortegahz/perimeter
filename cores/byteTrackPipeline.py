#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultralytics-YOLO(11) + BYTETracker
"""

import time
from typing import List, Dict

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

__all__ = ["ByteTrackPipeline"]


class ByteTrackPipeline:
    def __init__(
            self,
            # ---------- YOLO-11 ----------
            yolo_weight: str = "yolo11n.pt",
            conf_thres: float = 0.25,
            iou_thres: float = 0.7,
            device: str = "cuda",
            # ---------- BYTETracker ----------
            track_thresh: float = 0.5,
            track_buffer: int = 30,
            match_thresh: float = 0.8,
            aspect_ratio_thresh: float = 1.6,
            min_box_area: float = 10,
            fps: int = 30,
            mot20: bool = False,
    ):
        # 1. Detector ----------------------------------------------------------
        self.device = torch.device(
            "cuda" if device in ("cuda", "gpu") and torch.cuda.is_available() else "cpu"
        )
        self.detector = YOLO(yolo_weight)
        logger.info(f"[ByteTrackPipeline] YOLO-11 model '{yolo_weight}' loaded on {self.device}.")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 2. BYTETracker -------------------------------------------------------
        class _Args:  # 用来临时存放超参数
            pass

        args = _Args()
        args.track_thresh = track_thresh
        args.track_buffer = track_buffer
        args.match_thresh = match_thresh
        args.aspect_ratio_thresh = aspect_ratio_thresh
        args.min_box_area = min_box_area
        args.mot20 = mot20

        self.tracker = BYTETracker(args, frame_rate=fps)

        # 3. 其它参数 ----------------------------------------------------------
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.frame_id = 0
        self.timer = Timer()

    # ---------------------------------------------------------------------- #
    # 更新
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray, debug: bool = True) -> List[Dict]:
        if debug is None:
            debug = getattr(self, "debug", False)

        self.frame_id += 1
        t_total0 = time.perf_counter() if debug else 0.0

        # ---------------- 1. YOLO-11 推理 ----------------
        t_det0 = time.perf_counter() if debug else 0.0
        yolo_out = self.detector.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=0 if self.device.type == "cuda" else "cpu",
            verbose=False,
        )[0]

        # --- MODIFIED START: 过滤非 person 类的检测结果 ---
        detections = None  # 默认为空
        if yolo_out.boxes is not None and yolo_out.boxes.shape[0] > 0:
            xyxy = yolo_out.boxes.xyxy.cpu()  # (N,4)
            conf = yolo_out.boxes.conf.cpu()  # (N,)
            cls = yolo_out.boxes.cls.cpu()  # (N,)

            # 1. 创建一个布尔掩码，只选择类别ID为0 (person) 的检测框
            person_mask = (cls == 0)

            # 2. 应用掩码，过滤掉所有非 person 的检测结果
            filtered_xyxy = xyxy[person_mask]

            # 3. 只有在过滤后还有 person 的情况下，才继续处理
            if filtered_xyxy.shape[0] > 0:
                filtered_conf = conf[person_mask]
                filtered_cls = cls[person_mask]  # 虽然都是0，但为了逻辑一致性一起过滤

                # 将类别ID编码到分数中：分数的整数部分是类别ID，小数部分是置信度
                # 对于 person 来说，filtered_cls 都是0，所以 fused_scores 实质上就是 filtered_conf
                fused_scores = (filtered_conf + filtered_cls).unsqueeze(1)  # (N_person, 1)

                # 使用过滤后的 person 检测结果来构建 detections 张量
                detections = torch.cat([filtered_xyxy, fused_scores], dim=1)
        # --- MODIFIED END ---

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        det_time = time.perf_counter() - t_det0 if debug else 0.0

        # ---------------- 2. BYTETracker 更新 -------------
        t_track0 = time.perf_counter() if debug else 0.0
        online_targets = []
        if detections is not None:
            H, W = frame_bgr.shape[:2]
            online_targets = self.tracker.update(
                detections,
                img_info=(H, W),  # (img_h, img_w)
                img_size=(H, W),  # 关键改动：保持与原始分辨率一致
            )
        track_time = time.perf_counter() - t_track0 if debug else 0.0

        # ---------------- 3. 整理结果 ---------------------
        results: List[Dict] = []
        for t in online_targets:
            tlwh = t.tlwh
            # 这里的几何约束检查依然有效
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] <= self.min_box_area or vertical:
                continue

            # 从 track score 中解码出类别ID和真实置信度
            fused_score = t.score
            class_id = int(fused_score)  # 这里得到的 class_id 将永远是 0
            score = fused_score - class_id

            results.append(
                {
                    "id": int(t.track_id),
                    "tlwh": (float(tlwh[0]), float(tlwh[1]),
                             float(tlwh[2]), float(tlwh[3])),
                    "score": float(score),
                    "class_id": class_id,
                    "class_name": self.detector.names[class_id],
                }
            )

        # ---------------- 4. Debug -----------------------
        if debug:
            total_time = time.perf_counter() - t_total0
            logger.debug(
                f"[BT Frame {self.frame_id:>5}] "
                f"Det={det_time * 1e3:6.1f} ms | "
                f"Track={track_time * 1e3:6.1f} ms | "
                f"Total={total_time * 1e3:6.1f} ms"
            )

        return results
