#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Process 周界安全 Demo  (Decoder+Detector 合并版本)
-------------------------------------------------
P0 (Main)        : 创建队列 / 事件并拉起子进程
P1 Decode+Detect : 解码 + YOLOX + BYTETracker      → q_dd2feat
                  （输出：resized_frame, img_patches, det_results）
P2 Feature&Fusion: ReID / Face / 统一可视化        → q_feat2disp
P3 Streamer      : GStreamer UDP 推流
-------------------------------------------------
每个进程会打印：
    • FPS / 单帧耗时
    • 队列长度
    • put_nowait 失败会报 “queue full”
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import signal
import subprocess
import time
from typing import List

import cv2
from loguru import logger
from yolox.utils.visualize import plot_tracking

# -------------------------------------------------- #
# 内部模块
# -------------------------------------------------- #
from cores.byteTrackPipeline import ByteTrackPipeline

# -------------------------------------------------- #
# 公共参数
# -------------------------------------------------- #
SENTINEL = None  # 队列停止标记
QUEUE_SIZE = 8  # 每级队列长度
SHOW_SCALE = 0.5  # 传给 Feature / Display 的整帧缩放比例


def safe_qsize(q: mp.Queue):
    try:
        return q.qsize()
    except (AttributeError, NotImplementedError):
        return "NA"


# -------------------------------------------------- #
# 进程 1 : Decode + Detect + Track
# -------------------------------------------------- #
def dec_det_proc(src: str,
                 q_out: mp.Queue,
                 stop_evt: mp.Event,
                 skip: int = 1):
    """
    解码、检测、跟踪合一进程。
    输出：
        fid          : 帧序号
        small_frame  : 缩放后的整帧 (BGR)
        patches      : List[np.ndarray]  每个检测框裁剪出的 patch（同样为 BGR）
        det_results  : List[dict]        ByteTrackPipeline 返回的结果
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"[DecDet] failed to open source: {src}")
        q_out.put(SENTINEL)
        return

    bt = ByteTrackPipeline(device="cuda")
    logger.info("[DecDet] model ready.")

    fid, last = 0, time.perf_counter()

    while not stop_evt.is_set():
        ok, frame = cap.read()
        if not ok:
            logger.warning("[DecDet] stream ended / failed.")
            break

        fid += 1
        if fid % skip:  # 主动跳帧
            continue

        # ---------- 检测 / 跟踪 ----------
        t_alg0 = time.perf_counter()
        dets = bt.update(frame, debug=False)  # 每个元素：dict(id, tlwh, score, cls)
        t_alg = time.perf_counter() - t_alg0

        # ---------- 生成缩放帧 ----------
        small_frame = cv2.resize(frame, None,
                                 fx=SHOW_SCALE,
                                 fy=SHOW_SCALE,
                                 interpolation=cv2.INTER_LINEAR)

        # ---------- 对每个 bbox 裁剪 patch ----------
        patches: List = []
        h_img, w_img = frame.shape[:2]
        for r in dets:
            x, y, w, h = r["tlwh"]
            x1, y1 = int(max(x, 0)), int(max(y, 0))
            x2, y2 = int(min(x + w, w_img - 1)), int(min(y + h, h_img - 1))
            patch = frame[y1:y2, x1:x2].copy()
            patches.append(patch)

        # ---------- 送下游 ----------
        try:
            q_out.put_nowait((fid, small_frame, patches, dets))
        except queue.Full:
            logger.warning("[DecDet] q_out full, dropping a frame")
            continue

        # ---------- 日志 ----------
        fps = 1.0 / (time.perf_counter() - last)
        last = time.perf_counter()
        logger.debug(
            f"[DecDet]  Fid={fid:<6d}  "
            f"alg={t_alg * 1e3:5.1f} ms  {fps:5.1f} fps  "
            f"q_out={safe_qsize(q_out)}"
        )

    cap.release()
    q_out.put(SENTINEL)
    logger.info("[DecDet] finished.")


# -------------------------------------------------- #
# 进程 2 : Feature & Fusion
# -------------------------------------------------- #
def feature_proc(q_in: mp.Queue, q_out: mp.Queue, stop_evt: mp.Event):
    """
    演示用：仅将检测结果画到 small_frame 上，
           后续可在此处做 ReID / Face / 全局身份融合等。
    """
    last = time.perf_counter()
    logger.info("[Feature] started.")

    while not stop_evt.is_set():
        pack = q_in.get()
        if pack is SENTINEL:
            break

        fid, small_frame, patches, dets = pack

        # 示例：简单可视化在 small_frame 上
        ratio = SHOW_SCALE
        tlwhs = [tuple([c * ratio for c in r["tlwh"]]) for r in dets]
        ids = [r["id"] for r in dets]
        vis = plot_tracking(small_frame, tlwhs, ids, frame_id=fid)

        # 这里如果需要把 patches 继续送到下游，也可以放到 q_out。
        try:
            q_out.put_nowait((fid, vis))
        except queue.Full:
            logger.warning("[Feature] q_out full, dropping a frame")
            continue

        fps = 1.0 / (time.perf_counter() - last)
        last = time.perf_counter()
        logger.debug(f"[Feature]  Fid={fid:<6d}  {fps:5.1f} fps  "
                     f"q_in={safe_qsize(q_in)}  q_out={safe_qsize(q_out)}")

    q_out.put(SENTINEL)
    logger.info("[Feature] finished.")


# -------------------------------------------------- #
# 进程 3 : Display (GStreamer UDP)
# -------------------------------------------------- #
def init_gst(w, h, fps, host="127.0.0.1", port=5000):
    cmd = [
        "gst-launch-1.0", "-q",
        "fdsrc", f"blocksize={w * h * 3}", "!",
        "videoparse", f"width={w}", f"height={h}",
        "format=bgr", f"framerate={int(fps)}/1", "!",
        "videoconvert", "!",
        "queue", "leaky=downstream",
        "max-size-buffers=0", "max-size-bytes=0", "max-size-time=0", "!",
        "x264enc",
        "tune=zerolatency", "speed-preset=ultrafast",
        f"key-int-max={int(fps)}", "bframes=0", "byte-stream=true",
        "threads=1", "!",
        "h264parse", "!",
        "mpegtsmux", "alignment=7", "latency=0", "!",
        "udpsink", f"host={host}", f"port={port}",
        "sync=false", "async=false", "qos=false"
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def display_proc(q_in: mp.Queue, stop_evt: mp.Event,
                 host="127.0.0.1", port=5000, exp_fps=25.0):
    logger.info(f"[Display] started, push → udp://{host}:{port}")
    gst_proc, first = None, True
    last = time.perf_counter()

    while not stop_evt.is_set():
        itm = q_in.get()
        if itm is SENTINEL:
            break

        fid, frame = itm
        if first:
            H, W = frame.shape[:2]
            gst_proc = init_gst(W, H, exp_fps, host, port)
            logger.info(f"[Display] GST ready  ({W}x{H}@{exp_fps})")
            first = False

        try:
            if gst_proc and gst_proc.poll() is None:
                gst_proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError):
            logger.warning("[Display] GST pipe closed")
            gst_proc = None

        fps = 1.0 / (time.perf_counter() - last)
        last = time.perf_counter()
        logger.debug(f"[Display] Fid={fid:<6d}  {fps:5.1f} fps  "
                     f"q_in={safe_qsize(q_in)}")

    if gst_proc:
        gst_proc.stdin.close()
        gst_proc.wait()
    logger.info("[Display] finished.")


# -------------------------------------------------- #
# 主程序
# -------------------------------------------------- #
def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",
                        default="rtsp://admin:1QAZ2wsx@172.20.20.64")
    parser.add_argument("--udp_host", default="127.0.0.1")
    parser.add_argument("--udp_port", type=int, default=5000)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--skip", type=int, default=1,)
    args = parser.parse_args()

    # 队列 / 事件
    q_dd2feat = mp.Queue(maxsize=QUEUE_SIZE)
    q_feat2disp = mp.Queue(maxsize=QUEUE_SIZE)
    stop_evt = mp.Event()

    procs = [
        mp.Process(target=dec_det_proc,
                   args=(args.video, q_dd2feat, stop_evt, args.skip),
                   name="DecDet"),
        mp.Process(target=feature_proc,
                   args=(q_dd2feat, q_feat2disp, stop_evt),
                   name="Feature"),
        mp.Process(target=display_proc,
                   args=(q_feat2disp, stop_evt,
                         args.udp_host, args.udp_port, args.fps),
                   name="Display"),
    ]
    [p.start() for p in procs]

    def _sigint(sig, frm):
        logger.warning("SIGINT received → stopping ...")
        stop_evt.set()

    signal.signal(signal.SIGINT, _sigint)

    try:
        while any(p.is_alive() for p in procs):
            time.sleep(0.5)
    finally:
        stop_evt.set()
        # 向所有队列广播 SENTINEL，帮助子进程尽快退出
        for q in (q_dd2feat, q_feat2disp):
            try:
                q.put_nowait(SENTINEL)
            except Exception:
                pass
        [p.join() for p in procs]

    logger.info("All done.")


if __name__ == "__main__":
    main()
