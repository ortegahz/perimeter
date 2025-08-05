#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import subprocess
import time
from pathlib import Path  # 仍可能用到录像目录

import cv2

from cores.byteTrackPipeline import ByteTrackPipeline

id2color = {}  # tid → BGR 颜色


def rand_color():
    return tuple(int(x) for x in random.sample(range(64, 256), 3))


def init_gst(w, h, fps, host='127.0.0.1', port=5000):
    """
    裸帧 → videoparse → x264enc(极限低延迟) → h264parse
         → mpegtsmux(latency=0) → udpsink(sync=false)
    """
    cmd = [
        'gst-launch-1.0', '-q',
        # ---------- 裸帧输入 ----------
        'fdsrc', f'blocksize={w * h * 3}', '!',
        'videoparse', f'width={w}', f'height={h}',
        'format=bgr', f'framerate={int(fps)}/1', '!',
        'videoconvert', '!',
        # ---------- 泄洪队列 ----------
        'queue', 'leaky=downstream',
        'max-size-buffers=0', 'max-size-bytes=0',
        'max-size-time=0', '!',
        # ---------- H.264 编码 ----------
        'x264enc',
        'tune=zerolatency',
        'speed-preset=ultrafast',
        f'key-int-max={int(fps)}',  # 每秒 1 个 I 帧
        'bframes=0',
        'byte-stream=true',
        'threads=1',
        '!',
        'h264parse', '!',
        # ---------- TS 封装 ----------
        'mpegtsmux', 'alignment=7', 'latency=0', '!',
        # ---------- UDP 发送 ----------
        'udpsink', f'host={host}', f'port={port}',
        'sync=false', 'async=false', 'qos=false'
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


if __name__ == "__main__":
    # ==============================================================
    # 1. 打开输入源
    # ==============================================================
    cap = cv2.VideoCapture("rtsp://admin:1QAZ2wsx@172.20.20.64")
    if not cap.isOpened():
        raise RuntimeError("Cannot open RTSP source")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ==============================================================
    # 2. 初始化：录像 + GStreamer 推流
    # ==============================================================
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path('/home/manu/tmp').mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter('/home/manu/tmp/annotated.mp4',
                             fourcc, FPS, (W, H))

    HOST, PORT = '127.0.0.1', 5000
    gst_proc = init_gst(W, H, FPS, HOST, PORT)
    tracker = ByteTrackPipeline()

    # ==============================================================
    # 3. 主循环
    # ==============================================================
    frame_id = 0
    while True:
        # ---------- 解码 ----------
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t_decode = time.perf_counter() - t0
        if not ok:
            print("Video finished or cannot fetch frame.")
            continue

        frame_id += 1

        # ---------- 跟踪 ----------
        t0 = time.perf_counter()
        results = tracker.update(frame, debug=False)
        t_update = time.perf_counter() - t0

        # ---------- 画框 ----------
        t0 = time.perf_counter()
        for r in results:
            tid = r["id"]
            x, y, w, h = map(int, r["tlwh"])
            if tid not in id2color:
                id2color[tid] = rand_color()
            color = id2color[tid]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'ID:{tid}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        t_draw = time.perf_counter() - t0

        # ---------- 叠加时间戳 ----------
        now = time.time()
        timestr = time.strftime('%H:%M:%S.') + f'{int(now * 1000) % 1000:03d}'
        cv2.putText(frame, timestr, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---------- 输出 ----------
        # writer.write(frame)
        try:
            if gst_proc and gst_proc.poll() is None:
                gst_proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError):
            print('[Warn] GStreamer pipe closed.')
            gst_proc = None

        # ---------- 日志 ----------
        print(f"[Frame {frame_id:6d}] "
              f"Decode={t_decode * 1e3:6.1f} ms  "
              f"Update={t_update * 1e3:6.1f} ms  "
              f"Draw={t_draw * 1e3:6.1f} ms")

    # ==============================================================
    # 4. 资源释放
    # ==============================================================
    cap.release()
    writer.release()
    if gst_proc:
        gst_proc.stdin.close()
        gst_proc.wait()
