#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
plt.ion()

from cores.byteTrackPipeline import ByteTrackPipeline

SKIP = 8  # 只对每 SKIP 帧做一次推理/跟踪
id2color = {}  # tid → BGR 颜色


def rand_color():
    return tuple(int(x) for x in random.sample(range(64, 256), 3))


def imshow_plt(frame_bgr, last_handle=None):
    quit_flg = False
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if last_handle is None:  # 第一次
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.axis("off")
        img_artist = ax.imshow(rgb)
        fig.canvas.manager.set_window_title("ByteTrack (matplotlib)")
        plt.tight_layout()

        fig.quit_flg = False
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda e: setattr(fig, "quit_flg", e.key in ("q", "Q", "escape")),
        )

        plt.show(block=False)
        return (fig, ax, img_artist), quit_flg

    fig, ax, img_artist = last_handle
    img_artist.set_data(rgb)
    fig.canvas.draw_idle()
    plt.pause(0.001)

    quit_flg = getattr(fig, "quit_flg", False)
    return last_handle, quit_flg


if __name__ == "__main__":
    tracker = ByteTrackPipeline()  # ByteTrackPipeline 本身会打印 det/track 耗时

    cap = cv2.VideoCapture("rtsp://admin:1QAZ2wsx@172.20.20.64")

    vis_handle = None
    frame_id = 0

    while True:
        # ---------------- 解码计时 ----------------
        t_decode0 = time.perf_counter()
        ok, frame = cap.read()
        t_decode = time.perf_counter() - t_decode0
        if not ok:
            print("Video finished or cannot fetch frame.")
            break

        frame_id += 1
        if frame_id % SKIP != 0:
            continue

        # ---------------- 推理计时 ----------------
        t_update0 = time.perf_counter()
        results = tracker.update(frame, debug=False)  # debug=False 防止重复打印
        t_update = time.perf_counter() - t_update0

        # ---------------- 绘图计时 ----------------
        t_draw0 = time.perf_counter()
        # ---------- 绘制结果 ----------
        for r in results:
            tid = r["id"]
            x, y, w, h = map(int, r["tlwh"])

            if tid not in id2color:
                id2color[tid] = rand_color()
            color = id2color[tid]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'ID:{tid}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        vis_handle, quit_flag = imshow_plt(frame, vis_handle)
        t_draw = time.perf_counter() - t_draw0

        # ---------------- 打印三段耗时 --------------
        print(f"[Frame {frame_id:6d}]  "
              f"Decode={t_decode * 1e3:6.1f} ms  "
              f"Update={t_update * 1e3:6.1f} ms  "
              f"Draw={t_draw * 1e3:6.1f} ms")

        if quit_flag:
            print("Quit key pressed, exiting ...")
            break

    cap.release()
    plt.close("all")
