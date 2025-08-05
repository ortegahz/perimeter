#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.ion()

from cores.byteTrackPipeline import ByteTrackPipeline
from utils_peri.macros import DIR_BYTE_TRACK

SKIP = 6  # 只对每 SKIP 帧做一次推理/跟踪
id2color = {}  # ---------- id -> BGR 颜色 ----------


def rand_color():
    """生成一条亮色 (B, G, R)"""
    return tuple(int(x) for x in random.sample(range(64, 256), 3))  # 亮色


def imshow_plt(frame_bgr, last_handle=None):
    quit_flg = False
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if last_handle is None:
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
    tracker = ByteTrackPipeline(
        exp_file=os.path.join(DIR_BYTE_TRACK, "exps/example/mot/yolox_s_mix_det.py"),
        ckpt=os.path.join(DIR_BYTE_TRACK, "pretrained/bytetrack_s_mot17.pth.tar"),
        device="cuda",
        fp16=True,
    )

    cap = cv2.VideoCapture("rtsp://admin:1QAZ2wsx@172.20.20.64")

    vis_handle = None
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Video finished or cannot fetch frame.")
            break

        frame_id += 1
        if frame_id % SKIP != 0:
            continue  # 跳过未处理帧

        results = tracker.update(frame)

        # ---------- 绘制结果 ----------
        for r in results:
            tid = r["id"]
            x, y, w, h = map(int, r["tlwh"])

            # 获取颜色：如果这个 id 第一次出现就随机生成并保存
            if tid not in id2color:
                id2color[tid] = rand_color()
            color = id2color[tid]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'ID:{tid}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        vis_handle, quit_flag = imshow_plt(frame, vis_handle)
        if quit_flag:
            print("Quit key pressed, exiting ...")
            break

    cap.release()
    plt.close("all")
