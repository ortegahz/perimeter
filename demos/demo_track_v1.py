#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.ion()

from cores.byteTrackPipeline import ByteTrackPipeline
from utils_peri.macros import DIR_BYTE_TRACK

# ------------------ 跳帧参数 ------------------
SKIP = 6  # 只对每 SKIP 帧做一次推理/跟踪


# --------------------------------------------

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

        def _on_key(event):
            if event.key in ("q", "Q", "escape"):
                fig.quit_flg = True

        fig.canvas.mpl_connect("key_press_event", _on_key)
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
    results = []  # 保存最近一次推理得到的结果

    while True:
        ok, frame = cap.read()
        frame_id += 1
        if not ok:
            print("Video finished or cannot fetch frame.")
            break

        if frame_id % SKIP != 0:
            continue

        results = tracker.update(frame)

        # 绘制检测/跟踪框（使用最新一次 results）
        for r in results:
            x, y, w, h = map(int, r["tlwh"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{r["id"]}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Matplotlib 显示
        vis_handle, quit_flag = imshow_plt(frame, vis_handle)
        if quit_flag:
            print("Quit key pressed, exiting ...")
            break

    cap.release()
    plt.close("all")
