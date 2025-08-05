import os
import os.path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")
plt.ion()

from utils_peri.macros import IMAGE_EXT


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


# --------------------------------------------------
# matplotlib 显示辅助
# --------------------------------------------------
def imshow_plt(frame_bgr: np.ndarray,
               last_handle=None):
    """
    用 matplotlib 实时显示 BGR 图像
    ----------
    返回
        handle    : (fig, ax, img_artist)，供下次刷新
        quit_flag : bool，用户按 q/Q/ESC 时为 True
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 第一次绘制
    if last_handle is None:
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.axis("off")
        img_artist = ax.imshow(rgb)
        fig.canvas.manager.set_window_title("Perimeter")
        plt.tight_layout()

        fig.quit_flag = False

        def _on_key(event):
            if event.key in ("q", "Q", "escape"):
                fig.quit_flag = True

        fig.canvas.mpl_connect("key_press_event", _on_key)
        plt.show(block=False)
        return (fig, ax, img_artist), False

    # 刷新
    fig, ax, img_artist = last_handle
    img_artist.set_data(rgb)
    fig.canvas.draw_idle()
    plt.pause(0.001)

    return last_handle, getattr(fig, "quit_flag", False)
