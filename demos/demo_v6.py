#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双路视频 + 全局GID
- 使用严格候选检测、多帧确认、防抖&锁定机制
- 第一个GID：有人脸+body -> 立即创建

- 新增功能：长时间没有更新的gid会被自动删除（包含内存和磁盘数据）
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import signal
import subprocess

# ------------ 内部模块 ------------
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.featureProcessor import *


# ---------------------------------


class LatestQueue(mpq.Queue):
    def __init__(self, maxsize=1, *, ctx=None):
        super().__init__(maxsize, ctx=ctx or mp.get_context())

    def put(self, obj, block=True, timeout=None):
        try:
            while True:
                self.get_nowait()
        except queue.Empty:
            pass
        super().put(obj, block, timeout)


def dec_det_proc(stream_id, src, q_det2feat, q_det2disp, stop_evt, skip):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"[{stream_id}] open failed")
        q_det2feat.put(SENTINEL)
        q_det2disp.put(SENTINEL)
        return
    bt = ByteTrackPipeline(device="cuda")
    face_app = FaceSearcher(provider="CUDAExecutionProvider").app  # 仅初始化一次
    logger.info(f"[{stream_id}] ready")
    fid = 0
    while not stop_evt.is_set():
        ok, frm = cap.read()
        if not ok: break
        fid += 1
        if fid % skip: continue
        dets = bt.update(frm, debug=False)
        small = cv2.resize(frm, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
        H, W = frm.shape[:2]
        patches = [
            frm[max(int(y), 0):min(int(y + h), H), max(int(x), 0):min(int(x + w), W)].copy()
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
        q_det2disp.put((stream_id, fid, small, dets, face_info))
        q_det2feat.put((stream_id, fid, patches, dets))
    cap.release()
    q_det2feat.put(SENTINEL)
    q_det2disp.put(SENTINEL)
    logger.info(f"[{stream_id}] finished")


def feature_proc(q_det2feat, q_map2disp, stop_evt):
    processor = FeatureProcessor(device="cuda")
    while not stop_evt.is_set():
        pkt = q_det2feat.get()
        if pkt is SENTINEL:
            break
        realtime_map = processor.process_packet(pkt)
        q_map2disp.put(realtime_map)
    q_map2disp.put(SENTINEL)
    logger.info("[Feature] finished")


def init_gst(W, H, fps, host, port, use_nvenc=True):
    if use_nvenc:
        cmd = ["gst-launch-1.0", "-q", "fdsrc", "!", "videoparse", f"width={W}", f"height={H}",
               "format=bgr", f"framerate={int(fps)}/1", "!", "videoconvert", "!", "nvh264enc", "zerolatency=true", "!",
               "h264parse", "!", "mpegtsmux", "!", "udpsink", f"host={host}", f"port={port}", "sync=false",
               "async=false"]
    else:
        cmd = ["gst-launch-1.0", "-q", "fdsrc", "!", "videoparse", f"width={W}", f"height={H}",
               "format=bgr", f"framerate={int(fps)}/1", "!", "videoconvert", "!", "x264enc", "tune=zerolatency",
               "speed-preset=ultrafast", "!", "h264parse", "!", "mpegtsmux", "!", "udpsink", f"host={host}",
               f"port={port}",
               "sync=false", "async=false"]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


COMMON_COLORS = [
    (255, 0, 0), (0, 255, 0),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (64, 64, 255), (255, 64, 64), (64, 255, 64)
]


def get_tid_color(tid, tid2color, cmap=COMMON_COLORS):
    if tid not in tid2color:
        color = cmap[len(tid2color) % len(cmap)]
        tid2color[tid] = color
    return tid2color[tid]


def display_proc_win(my_stream_id,
                     q_det2disp,
                     q_map2disp,
                     stop_evt,
                     host=None,  # 依然保留，但后面不用
                     port=None,
                     fps_exp=25.0):
    window_name = f"Stream-{my_stream_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    tid2info = {}  # {tid: (gid, score, n_tid)}

    while not stop_evt.is_set():
        # ① 更新 tid->gid 对应关系 -----------------------------------------
        try:
            m = q_map2disp.get_nowait()
            if m is SENTINEL:
                q_det2disp.put(SENTINEL)  # 通知下游
                break
            tid2info = m.get(my_stream_id, {})
        except queue.Empty:
            pass

        # ② 取一帧检测/跟踪结果 --------------------------------------------
        pkt = q_det2disp.get()
        if pkt is SENTINEL:
            break

        stream_id, fid, frame, dets, all_faces = pkt

        # ③ 绘制检测框 ------------------------------------------------------
        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            gid, score, n_tid = tid2info.get(d["id"], ("-1", -1.0, 0))
            color = (0, 255, 0)
            cv2.rectangle(frame,
                          (x, y), (x + w, y + h),
                          (0, 0, 255) if n_tid >= 2 else color,
                          2)
            cv2.putText(frame, f"{gid}",
                        (x, max(y + 15, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 1)
            cv2.putText(frame, f"n={n_tid} s={score:.2f}",
                        (x, max(y + 30, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ④ 绘制人脸框 ------------------------------------------------------
        for face in all_faces:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{face['score']:.2f}",
                        (x1, max(y1 - 2, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

            if "kps" in face:
                for kx, ky in face["kps"]:
                    kx = int(kx * SHOW_SCALE)
                    ky = int(ky * SHOW_SCALE)
                    cv2.circle(frame, (kx, ky), 1, (0, 0, 255), 2)

        # ⑤ 显示 ------------------------------------------------------------
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q 或 ESC 退出
            break

    # ⑥ 资源释放 ------------------------------------------------------------
    cv2.destroyWindow(window_name)
    print(f"[Display-{my_stream_id}] finished")


def display_proc(my_stream_id, q_det2disp, q_map2disp, stop_evt, host, port, fps_exp):
    gst, first = None, True
    tid2info = {}
    tid2color = {}
    videowriter = None

    while not stop_evt.is_set():
        try:
            m = q_map2disp.get_nowait()
            if m is SENTINEL:
                q_det2disp.put(SENTINEL)
                break
            tid2info = m.get(my_stream_id, {})
        except queue.Empty:
            pass
        pkt = q_det2disp.get()
        if pkt is SENTINEL:
            break
        stream_id, fid, frame, dets, all_faces = pkt

        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            gid, score, n_tid = tid2info.get(d["id"], ("-1", -1.0, 0))
            tid = d['id']
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255) if n_tid >= 2 else color, 2)
            cv2.putText(frame, f"{gid}", (x, max(y + 15, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 1)
            cv2.putText(frame, f"n={n_tid} s={score:.2f}", (x, max(y + 30, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for face in all_faces:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{face['score']:.2f}", (x1, max(y1 - 2, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 128, 0), 1)
            if "kps" in face:
                for kx, ky in face["kps"]:
                    kx = int(kx * SHOW_SCALE)
                    ky = int(ky * SHOW_SCALE)
                    cv2.circle(frame, (kx, ky), 1, (0, 0, 255), 2)

        if first:
            H, W = frame.shape[:2]
            gst = init_gst(W, H, fps_exp, host, port, use_nvenc=False)
            save_path = f'/home/manu/tmp/{my_stream_id}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videowriter = cv2.VideoWriter(save_path, fourcc, fps_exp, (W, H))
            first = False

        # if videowriter:
        #     videowriter.write(frame)

        if gst and gst.poll() is None:
            try:
                gst.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                break

    # 资源释放
    if gst:
        gst.stdin.close()
        gst.wait()
    if videowriter:
        videowriter.release()
    logger.info(f"[Display-{my_stream_id}] finished")


def main():
    mp.set_start_method("spawn", force=True)
    pa = argparse.ArgumentParser()
    pa.add_argument("--video1", default="rtsp://admin:1QAZ2wsx@172.20.20.64")
    pa.add_argument("--video2", default="rtsp://admin:1qaz2wsx@172.20.20.150")
    pa.add_argument("--skip", type=int, default=2)
    args = pa.parse_args()

    stop_evt = mp.Event()
    q_det2feat, q_map2disp = LatestQueue(1), LatestQueue(1)
    q_det2disp1, q_det2disp2 = LatestQueue(1), LatestQueue(1)
    procs = [
        mp.Process(target=dec_det_proc, args=("cam1", args.video1, q_det2feat, q_det2disp1, stop_evt, args.skip)),
        mp.Process(target=display_proc, args=("cam1", q_det2disp1, q_map2disp, stop_evt, "127.0.0.1", 5000, 25)),
        mp.Process(target=dec_det_proc, args=("cam2", args.video2, q_det2feat, q_det2disp2, stop_evt, args.skip)),
        mp.Process(target=display_proc, args=("cam2", q_det2disp2, q_map2disp, stop_evt, "127.0.0.1", 5001, 25)),
        mp.Process(target=feature_proc, args=(q_det2feat, q_map2disp, stop_evt))
    ]
    [p.start() for p in procs]
    signal.signal(signal.SIGINT, lambda s, f: stop_evt.set())
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(.5)
    finally:
        stop_evt.set()
        for q in (q_det2feat, q_map2disp):
            try:
                q.put_nowait(SENTINEL)
            except:
                pass
        [p.join() for p in procs]
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
