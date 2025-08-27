#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双路视频 + 全局GID
- 使用严格候选检测、多帧确认、防抖&锁定机制
- 第一个GID：有人脸+body -> 立即创建

- 新增功能：长时间没有更新的gid会被自动删除（包含内存和磁盘数据）
- 新增功能：入侵检测和跨线检测区域配置及可视化
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import signal
import subprocess

# ###-NEW-### Import necessary libraries for drawing

# --- 假装导入了这些模块，以使得代码能独立运行 ---
# (As per your request, I'm assuming these are available in your project)
SENTINEL = None
SHOW_SCALE = 0.5
# ---------------------------------------------

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
    try:  # Robustness: ensure sentinel is sent on error
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"[{stream_id}] open failed: {src}")
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
    finally:
        q_det2feat.put(SENTINEL)
        q_det2disp.put(SENTINEL)
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        logger.info(f"[{stream_id}] finished")


def feature_proc(q_det2feat, q_map2disp, stop_evt, boundary_config):  # ###-MODIFIED-### Added boundary_config parameter
    # ###-MODIFIED-### Pass the config to the FeatureProcessor
    processor = FeatureProcessor(device="cuda", boundary_config=boundary_config)
    while not stop_evt.is_set():
        pkt = q_det2feat.get()
        if pkt is SENTINEL:
            break
        realtime_map = processor.process_packet(pkt)
        q_map2disp.put(realtime_map)
    q_map2disp.put(SENTINEL)
    logger.info("[Feature] finished")


def init_gst(W, H, fps, host, port, use_nvenc=True):
    # (Function unchanged)
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


# ###-NEW-###
# Define intrusion zones and crossing lines here.
# Coordinates are based on the original full resolution of the video (e.g., 1920x1080 2560x1440).
BOUNDARY_CONFIG = {
    "cam1": {
        "intrusion_poly": [
            (50, 1400), (1200, 1400), (1100, 500), (50, 500)  # Example: bottom-left area
        ],
        "crossing_line": {
            "start": (1200, 60),  # Example: a vertical line in the middle
            "end": (1400, 1280),
            "direction": "any"
        }
    },
    "cam2": {
        "intrusion_poly": [
            (1500, 100), (1850, 100), (1850, 400), (1500, 400)  # Example: top-right area
        ]
    }
}


# ###-NEW-###
def draw_boundaries(frame, stream_id):
    """Helper function to draw the pre-defined boundaries on the frame for debugging."""
    config = BOUNDARY_CONFIG.get(stream_id)
    if not config:
        return

    # Draw intrusion polygon
    if "intrusion_poly" in config:
        # Scale points to the display size
        poly_points = (np.array(config["intrusion_poly"]) * SHOW_SCALE).astype(np.int32)
        cv2.polylines(frame, [poly_points], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        label_pos = (poly_points[0][0], poly_points[0][1] - 10)
        cv2.putText(frame, "Intrusion Zone", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw crossing line
    if "crossing_line" in config:
        # Scale points to the display size
        start_pt = tuple((np.array(config["crossing_line"]["start"]) * SHOW_SCALE).astype(int))
        end_pt = tuple((np.array(config["crossing_line"]["end"]) * SHOW_SCALE).astype(int))
        cv2.line(frame, start_pt, end_pt, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        label_pos = (start_pt[0], start_pt[1] - 10)
        cv2.putText(frame, "Crossing Line", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def display_proc_win(my_stream_id,
                     q_det2disp,
                     q_map2disp,
                     stop_evt,
                     host=None, port=None, fps_exp=25.0):
    window_name = f"Stream-{my_stream_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    tid2info = {}
    while not stop_evt.is_set():
        try:
            m = q_map2disp.get_nowait()
            if m is SENTINEL: q_det2disp.put(SENTINEL); break
            tid2info = m.get(my_stream_id, {})
        except queue.Empty:
            pass

        pkt = q_det2disp.get()
        if pkt is SENTINEL: break
        stream_id, fid, frame, dets, all_faces = pkt

        # ###-MODIFIED-### Draw the defined boundaries on the frame.
        draw_boundaries(frame, my_stream_id)

        # Drawing logic for detections and faces remains the same...
        for d in dets:
            # Your original code used SHOW_SCALE on tlwh, which is correct since tlwh is full-res
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            tid, class_name = d['id'], d.get('class_name', 'UNK')
            if d.get('class_id') == 0:
                gid, score, n_tid = tid2info.get(tid, ("-1", -1.0, 0))
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255) if n_tid >= 2 else color, 2)
                cv2.putText(frame, f"{gid} [{class_name}]", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                            2)
                cv2.putText(frame, f"n={n_tid} s={score:.2f}", (x, max(y + 30, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1)
            else:
                color = (255, 182, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID:{tid} [{class_name}]", (x, max(y - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1)

        for face in all_faces:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{face['score']:.2f}", (x1, max(y1 - 2, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 128, 0), 1)
            if "kps" in face and face["kps"]:
                for kx, ky in face["kps"]:
                    cv2.circle(frame, (int(kx * SHOW_SCALE), int(ky * SHOW_SCALE)), 1, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            stop_evt.set()  # Set the event to signal all processes to stop
            break
    cv2.destroyWindow(window_name)
    logger.info(f"[Display-{my_stream_id}] finished")


def display_proc(my_stream_id, q_det2disp, q_map2disp, stop_evt, host, port, fps_exp):
    gst, first = None, True
    tid2info = {}
    videowriter = None
    while not stop_evt.is_set():
        try:
            m = q_map2disp.get_nowait()
            if m is SENTINEL: q_det2disp.put(SENTINEL); break
            tid2info = m.get(my_stream_id, {})
        except queue.Empty:
            pass

        pkt = q_det2disp.get()
        if pkt is SENTINEL: break
        stream_id, fid, frame, dets, all_faces = pkt

        # ###-MODIFIED-### Draw the defined boundaries on the frame.
        draw_boundaries(frame, my_stream_id)

        # Drawing logic remains the same...
        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            tid, class_name = d['id'], d.get('class_name', 'UNK')
            if d.get('class_id') == 0:
                gid, score, n_tid = tid2info.get(tid, ("-1", -1.0, 0))
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255) if n_tid >= 2 else color, 2)
                cv2.putText(frame, f"{gid} [{class_name}]", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                            2)
                cv2.putText(frame, f"n={n_tid} s={score:.2f}", (x, max(y + 30, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1)
            else:
                color = (255, 182, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID:{tid} [{class_name}]", (x, max(y - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1)

        for face in all_faces:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if "kps" in face and face["kps"]:
                for kx, ky in face['kps']:
                    cv2.circle(frame, (int(kx * SHOW_SCALE), int(ky * SHOW_SCALE)), 1, (0, 0, 255), 2)

        if first:
            H, W = frame.shape[:2]
            gst = init_gst(W, H, fps_exp, host, port, use_nvenc=False)
            # Your videowriter logic...
            first = False
        if gst and gst.poll() is None:
            try:
                gst.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                break

    if gst: gst.stdin.close(); gst.wait()
    logger.info(f"[Display-{my_stream_id}] finished")


# Your utility functions (get_tid_color, etc) can remain if needed. I removed them for brevity as they were not used in the modified display functions.

def main():
    mp.set_start_method("spawn", force=True)
    pa = argparse.ArgumentParser()
    pa.add_argument("--video1", default="rtsp://admin:1QAZ2wsx@172.20.20.64")
    pa.add_argument("--video2", default="rtsp://admin:1qaz2wsx@172.20.20.150")
    pa.add_argument("--skip", type=int, default=2)
    # ###-NEW-### Add argument to switch between local window and GStreamer
    pa.add_argument("--display", choices=["local", "gst"], default="gst", help="Display method")
    args = pa.parse_args()

    stop_evt = mp.Event()
    q_det2feat, q_map2disp = LatestQueue(1), LatestQueue(1)
    q_det2disp1, q_det2disp2 = LatestQueue(1), LatestQueue(1)

    # ###-MODIFIED-### Choose display function based on argument
    if args.display == "local":
        display_func = display_proc_win
    else:
        display_func = display_proc

    procs = [
        mp.Process(target=dec_det_proc, args=("cam1", args.video1, q_det2feat, q_det2disp1, stop_evt, args.skip)),
        mp.Process(target=display_func, args=("cam1", q_det2disp1, q_map2disp, stop_evt),
                   kwargs={"host": "127.0.0.1", "port": 5000, "fps_exp": 25}),
        mp.Process(target=dec_det_proc, args=("cam2", args.video2, q_det2feat, q_det2disp2, stop_evt, args.skip)),
        mp.Process(target=display_func, args=("cam2", q_det2disp2, q_map2disp, stop_evt),
                   kwargs={"host": "127.0.0.1", "port": 5001, "fps_exp": 25}),
        # ###-MODIFIED-### Pass the BOUNDARY_CONFIG dictionary to the feature_proc process
        mp.Process(target=feature_proc, args=(q_det2feat, q_map2disp, stop_evt, BOUNDARY_CONFIG))
    ]
    [p.start() for p in procs]

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: stop_evt.set())

    try:
        # Wait for processes to complete
        [p.join() for p in procs]
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received in main. Ensuring shutdown.")
    finally:
        # Final cleanup
        if not stop_evt.is_set():
            stop_evt.set()

        # Give processes a moment to exit after event is set
        time.sleep(1)

        for p in procs:
            if p.is_alive():
                logger.warning(f"Process {p.name} did not exit gracefully. Terminating.")
                p.terminate()
                p.join(timeout=2)

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
