# FILE: demo_v7.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双路视频 + 全局GID
- 使用严格候选检测、多帧确认、防抖&锁定机制
- 第一个GID：有人脸+body -> 立即创建

- 新增功能：长时间没有更新的gid会被自动删除（包含内存和磁盘数据）
- 新增功能：入侵检测和跨线检测区域配置及可视化
- 修改：仅保留GST推流显示模式，移除本地窗口显示。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import signal
import subprocess

import cv2
import numpy as np

from tools.test_frontal_face_3d import estimate_pose, YAW_TH, ROLL_TH, PITCH_RATIO_LOWER_TH, PITCH_RATIO_UPPER_TH

# --- 常量定义 ---
SENTINEL = None
SHOW_SCALE = 0.5
PITCH_SCORE_LOWER_TH = PITCH_RATIO_LOWER_TH
PITCH_SCORE_UPPER_TH = PITCH_RATIO_UPPER_TH

PROJECTION_DEPTH = 256

cv2.imshow("__init__", np.zeros((1, 1, 3), np.uint8))
cv2.waitKey(1)

# ------------ 内部模块 ------------
from cores.byteTrackPipeline import ByteTrackPipeline
from cores.featureProcessor import *


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
    try:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"[{stream_id}] open failed: {src}")
            return
        bt = ByteTrackPipeline(device="cuda")
        face_app = FaceSearcher(provider="CUDAExecutionProvider").app
        logger.info(f"[{stream_id}] ready")
        fid = 0
        while not stop_evt.is_set():
            ok, frm = cap.read()
            if not ok:
                break
            fid += 1
            if fid % skip:
                continue
            dets = bt.update(frm, debug=False)
            small = cv2.resize(frm, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
            H, W = frm.shape[:2]
            patches = [
                frm[max(int(y), 0):min(int(y + h), H),
                max(int(x), 0):min(int(x + w), W)].copy()
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
            # === 这里保持原 q_det2disp 数据结构不变 ===
            q_det2disp.put((stream_id, fid, small, dets, face_info))
            # === 修改这里：把 face_info 和 full_frame 一起传给 feature_proc ===
            q_det2feat.put({
                "cam_id": stream_id,
                "fid": fid,
                "dets": dets,
                "full_frame": frm
            })
    finally:
        q_det2feat.put(SENTINEL)
        q_det2disp.put(SENTINEL)
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        logger.info(f"[{stream_id}] finished")


def feature_proc(q_det2feat, q_map2disp, stop_evt, boundary_config):
    processor = FeatureProcessor(device="cuda", boundary_config=boundary_config, use_fid_time=False)
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
               f"port={port}", "sync=false", "async=false"]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# Define intrusion zones and crossing lines here.
# Coordinates are based on the original full resolution of the video (e.g., 1920x1080 2560x1440).
BOUNDARY_CONFIG = {
    "cam1": {
        "intrusion_poly": [
            (50, 1400), (1200, 1400), (1100, 500), (50, 500)  # Example: bottom-left area
        ],
        "crossing_lines": [
            {
                "name": "Line_1",
                "start": (1200, 60),  # Example: a vertical line in the middle
                "end": (1400, 1280),
                "direction": "any",
                "projection_depth": PROJECTION_DEPTH  # 沿法线方向延伸的深度（像素）
            },
            {
                "name": "Line_2",
                "start": (100, 700),
                "end": (800, 700),
                "direction": "any",
                "projection_depth": PROJECTION_DEPTH
            }
        ]
    },
    "cam2": {
        "intrusion_poly": [
            (1500, 100), (1850, 100), (1850, 400), (1500, 400)  # Example: top-right area
        ]
    }
}


def draw_boundaries(frame: np.ndarray, stream_id: str, simple_display: bool = False):
    """Helper function to draw the pre-defined boundaries on the frame for debugging."""
    if simple_display:
        return
    config = BOUNDARY_CONFIG.get(stream_id)
    if not config:
        return

    # Draw intrusion polygon
    if "intrusion_poly" in config:
        poly_points = (np.array(config["intrusion_poly"]) * SHOW_SCALE).astype(np.int32)
        cv2.polylines(frame, [poly_points], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        label_pos = (poly_points[0][0], poly_points[0][1] - 10)
        cv2.putText(frame, "Intrusion Zone", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw crossing line
    if "crossing_lines" in config:
        for line_cfg in config["crossing_lines"]:
            start_pt = tuple((np.array(line_cfg["start"]) * SHOW_SCALE).astype(int))
            end_pt = tuple((np.array(line_cfg["end"]) * SHOW_SCALE).astype(int))
            depth = line_cfg.get("projection_depth", 50) * SHOW_SCALE

            # ----- 修改开始: 在线的两侧绘制投射区域 -----
            overlay = frame.copy()
            alpha_blend = 0.15  # 透明度

            v = np.array(end_pt) - np.array(start_pt)
            # 避免零向量导致除零错误
            if np.linalg.norm(v) < 1e-6:
                continue
            normal = np.array([-v[1], v[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            # 正方向区域
            p_end_pos = (np.array(end_pt) + normal * depth).astype(int)
            p_start_pos = (np.array(start_pt) + normal * depth).astype(int)
            poly_area_pos = np.array([start_pt, end_pt, p_end_pos, p_start_pos], dtype=np.int32)
            cv2.fillPoly(overlay, [poly_area_pos], color=(255, 0, 255), lineType=cv2.LINE_AA)

            # 负方向区域
            p_end_neg = (np.array(end_pt) - normal * depth).astype(int)
            p_start_neg = (np.array(start_pt) - normal * depth).astype(int)
            poly_area_neg = np.array([start_pt, end_pt, p_end_neg, p_start_neg], dtype=np.int32)
            cv2.fillPoly(overlay, [poly_area_neg], color=(255, 0, 255), lineType=cv2.LINE_AA)

            # 将带有透明区域的图层混合回原图
            cv2.addWeighted(overlay, alpha_blend, frame, 1 - alpha_blend, 0, frame)
            # ----- 修改结束 -----

            # 绘制中心线和标签
            cv2.line(frame, start_pt, end_pt, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            label_pos = (start_pt[0], start_pt[1] - 10)
            label_text = line_cfg.get("name", "Crossing Line")
            cv2.putText(frame, label_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

def display_proc(my_stream_id, q_det2disp, q_map2disp, stop_evt, host, port, fps_exp, simple_display=False):
    gst, first = None, True
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
        draw_boundaries(frame, my_stream_id, simple_display=simple_display)

        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            tid, class_name = d['id'], d.get('class_name', 'UNK')
            if d.get('class_id') == 0:
                info_str, score, n_tid = tid2info.get(tid, (f"{my_stream_id}_{tid}_-1", -1.0, 0))

                # Color logic for alarms and matches
                # if info_str.endswith("_AA") or info_str.endswith("_AL"):
                #     color = (0, 255, 255)  # Yellow for behavior alarm
                if n_tid >= 2:
                    color = (0, 0, 255)  # Red for multi-cam match
                else:
                    color = (0, 255, 0)  # Green for normal

                # --- Text display logic ---
                gid_part = None
                display_text = ""
                parts = info_str.split('_')
                for part in parts:
                    if part.startswith('G') and part[1:].isdigit():
                        gid_part = part
                        break

                if simple_display:
                    if gid_part:
                        display_text = f"{gid_part}"
                else:
                    display_text = f"{info_str}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if display_text:
                    cv2.putText(frame, display_text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if (simple_display and gid_part) or not simple_display:
                    cv2.putText(frame, f"n={n_tid} s={score:.2f}", (x, max(y + 30, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1)
            else:
                color = (255, 182, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID:{tid}", (x, max(y - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1)

        for face in all_faces:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            score = face.get("score", 0.0)

            # --- Pose Estimation Logic ---
            box_color = (255, 0, 0)  # Default blue
            pose_text = None
            if "kps" in face and face["kps"] and len(face["kps"]) == 5:
                # Original frame dimensions are needed for camera matrix
                h_orig = int(frame.shape[0] / SHOW_SCALE)
                w_orig = int(frame.shape[1] / SHOW_SCALE)

                image_pts = np.array(face["kps"], dtype=np.float32)
                yaw_pitch_roll = estimate_pose((h_orig, w_orig), image_pts)

                if yaw_pitch_roll is not None:
                    yaw, pitch_score, roll = yaw_pitch_roll
                    pose_text = f"Y:{yaw:.1f} P_score:{pitch_score:.2f} R:{roll:.1f}"

                    # Thresholds for frontal face detection
                    yaw_threshold = YAW_TH
                    roll_threshold = ROLL_TH

                    if abs(yaw) < yaw_threshold and abs(roll) < roll_threshold and \
                            PITCH_SCORE_LOWER_TH < pitch_score < PITCH_SCORE_UPPER_TH:
                        box_color = (0, 255, 0)  # Green for frontal face
                    else:
                        box_color = (0, 0, 255)  # Red for side face

            # --- Drawing ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            # Display pose text if available
            if pose_text:
                cv2.putText(frame, pose_text, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            # Always display score below the box
            cv2.putText(frame, f"S:{score:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            if "kps" in face and face["kps"]:
                for kx, ky in face['kps']:
                    cv2.circle(frame, (int(kx * SHOW_SCALE), int(ky * SHOW_SCALE)), 1, (0, 0, 255), 2)

        if first:
            H, W = frame.shape[:2]
            gst = init_gst(W, H, fps_exp, host, port, use_nvenc=False)
            first = False
        if gst and gst.poll() is None:
            try:
                gst.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                break

    if gst: gst.stdin.close(); gst.wait()
    logger.info(f"[Display-{my_stream_id}] finished")


def local_display_proc(my_stream_id, q_det2disp, q_map2disp, stop_evt, simple_display=False):
    """使用 cv2.imshow 在本地窗口中显示结果，并自动全屏"""
    tid2info = {}
    window_name = f"Display - {my_stream_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    is_fullscreen = False

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
        if pkt is SENTINEL: break
        stream_id, fid, frame, dets, all_faces = pkt
        draw_boundaries(frame, my_stream_id, simple_display=simple_display)

        for d in dets:
            x, y, w, h = [int(c * SHOW_SCALE) for c in d["tlwh"]]
            tid, class_name = d['id'], d.get('class_name', 'UNK')
            if d.get('class_id') == 0:
                info_str, score, n_tid = tid2info.get(tid, (f"{my_stream_id}_{tid}_-1", -1.0, 0))

                # Color logic for alarms and matches
                # if info_str.endswith("_AA") or info_str.endswith("_AL"):
                #     color = (0, 255, 255)  # Yellow for behavior alarm
                if n_tid >= 2:
                    color = (0, 0, 255)  # Red for multi-cam match
                else:
                    color = (0, 255, 0)  # Green for normal

                # --- Text display logic ---
                gid_part = None
                display_text = ""
                parts = info_str.split('_')
                for part in parts:
                    if part.startswith('G') and part[1:].isdigit():
                        gid_part = part
                        break

                if simple_display:
                    if gid_part:
                        display_text = f"{gid_part}"
                else:
                    display_text = f"{info_str}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if display_text:
                    cv2.putText(frame, display_text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if (simple_display and gid_part) or not simple_display:
                    cv2.putText(frame, f"n={n_tid} s={score:.2f}", (x, max(y + 30, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1)
            else:
                color = (255, 182, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID:{tid}", (x, max(y - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1)

        for face in all_faces:
            x1, y1, x2, y2 = [int(v * SHOW_SCALE) for v in face["bbox"]]
            score = face.get("score", 0.0)

            # --- Pose Estimation Logic ---
            box_color = (255, 0, 0)  # Default blue
            pose_text = None
            if "kps" in face and face["kps"] and len(face["kps"]) == 5:
                # Original frame dimensions are needed for camera matrix
                h_orig = int(frame.shape[0] / SHOW_SCALE)
                w_orig = int(frame.shape[1] / SHOW_SCALE)

                image_pts = np.array(face["kps"], dtype=np.float32)
                yaw_pitch_roll = estimate_pose((h_orig, w_orig), image_pts)

                if yaw_pitch_roll is not None:
                    yaw, pitch_score, roll = yaw_pitch_roll
                    pose_text = f"Y:{yaw:.1f} P_score:{pitch_score:.2f} R:{roll:.1f}"

                    # Thresholds for frontal face detection
                    yaw_threshold = YAW_TH
                    roll_threshold = ROLL_TH

                    if abs(yaw) < yaw_threshold and abs(roll) < roll_threshold and \
                            PITCH_SCORE_LOWER_TH < pitch_score < PITCH_SCORE_UPPER_TH:
                        box_color = (0, 255, 0)  # Green for frontal face
                    else:
                        box_color = (0, 0, 255)  # Red for side face

            # --- Drawing ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            # Display pose text if available
            if pose_text:
                cv2.putText(frame, pose_text, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            # Always display score below the box
            cv2.putText(frame, f"S:{score:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            if "kps" in face and face["kps"]:
                for kx, ky in face['kps']:
                    cv2.circle(frame, (int(kx * SHOW_SCALE), int(ky * SHOW_SCALE)), 1, (0, 0, 255), 2)

        # if not is_fullscreen:
        #     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #     is_fullscreen = True

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_evt.set()
            break

    cv2.destroyWindow(window_name)
    logger.info(f"[Display-{my_stream_id}] finished")


def main():
    mp.set_start_method("spawn", force=True)
    pa = argparse.ArgumentParser()
    pa.add_argument("--video1", default="rtsp://admin:1qaz2wsx@172.20.20.64")
    pa.add_argument("--video2", default="")
    pa.add_argument("--skip", type=int, default=2)
    pa.add_argument("--display_mode", default="local", choices=["gst", "local"],
                    help="显示模式: 'gst' 推流 或 'local' 本地窗口")
    pa.add_argument("--simple_display", default=False)
    args = pa.parse_args()

    stop_evt = mp.Event()
    q_det2feat, q_map2disp = LatestQueue(1), LatestQueue(1)
    procs = []

    if args.display_mode == 'local':
        display_target = local_display_proc
        kwargs1, kwargs2 = {"simple_display": args.simple_display}, {"simple_display": args.simple_display}
    else:  # gst
        display_target = display_proc
        kwargs1 = {"host": "127.0.0.1", "port": 5000, "fps_exp": 25, "simple_display": args.simple_display}
        kwargs2 = {"host": "127.0.0.1", "port": 5001, "fps_exp": 25, "simple_display": args.simple_display}

    # Always open video1
    q_det2disp1 = LatestQueue(1)
    procs.append(
        mp.Process(target=dec_det_proc, args=("cam1", args.video1, q_det2feat, q_det2disp1, stop_evt, args.skip)))
    procs.append(mp.Process(target=display_target, args=("cam1", q_det2disp1, q_map2disp, stop_evt), kwargs=kwargs1))

    # Only open video2 if it's not an empty string
    if args.video2:
        q_det2disp2 = LatestQueue(1)
        procs.append(
            mp.Process(target=dec_det_proc, args=("cam2", args.video2, q_det2feat, q_det2disp2, stop_evt, args.skip)))
        procs.append(
            mp.Process(target=display_target, args=("cam2", q_det2disp2, q_map2disp, stop_evt), kwargs=kwargs2))

    procs.append(mp.Process(target=feature_proc, args=(q_det2feat, q_map2disp, stop_evt, BOUNDARY_CONFIG)))
    [p.start() for p in procs]

    signal.signal(signal.SIGINT, lambda s, f: stop_evt.set())

    try:
        [p.join() for p in procs]
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received in main. Ensuring shutdown.")
    finally:
        if not stop_evt.is_set():
            stop_evt.set()
        import time
        time.sleep(1)
        for p in procs:
            if p.is_alive():
                logger.warning(f"Process {p.name} did not exit gracefully. Terminating.")
                p.terminate()
                p.join(timeout=2)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
