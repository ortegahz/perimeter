#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ArcFace 五点关键点，通过 solvePnP 计算人脸的 Yaw/Pitch/Roll。
功能：
1. 递归读取指定文件夹下的所有图片。
2. 使用 FaceSearcher 检测人脸及其 5 个关键点。
3. 通过 solvePnP 计算三维姿态角（Yaw/Pitch/Roll），打印到终端。
4. 可选：在图片上绘制边界框、关键点和姿态数值显示。
"""

import argparse
import glob
import math
import os

import cv2
import numpy as np

cv2.imshow("__init__", np.zeros((1, 1, 3), np.uint8))
cv2.waitKey(1)

YAW_TH = 30
PITCH_RATIO_LOWER_TH = 0.6
PITCH_RATIO_UPPER_TH = 1.0
ROLL_TH = 25
PITCH_TH = 256

try:
    from cores.featureProcessor import FaceSearcher
except ImportError:
    print("错误：无法导入 'FaceSearcher'，请检查项目结构和依赖。")
    exit(1)

# 假设的 3D 模型点（单位：mm），对应 ArcFace 五点关键点顺序：
# [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
# 数值来源可按人脸解剖比例，绝对值不影响姿态方向，只影响位移。
OBJECT_POINTS_3D = np.array([
    [-30.0, 40.0, 0.0],  # 左眼
    [30.0, 40.0, 0.0],  # 右眼
    [0.0, 20.0, 30.0],  # 鼻尖
    [-25.0, -20.0, 0.0],  # 左嘴角
    [25.0, -20.0, 0.0],  # 右嘴角
], dtype=np.float32)


def rotationMatrixToEulerAngles(R):
    """旋转矩阵转为欧拉角，返回 (pitch, yaw, roll) 单位为度。"""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])  # pitch, yaw, roll


def estimate_pose(image_size, image_pts):
    """
    使用 solvePnP 计算 Yaw 和 Roll，并根据关键点比例计算 Pitch Score。

    OpenCV ≥4.12 中 ITERATIVE 会先用 DLT 求初值，DLT 需要 ≥6 点。
    因此：
        • ≥6 点：直接 ITERATIVE
        • 4~5 点：EPNP 初值 + LM 细化
        • <4 点：返回 None

    返回: (yaw, pitch_score, roll)
    """
    h, w = image_size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # 假设无畸变

    n_points = image_pts.shape[0]
    if n_points < 4:
        return None

    # —— 先求初值 ——
    if n_points >= 6:
        success, rvec, tvec = cv2.solvePnP(
            OBJECT_POINTS_3D, image_pts,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    else:  # 4~5 点
        success, rvec, tvec = cv2.solvePnP(
            OBJECT_POINTS_3D, image_pts,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )

    if not success:
        return None

    # —— 关键：少于 6 点时用 LM 再细化一次 ——
    if n_points < 6:
        # solvePnPRefineLM 只返回 rvec, tvec
        rvec, tvec = cv2.solvePnPRefineLM(
            OBJECT_POINTS_3D, image_pts,
            camera_matrix, dist_coeffs,
            rvec, tvec
        )

    # —— rvec → 欧拉角 ——
    rot_mat, _ = cv2.Rodrigues(rvec)
    _, yaw, roll = rotationMatrixToEulerAngles(rot_mat)

    # --- 新增：根据关键点计算俯仰比例分数以替代 Pitch ---
    # 顺序: [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
    left_eye = image_pts[0]
    right_eye = image_pts[1]
    nose = image_pts[2]
    left_mouth = image_pts[3]
    right_mouth = image_pts[4]

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    eye_to_nose = np.linalg.norm(nose - eye_center)
    if eye_to_nose < 1e-6:
        pitch_score = 1.0  # 避免除以零，返回中性值
    else:
        nose_to_mouth = np.linalg.norm(mouth_center - nose)
        pitch_score = nose_to_mouth / eye_to_nose

    return yaw, pitch_score, roll


def main():
    pa = argparse.ArgumentParser(description="使用 ArcFace + solvePnP 计算人脸姿态角。")
    pa.add_argument("--image_dir", type=str, default="/home/manu/tmp/perimeter_cpp/G00005/bodies/")
    pa.add_argument("--provider", type=str, default="CPUExecutionProvider",
                    choices=["CPUExecutionProvider", "CUDAExecutionProvider"])
    pa.add_argument("--show", default=True)
    pa.add_argument("--output_file", type=str, default="/home/manu/nfs/pose_results_py.txt",
                    help="用于保存姿态估计结果（yaw, pitch, roll）的文本文件路径。")
    args = pa.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"错误：找不到目录 {args.image_dir}")
        return

    print("正在初始化人脸检测器...")
    try:
        face_app = FaceSearcher(provider=args.provider).app
        print("人脸检测器初始化完成。")
    except Exception as e:
        print(f"初始化 FaceSearcher 失败: {e}")
        return

    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True))

    # 按文件名排序，确保处理顺序一致
    image_paths.sort()

    if not image_paths:
        print(f"在 {args.image_dir} 中未找到任何图片。")
        return

    print(f"找到 {len(image_paths)} 张图片，开始处理...")
    print(f"结果将保存到: {args.output_file}")

    with open(args.output_file, 'w') as f_out:
        # 写入CSV文件头
        f_out.write("ImagePath,FaceIndex,Yaw,Pitch,Roll\n")

        for img_path in image_paths:
            print(f"\n--- 处理: {img_path} ---")
            frame = cv2.imread(img_path)
            if frame is None:
                print("  -> 无法读取图片。")
                continue

            faces_bboxes, faces_kpss = face_app.det_model.detect(frame, max_num=0, metric='default')
            if faces_kpss is None or len(faces_kpss) == 0:
                print("  -> 未检测到人脸。")
                if args.show:
                    cv2.imshow("Pose", frame)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
                continue

            h, w = frame.shape[:2]
            print(f"  -> 检测到 {len(faces_kpss)} 张人脸")
            for i, (bbox, kps) in enumerate(zip(faces_bboxes, faces_kpss)):
                yaw_pitch_roll = estimate_pose((h, w), np.array(kps, dtype=np.float32))
                if yaw_pitch_roll is None:
                    print(f"  人脸 #{i + 1}: 姿态计算失败。")
                    continue
                yaw, pitch_score, roll = yaw_pitch_roll
                print(f"  人脸 #{i + 1} 姿态: Yaw={yaw:.2f}°, Pitch_Score={pitch_score:.2f}, Roll={roll:.2f}°")

                # 将结果写入文件
                # 注意：原始代码输出格式为 pitch,yaw,roll，此处用 pitch_score 替换 pitch
                output_line = f"{os.path.basename(img_path)},{i + 1},{pitch_score:.4f},{yaw:.4f},{roll:.4f}\n"
                f_out.write(output_line)

                if args.show:
                    # --- 修改：根据姿态分数判断是否为正脸 ---
                    yaw_threshold = YAW_TH
                    roll_threshold = ROLL_TH

                    if abs(yaw) < yaw_threshold and PITCH_RATIO_LOWER_TH < pitch_score < PITCH_RATIO_UPPER_TH and abs(roll) < roll_threshold:
                        box_color = (0, 255, 0)  # 绿色 (正脸)
                        print(f"    -> 判断为: 正脸")
                    else:
                        box_color = (0, 255, 255) # 黄色 (侧脸/姿态不佳)
                        print(f"    -> 判断为: 侧脸")

                    # 绘制关键点和框
                    x1, y1, x2, y2 = bbox[:4].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    for (x, y) in kps.astype(int):
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
                    # 在框上方绘制姿态文字
                    label = f"Y:{yaw:.1f} P_score:{pitch_score:.2f} R:{roll:.1f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if args.show:
                cv2.imshow("Pose", frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break

            # 如果在显示模式下按下了 'q', 提前跳出外层循环
            if 'key' in locals() and key == ord('q'):
                 break

    if args.show:
        cv2.destroyAllWindows()

    print("\n--- 处理完成 ---")


if __name__ == "__main__":
    main()
