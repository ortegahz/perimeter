#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 is_frontal_face_2d 函数的独立脚本。
功能：
1. 递归读取指定文件夹下的所有图片。
2. 使用 FaceSearcher 检测人脸及其5个关键点。
3. 对每个检测到的人脸，使用 is_frontal_face_2d_with_debug 函数判断是否为正脸。
4. 在控制台打印详细的判断依据。
5. (可选) 使用 --show 参数可视化检测结果。
"""

import argparse
import glob
import math
import os

import cv2
import numpy as np

cv2.imshow("__init__", np.zeros((1, 1, 3), np.uint8))
cv2.waitKey(1)

# 假设 FaceSearcher 类可以从 cores.featureProcessor 模块导入
# This is based on the provided main script.
try:
    from cores.featureProcessor import FaceSearcher
except ImportError:
    print("错误：无法导入 'FaceSearcher'。请确保此脚本与您的项目结构保持一致，")
    print("并且可以访问 'cores.featureProcessor' 模块。")
    exit(1)


def is_frontal_face_2d_with_debug(kps: np.ndarray, yaw_sym_threshold=0.7, roll_angle_threshold=25.0) -> bool:
    """
    使用简单的2D几何方法判断是否为正脸，并打印判断依据。
    :param kps: 5个关键点 (左眼, 右眼, 鼻子, 左嘴角, 右嘴角) 的 numpy 数组。
    :param yaw_sym_threshold: 偏航角对称性阈值。衡量鼻子到双眼距离的对称性，越接近1要求越严格。
    :param roll_angle_threshold: 翻滚角（头部倾斜）角度阈值。
    :return: 如果是正脸则返回 True，否则返回 False。
    """
    try:
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]

        # 1. 偏航角（Yaw）检测：基于鼻子与双眼的水平距离对称性
        dist_left = nose[0] - left_eye[0]
        dist_right = right_eye[0] - nose[0]

        # 确保关键点位置合理（例如，鼻子在双眼之间）
        if dist_left <= 0 or dist_right <= 0:
            print(f"  -> [判断依据] 非正脸：关键点布局无效 (鼻子不在双眼之间)。"
                  f"左眼到鼻子水平距离: {dist_left:.1f}, 鼻子到右眼水平距离: {dist_right:.1f}")
            return False

        # 计算对称性比例
        symmetry_ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        if symmetry_ratio < yaw_sym_threshold:
            print(f"  -> [判断依据] 非正脸：偏航角(Yaw)对称性未通过。")
            print(f"     对称度: {symmetry_ratio:.2f} < 阈值: {yaw_sym_threshold}")
            return False

        # 2. 翻滚角（Roll）检测：基于双眼连线的倾斜角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        # 避免除零错误
        if abs(dx) < 1e-6:
            print("  -> [判断依据] 非正脸：双眼几乎垂直对齐 (dx 接近于 0)。")
            return False

        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) > roll_angle_threshold:
            print(f"  -> [判断依据] 非正脸：翻滚角(Roll)过大。")
            print(f"     角度: {abs(angle):.2f}° > 阈值: {roll_angle_threshold}°")
            return False

    except (IndexError, TypeError) as e:
        print(f"  -> [判断依据] 错误：无效的关键点数据。 {e}")
        return False

    print(f"  -> [判断依据] 正脸：所有检查已通过。")
    print(f"     对称度: {symmetry_ratio:.2f} >= {yaw_sym_threshold}, "
          f"翻滚角: {abs(angle):.2f}° <= {roll_angle_threshold}°")
    return True


def main():
    parser = argparse.ArgumentParser(description="在图片目录中测试人脸朝向检测功能。")
    parser.add_argument("--image_dir", type=str, default="/home/manu/tmp/perimeter_cpp/G00005/bodies/")
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider",
                        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
                        help="人脸检测模型的执行后端。")
    parser.add_argument("--show", default=True)
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"错误：找不到目录 {args.image_dir}")
        return

    print("正在初始化人脸检测器...")
    try:
        face_app = FaceSearcher(provider=args.provider).app
        print("人脸检测器初始化完成。")
    except Exception as e:
        print(f"初始化 FaceSearcher 失败: {e}")
        print("请确保 'cores.featureProcessor' 及其依赖项已正确安装。")
        return

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))

    if not image_paths:
        print(f"在 {args.image_dir} 中未找到任何图片。")
        return

    print(f"找到 {len(image_paths)} 张图片。开始处理...")

    for img_path in image_paths:
        print(f"\n--- 正在处理: {img_path} ---")
        frame = cv2.imread(img_path)
        if frame is None:
            print("  -> 无法读取图片。")
            continue

        faces_bboxes, faces_kpss = face_app.det_model.detect(frame, max_num=0, metric='default')

        if faces_kpss is None or len(faces_kpss) == 0:
            print("  -> 未检测到人脸。")
            if args.show:
                cv2.imshow("Result", frame)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            continue

        print(f"  -> 检测到 {len(faces_kpss)} 张人脸。")
        for i, (bbox, kps) in enumerate(zip(faces_bboxes, faces_kpss)):
            print(f"  - 人脸 #{i + 1}:")
            is_frontal = is_frontal_face_2d_with_debug(kps)

            if args.show:
                # 绘制边界框和结果文本
                x1, y1, x2, y2 = bbox[:4].astype(int)
                color = (0, 255, 0) if is_frontal else (0, 0, 255)
                label = "Frontal" if is_frontal else "Side Face"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                # 绘制关键点
                for (x, y) in kps.astype(int):
                    cv2.circle(frame, (x, y), 3, (255, 128, 0), -1, cv2.LINE_AA)

        if args.show:
            cv2.imshow("Result", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()
    print("\n--- 所有图片处理完毕 ---")


if __name__ == '__main__':
    main()
