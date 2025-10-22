import cv2
import numpy as np

cv2.imshow("__init__", np.zeros((1, 1, 3), np.uint8))
cv2.waitKey(1)

from ultralytics import YOLO

# 1. 模型、RTSP 地址
model = YOLO("yolo11n.pt")  # 或者你的自训练权重
rtsp_url = "rtsp://admin:admin123@172.20.20.207:554/cam/realmonitor?channel=1&subtype=0"

# 2. 打开 RTSP 流
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # 建议加 CAP_FFMPEG 以提高兼容性
if not cap.isOpened():
    raise RuntimeError(f"无法打开 RTSP 流: {rtsp_url}")

# 3. 主循环
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  RTSP 取帧失败，尝试重连...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cv2.waitKey(200)  # 稍作等待，以免死循环占满 CPU
        continue

    # 4. YOLO 推理 (设置 stream=False, 因为我们手动读帧)
    results = model.predict(frame)

    # 5. 仅保留 person 类别并绘制
    for r in results:
        # a. 获取 person 类别 id
        try:
            person_id = [k for k, v in r.names.items() if v == "person"][0]
        except IndexError:
            person_id = None  # 当前模型不含 person 类别

        if person_id is not None:
            # b. 用布尔索引保留框
            mask = (r.boxes.cls.int() == person_id)
            r.boxes = r.boxes[mask]

        # c. 绘制 (r.plot 会在拷贝后的 BGR 图像上画框)
        im = r.plot()

        # d. 显示
        cv2.imshow("Pedestrian Detection (press ESC to quit)", im)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
            cap.release()
            cv2.destroyAllWindows()
            exit()

# 6. 结束清理（正常情况下不会走到这里，ESC 已经退出）
cap.release()
cv2.destroyAllWindows()
