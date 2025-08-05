import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
img_path = "/home/manu/图片/vlcsnap-2025-08-01-18h52m04s342.png"

# 推理
results = model.predict(img_path, device=0, verbose=False)

for r in results:
    # 1. 找到 "person" 的类别 id
    person_id = [k for k, v in r.names.items() if v == 'person'][0]

    # 2. 用布尔索引保留 person 框
    mask = (r.boxes.cls.int() == person_id)  # True/False 向量
    r.boxes = r.boxes[mask]  # 只剩 person

    # 3. 绘制并显示
    im = r.plot()  # BGR -> RGB
    plt.imshow(im[..., ::-1])
    plt.axis('off')
    plt.show()
