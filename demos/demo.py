# executor.py
# --------------------------------------------------
# 依赖：
#   - Predictor         (来自 cores.predictor)
#   - PersonReid        (来自 cores.personReid)
#   - FaceSearcher      (本文第二段代码)
#   - BYTETracker       (yolox/tracker/byte_tracker.py)
# --------------------------------------------------
from __future__ import annotations

import os
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils.visualize import plot_tracking

from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from cores.predictor import Predictor
from utils_peri.macros import DIR_BYTE_TRACK


# --------------------------------------------------
# Ⅰ. Helper：TrackFeatureAggregator
# --------------------------------------------------
class TrackFeatureAggregator:
    """
    对单条 TrackID 持有一个滑动窗口，负责：
        1) 每隔 k 帧抽取一次行人特征
        2) 维护若干张质量最高的脸特征
        3) 在需要时输出一个 TrackFeature
    """

    def __init__(self, maxlen: int = 10):
        self.body_feats: deque = deque(maxlen=maxlen)  # [(feat, score)]
        self.face_feats: deque = deque(maxlen=3)  # 较少即可
        self.last_update_frame: int = -1

    # -----------------------
    def add_body(self, feat: np.ndarray, q: float, frame_id: int):
        self.body_feats.append((feat, q))
        self.last_update_frame = frame_id

    def add_face(self, feat: np.ndarray, frame_id: int):
        self.face_feats.append(feat)
        self.last_update_frame = frame_id

    # -----------------------
    def get_body_feature(self) -> Optional[np.ndarray]:
        """
        质量加权平均
        """
        if not self.body_feats:
            return None
        feats, scores = zip(*self.body_feats)
        w = np.array(scores, dtype=np.float32)
        w = np.clip(w, 1e-2, None)
        w = w / w.sum()
        rep = (np.stack(feats, 0) * w[:, None]).sum(0)
        rep /= np.linalg.norm(rep) + 1e-9
        return rep.astype(np.float32)

    def get_face_feature(self) -> Optional[np.ndarray]:
        if not self.face_feats:
            return None
        # 直接取均值即可
        rep = np.mean(np.stack(self.face_feats, 0), axis=0)
        rep /= np.linalg.norm(rep) + 1e-9
        return rep.astype(np.float32)

    def is_stale(self, frame_id: int, max_gap: int = 60) -> bool:
        """
        当前帧 - 上次更新帧 ≥ max_gap 视为该 track 已结束
        """
        return (frame_id - self.last_update_frame) >= max_gap


# --------------------------------------------------
# Ⅱ. GlobalIDManager
# --------------------------------------------------
class GlobalIDManager:
    """
    维护全局身份库：
        global_id -> {"faces": [512-D], "bodies": [256-D], "last_seen": ts, "protos": ...}
    """

    def __init__(self,
                 face_thr: float = 0.45,
                 body_thr: float = 0.27,
                 max_proto: int = 5):
        self.face_thr = face_thr
        self.body_thr = body_thr
        self.max_proto = max_proto

        self._store: Dict[str, Dict] = {}  # global_id: dict

    # ----------------------------- 公开接口 -----------------------------
    def match(self,
              face_feat: Optional[np.ndarray],
              body_feat: np.ndarray) -> str:
        """
        返回匹配到的 global_id（若无则自动新建）
        """
        # 1) 先用人脸
        if face_feat is not None:
            gid = self._match_by_face(face_feat)
            if gid:
                self._update_global(gid, face_feat, body_feat)
                return gid

        # 2) 再用行人特征
        gid = self._match_by_body(body_feat)
        if gid:
            self._update_global(gid, face_feat, body_feat)
            return gid

        # 3) 没匹配到 → 新建
        gid = self._new_global(face_feat, body_feat)
        return gid

    # ----------------------------- 内部实现 -----------------------------
    def _match_by_face(self, face_feat: np.ndarray) -> Optional[str]:
        best_gid, best_sim = None, -1
        for gid, item in self._store.items():
            for proto in item["faces"]:
                sim = float(face_feat @ proto)
                if sim > best_sim:
                    best_gid, best_sim = gid, sim
        if best_sim >= self.face_thr:
            return best_gid
        return None

    def _match_by_body(self, body_feat: np.ndarray) -> Optional[str]:
        best_gid, best_sim = None, -1
        for gid, item in self._store.items():
            for proto in item["bodies"]:
                sim = float(body_feat @ proto)
                if sim > best_sim:
                    best_gid, best_sim = gid, sim
        if best_sim >= self.body_thr:
            return best_gid
        return None

    def _update_global(self,
                       gid: str,
                       face_feat: Optional[np.ndarray],
                       body_feat: Optional[np.ndarray]):
        entry = self._store[gid]
        if face_feat is not None:
            self._add_proto(entry["faces"], face_feat)
        if body_feat is not None:
            self._add_proto(entry["bodies"], body_feat)
        entry["last_seen"] = time.time()

    def _new_global(self,
                    face_feat: Optional[np.ndarray],
                    body_feat: Optional[np.ndarray]) -> str:
        gid = uuid.uuid4().hex[:16]
        self._store[gid] = dict(
            faces=[],
            bodies=[],
            first_seen=time.time(),
            last_seen=time.time()
        )
        self._update_global(gid, face_feat, body_feat)
        logger.info(f"[NEW] global_id={gid[:8]} created")
        return gid

    def _add_proto(self, proto_list: List[np.ndarray], feat: np.ndarray):
        if len(proto_list) < self.max_proto:
            proto_list.append(feat)
        else:
            # 简单替换：找距离最近的更新
            dists = [(feat @ p) for p in proto_list]  # 余弦相似度
            idx = int(np.argmax(dists))
            proto_list[idx] = 0.7 * proto_list[idx] + 0.3 * feat
            proto_list[idx] /= np.linalg.norm(proto_list[idx]) + 1e-9


# --------------------------------------------------
# Ⅲ. PerimeterExecutor
# --------------------------------------------------
class PerimeterExecutor:
    """
    组装整个周界安全算法的执行器
    """

    def __init__(self,
                 predictor: Predictor,
                 tracker_args,
                 reid_model_dir: str,
                 face_cache: str):
        # 1) 检测 + 跟踪
        self.predictor = predictor
        self.tracker = BYTETracker(tracker_args, frame_rate=30)

        # 2) 行人 ReID
        self.reid = PersonReid(reid_model_dir, which_epoch='last', gpu='0')

        # 3) 人脸
        self.face_app = FaceSearcher(
            provider="CPUExecutionProvider",
            cache_path=face_cache
        ).app  # 直接用 insightface.FaceAnalysis 实例

        # 4) 全局 ID 管理器
        self.gid_mgr = GlobalIDManager()

        # 5) TrackID -> TrackFeatureAggregator
        self.track_feat_pool: Dict[int, TrackFeatureAggregator] = {}

        # 6) ROI、徘徊参数
        self.loiter_time_thr = 90  # 秒
        self.speed_thr = 0.3  # m/s
        self.track_meta = {}  # track_id -> {'enter_ts':, 'enter_pos':}

    # --------------------------------------------------
    def run(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        timer = Timer()
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % 3 != 0:
                continue

            # ---------- A. 检测与跟踪 ----------
            outputs, img_info = self.predictor.inference(frame, timer)
            height, width = img_info['height'], img_info['width']
            online_targets = self.tracker.update(
                outputs[0], [height, width], self.predictor.exp.test_size
            )

            timer.toc()

            # ---------- B. 遍历所有 Track ----------
            for t in online_targets:
                tlwh = t.tlwh
                track_id = t.track_id
                # 初始化 track 容器
                if track_id not in self.track_feat_pool:
                    self.track_feat_pool[track_id] = TrackFeatureAggregator()

                # ---------- B1. Crop 行人并提取 ReID 特征 ----------
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                body_feat = self.reid.extract_feat(crop)  # (256,)
                conf = float(t.score)
                self.track_feat_pool[track_id].add_body(body_feat, conf, frame_id)

                # ---------- B2. 尝试提取人脸 ----------
                faces = self.face_app.get(crop)
                if faces:
                    face_feat = faces[0].embedding.astype(np.float32)
                    face_feat /= np.linalg.norm(face_feat) + 1e-9
                    self.track_feat_pool[track_id].add_face(face_feat, frame_id)

                # ---------- B3. Loitering enter 判定 ----------
                # 这里简单示例：第一帧在 ROI 就算进入
                if track_id not in self.track_meta:
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    if self._inside_roi(cx, cy):
                        self.track_meta[track_id] = dict(
                            enter_ts=time.time(),
                            enter_pos=(cx, cy)
                        )

            # ---------- C. 检查结束的 Track ----------
            self._flush_stale_tracks(frame_id)

            # ---------- D. 可视化 ----------
            online_im = plot_tracking(frame, [t.tlwh for t in online_targets],
                                      [t.track_id for t in online_targets],
                                      frame_id=frame_id,
                                      fps=1. / max(1e-5, timer.average_time))
            cv2.imshow("perimeter", online_im)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # --------------------------------------------------
    def _flush_stale_tracks(self, cur_frame: int):
        stale_ids = [tid for tid, agg in self.track_feat_pool.items()
                     if agg.is_stale(cur_frame)]
        for tid in stale_ids:
            agg = self.track_feat_pool.pop(tid)
            face_feat = agg.get_face_feature()
            body_feat = agg.get_body_feature()
            if body_feat is None:
                continue

            gid = self.gid_mgr.match(face_feat, body_feat)
            logger.info(f"Track {tid} → GlobalID {gid[:8]}")

            # ---------- loiter check ----------
            if tid in self.track_meta:
                enter_ts = self.track_meta[tid]['enter_ts']
                stay_time = time.time() - enter_ts
                if stay_time >= self.loiter_time_thr:
                    logger.warning(f"[ALERT] GlobalID={gid[:8]} loitering "
                                   f"{stay_time:.1f}s")
                self.track_meta.pop(tid, None)

    # --------------------------------------------------
    # 你自己的 ROI 定义（示例：全画面都算 ROI）
    def _inside_roi(self, x: float, y: float) -> bool:
        return True


# --------------------------------------------------
# Ⅳ. 运行入口
# --------------------------------------------------
if __name__ == "__main__":
    """
    示例：
    $ python executor.py --video /path/to/file.mp4
    """
    import argparse
    from yolox.exp import get_exp

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="rtsp://admin:1QAZ2wsx@172.20.20.64")
    parser.add_argument("--exp_file",
                        default=os.path.join(DIR_BYTE_TRACK, "exps/example/mot/yolox_s_mix_det.py"))
    parser.add_argument("--ckpt",
                        default=os.path.join(DIR_BYTE_TRACK, "pretrained/bytetrack_s_mot17.pth.tar"))
    args = parser.parse_args()

    # 1. YOLOX 实验文件
    exp = get_exp(args.exp_file, None)
    model = exp.get_model()

    predictor = Predictor(model, exp, trt_file=None,
                          decoder=None, device="cuda", fp16=False)

    # 2. executor
    executor = PerimeterExecutor(
        predictor=predictor,
        tracker_args=argparse.Namespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            aspect_ratio_thresh=1.6,
            min_box_area=10,
            mot20=False
        ),
        reid_model_dir=os.path.join("person_reid/model/ft_ResNet50"),
        face_cache="face_db_cache.pkl"
    )
    executor.run(args.video)
