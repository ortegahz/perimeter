# FILE: featureProcessor.py

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
# MODIFIED HERE: 导入 Face 对象
from insightface.app.common import Face
from shapely.geometry import Polygon
from loguru import logger

from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
from utils_peri.general_funcs import make_dirs
from utils_peri.macros import DIR_REID_MODEL

# ===============================================================
# 常量与目录 (保持不变)
# ===============================================================
SHOW_SCALE = 0.5
MIN_HW_RATIO = 1.5
MIN_BODY4GID = 8
MIN_FACE4GID = 8

SENTINEL = None

MATCH_THR = 0.5
THR_NEW_GID = 0.3
FACE_DET_MIN_SCORE = 0.40  # 0.9 for face only mode && 0.8 for mix mode

SAVE_DIR = "/home/manu/tmp/perimeter"
ALARM_DIR = "/home/manu/tmp/perimeter_alarm"
make_dirs(SAVE_DIR, reset=True)
make_dirs(ALARM_DIR, reset=True)

# -------- GID 更新 / 新建参数 ----------
UPDATE_THR = 0.65
FACE_THR_STRICT = 0.5
BODY_THR_STRICT = 0.4
NEW_GID_MIN_FRAMES = 3
NEW_GID_TIME_WINDOW = 50
BIND_LOCK_FRAMES = 15
CANDIDATE_FRAMES = 2
MAX_TID_GAP = 256  # 帧
GID_MAX_IDLE = 25 / 2 * 60 * 60 * 24  # 帧
WAIT_FRAMES_AMBIGUOUS = 10

FPS = 25 / 2
MAX_TID_IDLE_SEC = MAX_TID_GAP / FPS
GID_MAX_IDLE_SEC = GID_MAX_IDLE / FPS

# --------------- 计时基准开关 ----------------
TIME_BY_FRAME = True
MAX_TID_IDLE_FRAMES = MAX_TID_GAP
GID_MAX_IDLE_FRAMES = int(GID_MAX_IDLE)
# ---------------------------------------------

# -------- 报警去重相关参数 -----------------
ALARM_CNT_TH = 8
ALARM_DUP_THR = 0.4
EMB_FACE_DIM, EMB_BODY_DIM = 512, 2048
BEHAVIOR_ALARM_DURATION_FRAMES = 256


# ===============================================================
# ========== 新增：入侵和穿越检测模块 (辅助类) ==========
# ===============================================================
def get_foot_point(tlwh: list | tuple) -> tuple[int, int]:
    """获取边界框的底边中心点"""
    x, y, w, h = tlwh
    return int(x + w / 2), int(y + h)


def is_inside_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    """判断点是否在多边形内部"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


class IntrusionDetector:
    """区域入侵检测器"""

    def __init__(self, boundary_poly: list[tuple[int, int]]):
        if not boundary_poly or len(boundary_poly) < 3:
            self.boundary = None
            logger.warning("Intrusion boundary is not a valid polygon. Detector disabled.")
        else:
            self.boundary = np.array(boundary_poly, dtype=np.int32)
        self.track_history = {}
        self.alarmed_tids = set()

    def check(self, dets: list[dict], stream_id: str) -> set[int]:
        if self.boundary is None: return set()
        newly_alarmed_tids = set()
        current_tids = {d['id'] for d in dets}
        for d in dets:
            tid = d['id']
            if tid in self.alarmed_tids: continue
            current_point = get_foot_point(d['tlwh'])
            last_point = self.track_history.get(tid)
            self.track_history[tid] = current_point
            if last_point is None: continue
            if not is_inside_polygon(last_point, self.boundary) and is_inside_polygon(current_point, self.boundary):
                logger.warning(f"[ALARM][{stream_id}] Intrusion Detected! TID:{tid} entered the area.")
                self.alarmed_tids.add(tid)
                newly_alarmed_tids.add(tid)

        disappeared_tids = set(self.track_history.keys()) - current_tids
        for tid in disappeared_tids:
            self.track_history.pop(tid, None)
            self.alarmed_tids.discard(tid)
        return newly_alarmed_tids


class LineCrossingDetectorPlus:
    """增强版线条穿越检测器，包含方向和深度计算"""

    def __init__(self, line_start: tuple[int, int], line_end: tuple[int, int],
                 direction: str = 'any', projection_depth: int = 50,
                 min_intersection_area: int = 100):
        self._p1 = np.array(line_start, dtype=np.float32)
        self._p2 = np.array(line_end, dtype=np.float32)
        self._direction_policy = direction
        self._projection_depth = projection_depth
        self._min_intersection_area = min_intersection_area  # 新增：最小相交面积阈值

        # 预计算线方程 ax + by + c = 0
        self._a = self._p1[1] - self._p2[1]
        self._b = self._p2[0] - self._p1[0]
        self._c = -self._a * self._p1[0] - self._b * self._p1[1]
        norm = math.sqrt(self._a ** 2 + self._b ** 2) + 1e-6
        self._a /= norm
        self._b /= norm
        self._c /= norm

        # 预计算法向量 (单位向量)
        self._normal_vector = np.array([self._a, self._b], dtype=np.float32)

        self._track_history = {}  # {tid: {"last_point": pt, "last_side": side, "has_alarmed": False}}

    def _get_side(self, point: np.ndarray) -> int:
        """计算点在线的哪一侧"""
        val = self._a * point[0] + self._b * point[1] + self._c
        if val > 1e-3: return 1
        if val < -1e-3: return -1
        return 0

    @staticmethod
    def _polygon_intersect_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
        """计算两个凸多边形的相交面积"""
        try:
            # 使用 shapely 库，因为它对几何计算更稳健且兼容性好
            poly1_geom = Polygon(poly1)
            poly2_geom = Polygon(poly2)
            if not poly1_geom.is_valid or not poly2_geom.is_valid:
                return 0.0
            return poly1_geom.intersection(poly2_geom).area
        except Exception:
            # 如果多边形创建失败（例如，点少于3个），返回0
            return 0.0

    def check(self, dets: list[dict], stream_id: str) -> dict[int, dict]:
        """
        检查检测结果，返回一个字典，包含触发警报的TID及其详细越界信息
        - key: tid (int)
        - value: {"direction": str, "distance": float, "ratio": float}
        """
        alarmed_tracks = {}
        current_tids_int = {d['id'] for d in dets}

        for d in dets:
            tid_int = d['id']
            history = self._track_history.setdefault(tid_int, {"last_point": None, "last_side": None, "has_alarmed": False})

            # If this track has already triggered an alarm, skip it.
            if history.get("has_alarmed", False):
                continue

            x, y, w, h = d['tlwh']
            bbox_poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            current_point = np.array(get_foot_point(d['tlwh']), dtype=np.float32)
            bbox_area = w * h

            # --- 1. 越界触发判断 ---
            # 条件A: 轨迹跨线
            current_side = self._get_side(current_point)
            last_side = history.get('last_side')
            trajectory_crossed = (last_side is not None and current_side != last_side and
                                  last_side != 0 and current_side != 0)

            # 条件B: BBox与线相交
            vertex_sides = [self._get_side(v) for v in bbox_poly]
            bbox_intersects = (1 in vertex_sides and -1 in vertex_sides)

            if not (trajectory_crossed or bbox_intersects):
                history.update({"last_point": current_point, "last_side": current_side})
                continue

            # --- 2. 越界方向判断 ---
            crossing_direction = 0  # 1 for positive, -1 for negative
            if trajectory_crossed:
                crossing_direction = current_side

                # 策略检查
                is_in = last_side < 0 and current_side > 0
                is_out = last_side > 0 and current_side < 0
                if self._direction_policy == 'in' and not is_in: continue
                if self._direction_policy == 'out' and not is_out: continue

            elif bbox_intersects and history["last_point"] is not None:
                # 改进：当仅BBox相交时，使用目标的实际运动轨迹来判断方向
                motion_vector = current_point - history["last_point"]
                # 将运动向量投影到线的法向量上，判断方向
                direction_sign = np.dot(motion_vector, self._normal_vector)

                if abs(direction_sign) > 1e-6:  # 避免运动方向与线平行的情况
                    # 同样应用方向策略
                    is_in = direction_sign > 0
                    is_out = direction_sign < 0
                    if self._direction_policy == 'in' and not is_in: continue
                    if self._direction_policy == 'out' and not is_out: continue
                    crossing_direction = 1 if is_in else -1

            if crossing_direction == 0: continue

            # --- 3. 越界深度计算 ---
            # a) 法向距离
            vec_p_c = current_point - self._p1
            perpendicular_dist = np.dot(vec_p_c, self._normal_vector)

            # b) 面积比例
            proj_dir = self._normal_vector if crossing_direction > 0 else -self._normal_vector
            p1_proj = self._p1 + proj_dir * self._projection_depth
            p2_proj = self._p2 + proj_dir * self._projection_depth
            crossing_zone_poly = np.array([self._p1, self._p2, p2_proj, p1_proj], dtype=np.float32)

            # ----- 修改：同时计算相交区域的多边形顶点 -----
            intersection_area = 0.0
            intersection_poly = None
            try:
                poly1_geom, poly2_geom = Polygon(bbox_poly), Polygon(crossing_zone_poly)
                if poly1_geom.is_valid and poly2_geom.is_valid:
                    intersection_geom = poly1_geom.intersection(poly2_geom)
                    intersection_area = intersection_geom.area
                    if not intersection_geom.is_empty:
                        intersection_poly = np.array(intersection_geom.exterior.coords)
            except Exception:
                pass  # 保持默认值
            overlap_ratio = intersection_area / (bbox_area + 1e-6)

            # --- 4. 报警有效性验证 & 输出 ---
            # 新增：仅当相交面积满足阈值时才触发报警，此举可有效过滤抖动
            if intersection_area >= self._min_intersection_area:
                dir_str = "in" if crossing_direction > 0 else "out"
                logger.warning(
                    f"[ALARM][{stream_id}] Line Crossing! TID:{tid_int}, Dir:{dir_str}, "
                    f"Dist:{perpendicular_dist:.1f}, Ratio:{overlap_ratio:.2f}, Area:{intersection_area:.0f}"
                )
                alarmed_tracks[tid_int] = {
                    "direction": dir_str,
                    "distance": perpendicular_dist,
                    "ratio": overlap_ratio,
                    "area": intersection_area,
                    "crossing_zone_poly": crossing_zone_poly.tolist(),
                    "intersection_poly": intersection_poly.tolist() if intersection_poly is not None else None,
                    "line_start": self._p1.tolist(),
                    "line_end": self._p2.tolist(),
                    "projection_vector": proj_dir.tolist()
                }
                history["has_alarmed"] = True  # Mark as alarmed to prevent re-triggering

            # 无论是否报警，都更新历史状态，为下一帧方向判断提供稳定依据
            history.update({"last_point": current_point, "last_side": current_side})

        # 清理消失的track
        disappeared_tids = set(self._track_history.keys()) - current_tids_int
        for tid in disappeared_tids:
            self._track_history.pop(tid, None)

        return alarmed_tracks


# ===============================================================

def is_long_patch(patch: np.ndarray, thr=MIN_HW_RATIO):
    if patch is None or patch.size == 0:
        return False
    h, w = patch.shape[:2]
    return h / (w + 1e-9) >= thr


# ===============================================================
# TrackAgg (保持不变)
# ===============================================================
class TrackAgg:
    """
    聚合单个 track 在多帧上的 body/face 特征
    body 存 (feat, score, patch)；face 存 (feat, patch)
    """

    def __init__(self, max_body=MIN_BODY4GID, max_face=MIN_FACE4GID):
        self.body: deque = deque(maxlen=max_body)
        self.face: deque = deque(maxlen=max_face)
        self.last_fid = -1

    @staticmethod
    def _check_consistency(feats, thr=0.35):
        if len(feats) < 2: return True
        sims = [float(feats[i] @ feats[j]) for i in range(len(feats)) for j in range(i + 1, len(feats))]
        return 1.0 - np.mean(sims) <= thr

    def _main_representation(self, feats, patches, outlier_thr=1.5):
        if len(feats) == 0: return None, None
        arr = np.stack(feats)
        mean_f = arr.mean(axis=0)
        mean_f /= (np.linalg.norm(mean_f) + 1e-9)
        dists = np.linalg.norm(arr - mean_f, axis=1)
        keep = dists < (dists.mean() + outlier_thr * dists.std())
        kept_arr = arr[keep] if keep.any() else arr
        kept_patches = [p for k, p in zip(keep, patches) if k] if keep.any() else patches
        mean_f = kept_arr.mean(axis=0)
        mean_f /= np.linalg.norm(mean_f) + 1e-9
        sims = kept_arr @ mean_f
        idx = int(np.argmax(sims))
        return kept_arr[idx], kept_patches[idx]

    def add_body(self, feat, scr, fid, patch):
        if feat is None: return
        self.body.append((np.asarray(feat, np.float32), scr, patch));
        self.last_fid = fid

    def add_face(self, feat, fid, patch):
        if feat is None: return
        self.face.append((np.asarray(feat, np.float32), patch));
        self.last_fid = fid

    def main_body_feat_and_patch(self):
        if not self.body: return None, None
        feats, scores, patches = zip(*self.body)
        if not self._check_consistency(feats, thr=0.5): return None, None
        return self._main_representation(feats, patches)

    def main_face_feat_and_patch(self):
        if not self.face: return None, None
        feats, patches = zip(*self.face)
        if not self._check_consistency(feats, thr=0.5): return None, None
        return self._main_representation(feats, patches)

    def body_feat(self):
        if not self.body: return None
        feats, scores, _ = zip(*self.body)
        if not self._check_consistency(feats, thr=0.5): return None
        w = np.clip(np.float32(scores), 1e-2, None);
        w /= w.sum()
        rep = (np.stack(feats) * w[:, None]).sum(0);
        rep /= np.linalg.norm(rep) + 1e-9
        return rep

    def face_feat(self):
        if not self.face: return None
        feats, _ = zip(*self.face)
        if not self._check_consistency(feats, thr=0.5): return None
        rep = np.mean(np.stack(feats), 0);
        rep /= np.linalg.norm(rep) + 1e-9
        return rep

    def body_patches(self):
        return [p for *_r, p in self.body]

    def face_patches(self):
        return [p for _f, p in self.face]


def is_frontal_face_2d(kps: np.ndarray, yaw_sym_threshold=0.7, roll_angle_threshold=25.0) -> bool:
    """
    使用简单的2D几何方法判断是否为正脸，不使用solvePnP。
    :param kps: 5个关键点 (左眼, 右眼, 鼻子, 左嘴角, 右嘴角) 的 numpy 数组。
    :param yaw_sym_threshold: 偏航角对称性阈值。衡量鼻子到双眼距离的对称性，越接近1要求越严格。
    :param roll_angle_threshold: 翻滚角（头部倾斜）角度阈值。
    :return: 如果是正脸则返回 True，否则返回 False。
    """
    try:
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]

        # 1. 偏航角（Yaw）检测：基于鼻子与双眼的水平距离对称性
        # 计算鼻子到左右眼的水平距离
        dist_left = nose[0] - left_eye[0]
        dist_right = right_eye[0] - nose[0]

        # 确保关键点位置合理（例如，鼻子在双眼之间）
        if dist_left <= 0 or dist_right <= 0:
            return False

        # 计算对称性比例
        symmetry_ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        if symmetry_ratio < yaw_sym_threshold:
            return False  # 侧脸

        # 2. 翻滚角（Roll）检测：基于双眼连线的倾斜角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        # 避免除零错误
        if abs(dx) < 1e-6:
            return False

        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) > roll_angle_threshold:
            return False  # 头部倾斜过大

    except Exception:
        return False  # 关键点数据异常

    return True


# ===============================================================
@torch.inference_mode()
def prep_patch(patch: np.ndarray) -> torch.Tensor:
    """行人图像预处理"""
    im = cv2.resize(patch, (128, 256))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    im = (im - mean) / std
    return torch.from_numpy(im.transpose(2, 0, 1))


def normv(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ===============================================================
# GlobalID (保持不变)
# ===============================================================
class GlobalID:
    """管理全局身份库"""

    def __init__(self, max_proto=8, w_face=0.6, w_body=0.4, thr=0.5, outlier_thresh=3.0):
        self.max_proto, self.w_face, self.w_body, self.thr, self.outlier_thresh = max_proto, w_face, w_body, thr, outlier_thresh
        self.bank: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.tid_hist: Dict[str, List[str]] = {}
        self.last_update: Dict[str, float] = {}
        self.gid_next = 1

    @staticmethod
    def _sim(a, b):
        return float(a @ b)

    @staticmethod
    def _avg(feats):
        feats = np.stack(feats)
        rep = feats.mean(axis=0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    @staticmethod
    def remove_outliers(embeddings: list[np.ndarray], thresh: float = 1.2):
        n = len(embeddings)
        if n < 3: return embeddings, [True] * n
        arr = np.stack(embeddings);
        mean_vec = arr.mean(axis=0)
        dist = np.linalg.norm(arr - mean_vec, axis=1)
        z_scores = (dist - dist.mean()) / (dist.std() + 1e-8)
        keep_mask = np.atleast_1d(np.abs(z_scores) < thresh)
        if keep_mask.ndim != 1 or keep_mask.size != n: keep_mask = np.full(n, True, dtype=bool)
        new_list = [e for e, k in zip(embeddings, keep_mask) if k]
        return new_list, keep_mask.tolist()

    def can_update_proto(self, gid, face_feat, body_feat):
        pool = self.bank[gid]
        if pool['faces'] and self._sim(face_feat, self._avg(pool['faces'])) < FACE_THR_STRICT: return -1
        if pool['bodies'] and self._sim(body_feat, self._avg(pool['bodies'])) < BODY_THR_STRICT: return -2
        return 0

    def _add(self, lst, feat, patch, dir_path):
        if feat is None or patch is None: return
        if lst and max(self._sim(feat, x) for x in lst) < UPDATE_THR: return
        if len(lst) < self.max_proto:
            idx = len(lst)
            lst.append(feat)
        else:
            idx = int(np.argmax([self._sim(feat, x) for x in lst]))
            lst[idx] = normv(0.7 * lst[idx] + 0.3 * feat)
        cv2.imwrite(os.path.join(dir_path, f"{idx:02d}.jpg"), patch)
        new_lst, keep_mask = self.remove_outliers(lst, self.outlier_thresh)
        if len(new_lst) != len(lst):
            logger.info(f"[GlobalID] Outlier removed: {len(lst) - len(new_lst)} from {dir_path}")
            lst.clear()
            lst.extend(new_lst)
            for i, img_path in enumerate(sorted(Path(dir_path).glob("*.jpg"))):
                if i >= len(keep_mask) or not keep_mask[i]: img_path.unlink(missing_ok=True)

    def _best_match(self, face, body):
        best_gid, best_score = None, -1.0
        for gid, pool in self.bank.items():
            if not pool['faces'] or not pool['bodies']: continue
            sc = self.w_face * self._sim(face, self._avg(pool['faces'])) + self.w_body * self._sim(body, self._avg(
                pool['bodies']))
            if sc > best_score: best_gid, best_score = gid, sc
        return best_gid, best_score

    def probe(self, face, body):
        return self._best_match(face, body)

    def bind(self, gid, face, body, agg=None, tid=None, current_ts=None):
        root = os.path.join(SAVE_DIR, gid)
        if agg is not None:
            f_feat, f_patch = agg.main_face_feat_and_patch()
            b_feat, b_patch = agg.main_body_feat_and_patch()
        else:
            f_feat, f_patch, b_feat, b_patch = face, None, body, None
        self._add(self.bank[gid]['faces'], f_feat, f_patch, os.path.join(root, "faces"))
        self._add(self.bank[gid]['bodies'], b_feat, b_patch, os.path.join(root, "bodies"))
        if tid:
            self.tid_hist.setdefault(gid, [])
            if tid not in self.tid_hist[gid]: self.tid_hist[gid].append(tid)
        if current_ts is not None: self.last_update[gid] = current_ts

    def new_gid(self):
        gid = f"G{self.gid_next:05d}"
        self.gid_next += 1
        self.bank[gid] = dict(faces=[], bodies=[])
        self.tid_hist[gid] = []
        os.makedirs(os.path.join(SAVE_DIR, gid, "faces"), exist_ok=True)
        os.makedirs(os.path.join(SAVE_DIR, gid, "bodies"), exist_ok=True)
        logger.info(f"[GlobalID] new {gid}")
        return gid


# ===============================================================
# FeatureProcessor
# ===============================================================
class FeatureProcessor:
    """
    检测 → 特征 → 绑定/新建 GID → （新增：行为分析）→ 报警 & 清理
    """

    @staticmethod
    def _fuse_feat(face_f: np.ndarray | None, body_f: np.ndarray | None, w_face: float, w_body: float) -> np.ndarray:
        if face_f is None and body_f is None: raise RuntimeError("Both face and body feature are None")
        face_f = np.zeros(EMB_FACE_DIM, np.float32) if face_f is None else face_f * w_face
        body_f = np.zeros(EMB_BODY_DIM, np.float32) if body_f is None else body_f * w_body
        combo = np.concatenate([face_f, body_f]).astype(np.float32)
        return combo / (np.linalg.norm(combo) + 1e-9)

    @staticmethod
    def _calculate_ioa(person_tlwh: list, face_xyxy: list) -> float:
        """计算人脸框与人体框的交集面积占人脸框面积的比例 (IoA)"""
        px, py, pw, ph = person_tlwh
        px1, py1, px2, py2 = px, py, px + pw, py + ph
        fx1, fy1, fx2, fy2 = face_xyxy

        x_left = max(px1, fx1)
        y_top = max(py1, fy1)
        x_right = min(px2, fx2)
        y_bottom = min(py2, fy2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        face_area = (fx2 - fx1) * (fy2 - fy1)

        return intersection_area / (face_area + 1e-6)

    def _gid_fused_rep(self, gid: str, w_face: float, w_body: float) -> np.ndarray:
        pool = self.gid_mgr.bank.get(gid, {})
        face_f = self.gid_mgr._avg(pool['faces']) if pool.get('faces') else None
        body_f = self.gid_mgr._avg(pool['bodies']) if pool.get('bodies') else None
        return self._fuse_feat(face_f, body_f, w_face, w_body)

    def __init__(self,
                 device="cuda",
                 use_fid_time: bool | None = None,
                 mode: str = 'realtime',
                 cache_path: str | None = None,
                 boundary_config: Dict | None = None
                 ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fid_time = TIME_BY_FRAME if use_fid_time is None else use_fid_time
        self.mode = mode
        self.cache_path = cache_path
        self.features_to_save = {}
        self.features_cache = {}
        self.reid = None
        self.face_app = None
        self.rec_model = None
        if self.mode == 'realtime':
            logger.info("FeatureProcessor in 'realtime' mode. Models will be loaded.")
            if self.cache_path:
                logger.info(f"Features will be saved to '{self.cache_path}' on exit.")
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                self.reid = PersonReid(DIR_REID_MODEL, which_epoch="last",
                                       gpu="0" if self.device.type == "cuda" else "")
                self.reid.model.to(self.device).eval()
                face_provider = "CUDAExecutionProvider" if self.device.type == "cuda" else "CPUExecutionProvider"
                self.face_app = FaceSearcher(provider=face_provider).app

                if 'recognition' in self.face_app.models:
                    self.rec_model = self.face_app.models['recognition']
                    logger.info("Recognition-only model extracted from FaceAnalysis app.")
                else:
                    raise ValueError("Recognition model not found in FaceAnalysis app!")

                logger.info("ReID and Face models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load models in 'realtime' mode: {e}")
                raise
        elif self.mode == 'load':
            logger.info("FeatureProcessor in 'load' mode. Models will NOT be loaded.")
            if not self.cache_path or not os.path.exists(self.cache_path):
                raise FileNotFoundError(f"In 'load' mode, but cache file not found: {self.cache_path}")
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.features_cache = json.load(f)
                logger.info(
                    f"Successfully loaded features for {len(self.features_cache)} frames from '{self.cache_path}'.")
            except Exception as e:
                raise IOError(f"Failed to read or parse cache file '{self.cache_path}': {e}")
        else:
            raise ValueError(f"Invalid mode: '{self.mode}'. Choose 'realtime' or 'load'.")

        self.gid_mgr = GlobalID()
        self.agg_pool: Dict[str, TrackAgg] = {}
        self.last_seen: Dict[str, float] = {}
        self.tid2gid: Dict[str, str] = {}
        self.candidate_state: Dict[str, dict] = {}
        self.new_gid_state: Dict[str, dict] = {}
        self.alarmed: set[str] = set()
        self.alarm_reprs: Dict[str, np.ndarray] = {}
        self.behavior_alarm_state: Dict[str, Tuple[int, str]] = {}
        self.intrusion_detectors: Dict[str, IntrusionDetector] = {}
        self.line_crossing_detectors: Dict[str, Dict[str, LineCrossingDetectorPlus]] = {}
        if boundary_config:
            for stream_id, config in boundary_config.items():
                if "intrusion_poly" in config:
                    self.intrusion_detectors[stream_id] = IntrusionDetector(config["intrusion_poly"])
                    logger.info(f"Initialized IntrusionDetector for stream '{stream_id}'.")
                if "crossing_lines" in config:
                    self.line_crossing_detectors[stream_id] = {}
                    for i, line_cfg in enumerate(config["crossing_lines"]):
                        line_name = line_cfg.get("name", f"line_{i}")
                        self.line_crossing_detectors[stream_id][line_name] = LineCrossingDetectorPlus(
                            line_cfg["start"],
                            line_cfg["end"],
                            line_cfg.get("direction", "any"),
                            line_cfg.get("projection_depth", 50),
                            line_cfg.get("min_intersection_area", 100)  # 从配置中读取面积阈值
                        )
                        logger.info(f"Initialized LineCrossingDetector '{line_name}' for stream '{stream_id}'.")

    def __del__(self):
        if self.mode == 'realtime' and self.cache_path and self.features_to_save:
            logger.info(f"Saving {len(self.features_to_save)} frames of features to '{self.cache_path}'...")
            try:
                serializable_features = {}
                for f_id, f_data in self.features_to_save.items():
                    serializable_features[f_id] = {}
                    for t_id, t_data in f_data.items():
                        serializable_features[f_id][t_id] = {
                            'body_feat': t_data['body_feat'].tolist() if 'body_feat' in t_data and t_data[
                                'body_feat'] is not None else None,
                            'face_feat': t_data['face_feat'].tolist() if 'face_feat' in t_data and t_data[
                                'face_feat'] is not None else None,
                        }
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_features, f)
                logger.info("Features saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save features to '{self.cache_path}': {e}")

    def trigger_alarm(self, gid: str, agg: TrackAgg, w_face: float, w_body: float):
        try:
            cur_rep = self._gid_fused_rep(gid, w_face, w_body)
        except Exception as e:
            logger.warning(f"[ALARM] 生成 {gid} 特征失败: {e}")
            return
        for ogid, rep in self.alarm_reprs.items():
            if float(cur_rep @ rep) >= ALARM_DUP_THR:
                logger.info(f"[ALARM] 跳过 {gid} (与 {ogid} 相似)")
                return
        if gid in self.alarmed: return
        ts = time.strftime("%Y%m%d_%H%M%S")
        dst_dir = os.path.join(ALARM_DIR, f"{gid}_{ts}")
        try:
            shutil.copytree(os.path.join(SAVE_DIR, gid), dst_dir, dirs_exist_ok=True)
            seq_face = Path(dst_dir, "agg_sequence/face")
            seq_face.mkdir(parents=True, exist_ok=True)
            seq_body = Path(dst_dir, "agg_sequence/body")
            seq_body.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(agg.face_patches()): cv2.imwrite(str(seq_face / f"{i:03d}.jpg"), img)
            for i, img in enumerate(agg.body_patches()): cv2.imwrite(str(seq_body / f"{i:03d}.jpg"), img)
            self.alarmed.add(gid)
            self.alarm_reprs[gid] = cur_rep
            logger.warning(f"[ALARM] GID {gid} 已报警并备份到 {dst_dir}")
        except Exception as e:
            logger.error(f"[ALARM] 处理 {gid} 失败: {e}")

    # MODIFIED HERE: Entire method is refactored for internal face detection
    def process_packet(self, pkt: Dict):
        # 1. 解包输入
        stream_id = pkt["cam_id"]
        fid = pkt["fid"]
        full_frame = pkt["full_frame"]
        dets = pkt["dets"]
        w_face = pkt.get("w_face", 0.6)
        w_body = pkt.get("w_body", 0.4)

        _is_face_only_mode = w_face > 0.99999

        # 更新 GID 管理器的权重
        self.gid_mgr.w_face = w_face
        self.gid_mgr.w_body = w_body

        H, W = full_frame.shape[:2]

        # 2. 内部人脸检测 (仅在实时模式下)
        face_info = []
        if self.mode == 'realtime' and self.face_app and hasattr(self.face_app, 'det_model'):
            # 为提高性能，在缩放后的帧上检测
            small_frame = cv2.resize(full_frame, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
            faces_bboxes, faces_kpss = self.face_app.det_model.detect(
                small_frame, max_num=0, metric="default"
            )

            if faces_bboxes is not None and faces_bboxes.shape[0] > 0:
                for i in range(faces_bboxes.shape[0]):
                    bi = faces_bboxes[i, :4].astype(int)
                    # 坐标缩放回原始尺寸
                    x1, y1, x2, y2 = [int(b / SHOW_SCALE) for b in bi]
                    sc = float(faces_bboxes[i, 4])
                    kps = (
                        faces_kpss[i].astype(int).tolist()
                        if faces_kpss is not None and i < len(faces_kpss)
                        else None
                    )
                    if kps:
                        kps = [
                            [int(k[0] / SHOW_SCALE), int(k[1] / SHOW_SCALE)]
                            for k in kps
                        ]
                    face_info.append({"bbox": [x1, y1, x2, y2], "score": sc, "kps": kps})

        # 3. 行为分析
        if self.intrusion_detectors.get(stream_id):
            for tid_int in self.intrusion_detectors[stream_id].check(dets, stream_id):
                self.behavior_alarm_state[f"{stream_id}_{tid_int}"] = (fid, '_AA')
        if self.line_crossing_detectors.get(stream_id):
            for line_name, detector in self.line_crossing_detectors[stream_id].items():
                for tid_int, crossing_info in detector.check(dets, stream_id).items():
                    alarm_type = f'_AL_{line_name}'
                    self.behavior_alarm_state[f"{stream_id}_{tid_int}"] = (fid, alarm_type, crossing_info)

        # 4. 时间戳和超时设置
        if self.use_fid_time:
            now_stamp, max_tid_idle, gid_max_idle = fid, MAX_TID_IDLE_FRAMES, GID_MAX_IDLE_FRAMES
        else:
            now_stamp, max_tid_idle, gid_max_idle = time.time(), MAX_TID_IDLE_SEC, GID_MAX_IDLE_SEC

        # 5. 特征提取 (根据模式)
        if self.mode == 'load':
            precomputed_features = self.features_cache.get(str(fid), {})
            for det in dets:
                tid_str = f"{stream_id}_{det['id']}"
                feats_dict = precomputed_features.get(tid_str)
                if not feats_dict: continue

                body_feat = np.array(feats_dict['body_feat'], dtype=np.float32) if feats_dict.get('body_feat') else None
                face_feat = np.array(feats_dict['face_feat'], dtype=np.float32) if feats_dict.get('face_feat') else None

                x, y, w, h = det['tlwh']
                patch = full_frame[max(int(y), 0):min(int(y + h), H), max(int(x), 0):min(int(x + w), W)].copy()
                score = det.get('score', 0.0)

                agg = self.agg_pool.setdefault(tid_str, TrackAgg())
                if body_feat is not None: agg.add_body(body_feat, score, fid, patch)
                if face_feat is not None: agg.add_face(face_feat, fid, patch)
                self.last_seen[tid_str] = now_stamp

        elif self.mode == 'realtime':
            extracted_features_for_this_frame = {}
            # 准备 re-id 的批处理
            tensors, metas, person_patches = [], [], []
            for det in dets:
                if det.get('class_id') != 0: continue

                x, y, w, h = det['tlwh']
                patch = full_frame[max(int(y), 0):min(int(y + h), H), max(int(x), 0):min(int(x + w), W)].copy()

                # if not is_long_patch(patch): continue

                tensors.append(prep_patch(patch))
                metas.append((f"{stream_id}_{det['id']}", det.get("score", 0.0)))
                person_patches.append(patch)

            # 提取行人特征
            if tensors:
                batch = torch.stack(tensors).to(self.device).float()
                with torch.no_grad():
                    feats = torch.nn.functional.normalize(self.reid.model(batch), dim=1)
                feats = feats.cpu().numpy()
                for (tid, scr), f, p in zip(metas, feats, person_patches):
                    agg = self.agg_pool.setdefault(tid, TrackAgg())
                    self.last_seen[tid] = now_stamp
                    agg.add_body(f, scr, fid, p)
                    extracted_features_for_this_frame.setdefault(tid, {})['body_feat'] = f

            # 提取人脸特征 (使用内部检测到的 face_info)
            if face_info and self.rec_model is not None:
                used_face_indices = set()
                # 遍历每个行人检测框
                for det in dets:
                    if det.get('class_id') != 0: continue

                    x, y, w, h = det['tlwh']
                    person_patch = full_frame[max(int(y), 0):min(int(y + h), H),
                                   max(int(x), 0):min(int(x + w), W)].copy()
                    if person_patch.size == 0: continue

                    matching_face_indices = []
                    for j, face_det in enumerate(face_info):
                        if j in used_face_indices: continue
                        if self._calculate_ioa(det['tlwh'], face_det['bbox']) > 0.8:
                            matching_face_indices.append(j)

                    if len(matching_face_indices) != 1: continue

                    unique_face_idx = matching_face_indices[0]
                    used_face_indices.add(unique_face_idx)
                    face_det = face_info[unique_face_idx]

                    if face_det['score'] < FACE_DET_MIN_SCORE: continue

                    try:
                        bbox = np.array(face_det['bbox'], dtype=np.float32)
                        kps = np.array(face_det['kps'], dtype=np.float32) if face_det.get('kps') else None
                        face = Face(bbox=bbox, kps=kps, det_score=face_det.get('score', 0.99))

                        if kps is None or (not is_frontal_face_2d(kps) and _is_face_only_mode):
                            continue

                        x1, y1, x2, y2 = map(int, bbox)
                        if (x2 - x1) < 32 or (y2 - y1) < 32: continue
                        face_crop_for_blur = full_frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                        if face_crop_for_blur.size == 0 or cv2.Laplacian(
                                cv2.cvtColor(face_crop_for_blur, cv2.COLOR_BGR2GRAY),
                                cv2.CV_64F).var() < 100: continue

                        self.rec_model.get(full_frame, face)

                        if face.embedding is not None:
                            f_emb = normv(face.embedding)
                            tid = f"{stream_id}_{det['id']}"
                            agg = self.agg_pool.setdefault(tid, TrackAgg())
                            agg.add_face(f_emb, fid, person_patch)
                            self.last_seen[tid] = now_stamp
                            extracted_features_for_this_frame.setdefault(tid, {})['face_feat'] = f_emb
                    except Exception as e:
                        logger.warning(f"Failed to get face embedding for TID {det['id']} due to: {e}")
                        continue

            if self.cache_path and extracted_features_for_this_frame:
                self.features_to_save.setdefault(str(fid), {}).update(extracted_features_for_this_frame)

        # ... [ GID 绑定和清理逻辑保持不变, 此处省略以保持简洁 ] ...
        # (这部分逻辑完全复用，不需要修改)
        realtime_map: Dict[str, Dict[int, Tuple]] = {}
        for tid, agg in list(self.agg_pool.items()):
            ts, tn = tid.split("_")
            if ts != stream_id:
                continue
            if len(agg.body) < MIN_BODY4GID or len(agg.face) < MIN_FACE4GID:
                flag = f"{tid}_-1_b_{len(agg.body)}" if len(agg.body) < MIN_BODY4GID else f"{tid}_-1_f_{len(agg.face)}"
                realtime_map.setdefault(ts, {})[int(tn)] = (flag, -1.0, 0, None)
                continue
            face_feat, _ = agg.main_face_feat_and_patch()
            body_feat, _ = agg.main_body_feat_and_patch()
            if face_feat is None or body_feat is None:
                realtime_map.setdefault(ts, {})[int(tn)] = (
                    f"{tid}_-2_f" if face_feat is None else f"{tid}_-2_b", -1.0, 0, None)
                continue

            cand_gid, cand_score = self.gid_mgr.probe(face_feat, body_feat)

            if tid in self.tid2gid:
                bound_gid = self.tid2gid[tid]
                lock_elapsed = fid - self.candidate_state.get(tid, {}).get("last_bind_fid", 0)
                if cand_gid != bound_gid and lock_elapsed < BIND_LOCK_FRAMES:
                    n_tid = len(self.gid_mgr.tid_hist[bound_gid])
                    realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_-3", cand_score, n_tid, None)
                    continue

            state = self.candidate_state.setdefault(tid, dict(cand_gid=None, count=0, last_bind_fid=0))
            time_since_last_new = fid - self.new_gid_state.get(tid, {}).get("last_new_fid", -1)
            ng_state = self.new_gid_state.setdefault(tid,
                                                     dict(count=0, last_new_fid=-NEW_GID_TIME_WINDOW, ambig_count=0))

            if cand_gid and cand_score >= MATCH_THR:
                ng_state["ambig_count"] = 0
                state["count"] = state["count"] + 1 if state["cand_gid"] == cand_gid else 1
                state["cand_gid"] = cand_gid
                if state["count"] >= CANDIDATE_FRAMES and self.gid_mgr.can_update_proto(cand_gid, face_feat,
                                                                                        body_feat) == 0:
                    self.gid_mgr.bind(cand_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp)
                    self.tid2gid[tid] = cand_gid
                    state["last_bind_fid"] = fid
                    n_tid = len(self.gid_mgr.tid_hist.get(cand_gid, []))
                    if n_tid >= ALARM_CNT_TH: self.trigger_alarm(cand_gid, agg, w_face, w_body)
                    realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_{cand_gid}", cand_score, n_tid, None)
                else:
                    flag = self.gid_mgr.can_update_proto(cand_gid, face_feat, body_feat)
                    realtime_map.setdefault(ts, {})[int(tn)] = (
                        f"{tid}_-4_ud_f" if flag == -1 else f"{tid}_-4_ud_b" if flag == -2 else f"{tid}_-4_c", -1.0, 0, None)

            elif len(self.gid_mgr.bank) < 1:
                ng_state["ambig_count"] = 0
                new_gid = self.gid_mgr.new_gid()
                self.gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp)
                self.tid2gid[tid] = new_gid
                state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid)
                ng_state["last_new_fid"] = fid
                n_tid = len(self.gid_mgr.tid_hist[new_gid])
                if n_tid >= ALARM_CNT_TH: self.trigger_alarm(new_gid, agg, w_face, w_body)
                realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_{new_gid}", cand_score, n_tid, None)

            elif cand_gid and THR_NEW_GID <= cand_score < MATCH_THR and self.mode == 'load':
                ng_state["ambig_count"] += 1
                if ng_state["ambig_count"] >= WAIT_FRAMES_AMBIGUOUS and time_since_last_new >= NEW_GID_TIME_WINDOW:
                    new_gid = self.gid_mgr.new_gid()
                    self.gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp)
                    self.tid2gid[tid] = new_gid
                    state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid)
                    ng_state.update(last_new_fid=fid, count=0, ambig_count=0)
                    n_tid = len(self.gid_mgr.tid_hist[new_gid])
                    if n_tid >= ALARM_CNT_TH: self.trigger_alarm(new_gid, agg, w_face, w_body)
                    realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_{new_gid}", cand_score, n_tid, None)
                else:
                    realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_-7", cand_score, 0, None)

            elif cand_score < THR_NEW_GID:  # cand_score < THR_NEW_GID
                ng_state["ambig_count"] = 0
                if time_since_last_new >= NEW_GID_TIME_WINDOW:
                    ng_state["count"] += 1
                    if ng_state["count"] >= NEW_GID_MIN_FRAMES:
                        new_gid = self.gid_mgr.new_gid()
                        self.gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp)
                        self.tid2gid[tid] = new_gid
                        state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid)
                        ng_state.update(last_new_fid=fid, count=0)
                        n_tid = len(self.gid_mgr.tid_hist[new_gid])
                        if n_tid >= ALARM_CNT_TH: self.trigger_alarm(new_gid, agg, w_face, w_body)
                        realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_{new_gid}", cand_score, n_tid, None)
                    else:
                        realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_-5", -1.0, 0, None)
                else:
                    realtime_map.setdefault(ts, {})[int(tn)] = (f"{tid}_-6", -1.0, 0, None)

        active_alarms = {}
        for full_tid, state_tuple in self.behavior_alarm_state.items():
            start_fid, alarm_type = state_tuple[0], state_tuple[1]
            if fid - start_fid <= BEHAVIOR_ALARM_DURATION_FRAMES:
                active_alarms[full_tid] = state_tuple
                s_id, t_id_str = full_tid.split("_")
                t_id_int = int(t_id_str)
                bound_gid = self.tid2gid.get(full_tid, "")
                n_tid = len(self.gid_mgr.tid_hist.get(bound_gid, [])) if bound_gid else 0

                info_str_base = f"{full_tid}_{bound_gid}{alarm_type}" if bound_gid else f"{full_tid}_-1{alarm_type}"
                alarm_geometry = None
                if alarm_type.startswith('_AL') and len(state_tuple) > 2:
                    cross_info = state_tuple[2]
                    dist = cross_info.get("distance", 0)
                    ratio = cross_info.get("ratio", 0)
                    area = cross_info.get("area", 0)
                    info_str = f"{info_str_base}_D{dist:.0f}_R{ratio:.2f}_A{area:.0f}"
                    alarm_geometry = cross_info
                else:
                    info_str = info_str_base
                realtime_map.setdefault(s_id, {})[t_id_int] = (info_str, 1.0, n_tid, alarm_geometry)
        self.behavior_alarm_state = active_alarms

        for tid in list(self.last_seen.keys()):
            if now_stamp - self.last_seen[tid] >= max_tid_idle:
                self.last_seen.pop(tid, None)
                self.candidate_state.pop(tid, None)
                self.tid2gid.pop(tid, None)
                self.new_gid_state.pop(tid, None)
                self.agg_pool.pop(tid, None)
                self.behavior_alarm_state.pop(tid, None)

        to_delete = [gid for gid, t in self.gid_mgr.last_update.items() if now_stamp - t >= gid_max_idle]
        for gid in to_delete:
            tids_linked = [tid for tid, g in self.tid2gid.items() if g == gid]
            for tid in tids_linked:
                self.tid2gid.pop(tid, None)
                self.candidate_state.pop(tid, None)
                self.new_gid_state.pop(tid, None)
                self.agg_pool.pop(tid, None)
                self.last_seen.pop(tid, None)
                self.behavior_alarm_state.pop(tid, None)
            self.gid_mgr.bank.pop(gid, None)
            self.gid_mgr.tid_hist.pop(gid, None)
            self.gid_mgr.last_update.pop(gid, None)
            try:
                shutil.rmtree(os.path.join(SAVE_DIR, gid))
            except Exception as e:
                logger.warning(f"删除 GID 目录失败: {e}")

        return realtime_map
