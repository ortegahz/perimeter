from __future__ import annotations

import json
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

# 注意：这里的导入路径需要根据你的项目结构调整
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

W_FACE, W_BODY = 0.6, 0.4
MATCH_THR = 0.5
THR_NEW_GID = 0.3
FACE_DET_MIN_SCORE = 0.60

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
FUSE_W_FACE, FUSE_W_BODY = 0.6, 0.4
EMB_FACE_DIM, EMB_BODY_DIM = 512, 2048


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

    def check(self, dets: list[dict], stream_id: str) -> None:
        if self.boundary is None: return
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
        disappeared_tids = set(self.track_history.keys()) - current_tids
        for tid in disappeared_tids:
            self.track_history.pop(tid, None)
            self.alarmed_tids.discard(tid)  # ###-FIXED-### Changed .pop(tid, None) to .discard(tid)


def get_point_side(p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    """计算点相对于有向直线AB的位置"""
    val = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if val > 0: return 1
    if val < 0: return -1
    return 0


class LineCrossingDetector:
    """线条穿越检测器"""

    def __init__(self, line_start: tuple[int, int], line_end: tuple[int, int], direction: str = 'any'):
        self.line_start = line_start
        self.line_end = line_end
        self.direction = direction
        self.track_side_history = {}
        self.alarmed_tids = set()

    def check(self, dets: list[dict], stream_id: str) -> None:
        current_tids = {d['id'] for d in dets}
        for d in dets:
            tid = d['id']
            if tid in self.alarmed_tids: continue
            current_point = get_foot_point(d['tlwh'])
            current_side = get_point_side(current_point, self.line_start, self.line_end)
            last_side = self.track_side_history.get(tid)
            self.track_side_history[tid] = current_side
            if last_side is not None and current_side != last_side and last_side != 0 and current_side != 0:
                crossed = (self.direction == 'any') or \
                          (self.direction == 'in' and last_side < 0 and current_side > 0) or \
                          (self.direction == 'out' and last_side > 0 and current_side < 0)
                if crossed:
                    logger.warning(f"[ALARM][{stream_id}] Line Crossing Detected! TID:{tid} crossed the line.")
                    self.alarmed_tids.add(tid)
        disappeared_tids = set(self.track_side_history.keys()) - current_tids
        for tid in disappeared_tids:
            self.track_side_history.pop(tid, None)
            self.alarmed_tids.discard(tid)  # ###-FIXED-### Changed .pop(tid, None) to .discard(tid)


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

    # ... (此类的所有方法保持不变) ...
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

    # ... (此类的所有方法保持不变) ...
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

    新增的边界检测功能通过 `boundary_config` 在初始化时配置，例如：

    boundary_config = {
        "cam1": {
            "intrusion_poly": [(100, 900), (800, 900), (800, 500), (100, 500)],
            "crossing_line": {"start": (100, 100), "end": (900, 900), "direction": "any"}
        },
        "cam2": { ... }
    }
    processor = FeatureProcessor(..., boundary_config=boundary_config)
    """

    # ----------------- 内部辅助 -----------------
    @staticmethod
    def _fuse_feat(face_f: np.ndarray | None, body_f: np.ndarray | None) -> np.ndarray:
        if face_f is None and body_f is None: raise RuntimeError("Both face and body feature are None")
        face_f = np.zeros(EMB_FACE_DIM, np.float32) if face_f is None else face_f * FUSE_W_FACE
        body_f = np.zeros(EMB_BODY_DIM, np.float32) if body_f is None else body_f * FUSE_W_BODY
        combo = np.concatenate([face_f, body_f]).astype(np.float32)
        return combo / (np.linalg.norm(combo) + 1e-9)

    def _gid_fused_rep(self, gid: str) -> np.ndarray:
        pool = self.gid_mgr.bank.get(gid, {})
        face_f = self.gid_mgr._avg(pool['faces']) if pool.get('faces') else None
        body_f = self.gid_mgr._avg(pool['bodies']) if pool.get('bodies') else None
        return self._fuse_feat(face_f, body_f)

    # ------------------------------------------------
    def __init__(self,
                 device="cuda",
                 use_fid_time: bool | None = None,
                 mode: str = 'realtime',  # 'realtime' (默认) 或 'load'
                 cache_path: str | None = None,  # 特征缓存文件路径
                 boundary_config: Dict | None = None  # ========== 新增：边界配置 ==========
                 ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fid_time = TIME_BY_FRAME if use_fid_time is None else use_fid_time
        self.mode = mode
        self.cache_path = cache_path
        self.features_to_save = {}
        self.features_cache = {}
        self.reid = None
        self.face_app = None

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

        # ========== 新增：初始化边界检测器 ==========
        self.intrusion_detectors: Dict[str, IntrusionDetector] = {}
        self.line_crossing_detectors: Dict[str, LineCrossingDetector] = {}
        if boundary_config:
            for stream_id, config in boundary_config.items():
                if "intrusion_poly" in config:
                    self.intrusion_detectors[stream_id] = IntrusionDetector(config["intrusion_poly"])
                    logger.info(f"Initialized IntrusionDetector for stream '{stream_id}'.")
                if "crossing_line" in config:
                    line_cfg = config["crossing_line"]
                    self.line_crossing_detectors[stream_id] = LineCrossingDetector(
                        line_cfg["start"], line_cfg["end"], line_cfg.get("direction", "any")
                    )
                    logger.info(f"Initialized LineCrossingDetector for stream '{stream_id}'.")
        # ===============================================

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

    # ... (报警逻辑保持不变) ...
    def trigger_alarm(self, gid: str, agg: TrackAgg):
        try:
            cur_rep = self._gid_fused_rep(gid)
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

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------
    def process_packet(self, pkt):
        """
        pkt = (stream_id, fid, patches, dets)
        返回 realtime_map: Dict[cam_id][tid] = (gid, score, n_tid)
        """
        stream_id, fid, patches, dets = pkt

        # ========== 新增：在此处执行行为检测 ==========
        # 这里的 `dets` 包含原始尺寸的 `tlwh`，可以直接使用
        intrusion_detector = self.intrusion_detectors.get(stream_id)
        if intrusion_detector:
            intrusion_detector.check(dets, stream_id)

        line_detector = self.line_crossing_detectors.get(stream_id)
        if line_detector:
            line_detector.check(dets, stream_id)
        # ===========================================

        # ========== 统一时间基准 (保持不变) ==========
        if self.use_fid_time:
            now_stamp, max_tid_idle, gid_max_idle = fid, MAX_TID_IDLE_FRAMES, GID_MAX_IDLE_FRAMES
        else:
            now_stamp, max_tid_idle, gid_max_idle = time.time(), MAX_TID_IDLE_SEC, GID_MAX_IDLE_SEC

        # -------- 特征提取或加载的分支 (保持不变) --------
        if self.mode == 'load':
            precomputed_features = self.features_cache.get(str(fid))
            if precomputed_features:
                for tid_str, feats_dict in precomputed_features.items():
                    body_feat = np.array(feats_dict['body_feat'], dtype=np.float32) if feats_dict.get(
                        'body_feat') else None
                    face_feat = np.array(feats_dict['face_feat'], dtype=np.float32) if feats_dict.get(
                        'face_feat') else None
                    num_tid = int(tid_str.split('_')[-1])
                    found_patch, found_score = None, 0.0
                    for det, patch in zip(dets, patches):
                        if det['id'] == num_tid: found_patch, found_score = patch, det.get('score', 0.0); break
                    if found_patch is None: continue
                    agg = self.agg_pool.setdefault(tid_str, TrackAgg())
                    if body_feat is not None: agg.add_body(body_feat, found_score, fid, found_patch)
                    if face_feat is not None: agg.add_face(face_feat, fid, found_patch)
                    self.last_seen[tid_str] = now_stamp
        elif self.mode == 'realtime':
            extracted_features_for_this_frame = {}
            tensors, metas, keep_patches = [], [], []
            for det, patch in zip(dets, patches):
                if det.get('class_id') != 0 or not is_long_patch(patch): continue
                tensors.append(prep_patch(patch))
                metas.append((f"{stream_id}_{det['id']}", det.get("score", 0.0)))
                keep_patches.append(patch)
            if tensors:
                batch = torch.stack(tensors).to(self.device).float()
                with torch.no_grad():
                    feats = torch.nn.functional.normalize(self.reid.model(batch), dim=1)
                feats = feats.cpu().numpy()
                for (tid, scr), f, p in zip(metas, feats, keep_patches):
                    agg = self.agg_pool.setdefault(tid, TrackAgg())
                    agg.add_body(f, scr, fid, p)
                    self.last_seen[tid] = now_stamp
                    extracted_features_for_this_frame.setdefault(tid, {})['body_feat'] = f
            for det, patch in zip(dets, patches):
                if det.get('class_id') != 0: continue
                try:
                    faces = self.face_app.get(patch)
                    if len(faces) != 1: continue
                    face_obj = faces[0]
                    if getattr(face_obj, "det_score", 1.0) < FACE_DET_MIN_SCORE: continue
                    if patch.shape[0] < 120 or patch.shape[1] < 120: continue
                    if cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 100: continue
                    f_emb = normv(face_obj.embedding)
                    tid = f"{stream_id}_{det['id']}"
                    agg = self.agg_pool.setdefault(tid, TrackAgg())
                    agg.add_face(f_emb, fid, patch)
                    self.last_seen[tid] = now_stamp
                    extracted_features_for_this_frame.setdefault(tid, {})['face_feat'] = f_emb
                except Exception:
                    continue
            if self.cache_path and extracted_features_for_this_frame:
                self.features_to_save.setdefault(str(fid), {}).update(extracted_features_for_this_frame)

        # ---------------- 3. GID 绑定 / 新建 (保持不变) ----------------
        realtime_map: Dict[str, Dict[int, Tuple[str, float, int]]] = {}
        for tid, agg in list(self.agg_pool.items()):
            if len(agg.body) < MIN_BODY4GID or len(agg.face) < MIN_FACE4GID:
                ts, tn = tid.split("_")
                flag = f"{tn}_-1_b_{len(agg.body)}" if len(agg.body) < MIN_BODY4GID else f"{tn}_-1_f_{len(agg.face)}"
                realtime_map.setdefault(ts, {})[int(tn)] = (flag, -1.0, 0)
                continue
            face_feat, _ = agg.main_face_feat_and_patch()
            body_feat, _ = agg.main_body_feat_and_patch()
            if face_feat is None or body_feat is None:
                ts, tn = tid.split("_")
                realtime_map.setdefault(ts, {})[int(tn)] = ("-2_f" if face_feat is None else "-2_b", -1.0, 0)
                continue
            cand_gid, cand_score = self.gid_mgr.probe(face_feat, body_feat)
            if tid in self.tid2gid:
                bound_gid = self.tid2gid[tid]
                lock_elapsed = fid - self.candidate_state.get(tid, {}).get("last_bind_fid", 0)
                if cand_gid != bound_gid and lock_elapsed < BIND_LOCK_FRAMES:
                    n_tid = len(self.gid_mgr.tid_hist[bound_gid])
                    ts, tn = tid.split("_")
                    realtime_map.setdefault(ts, {})[int(tn)] = ("-3", cand_score, n_tid)
                    continue
            state = self.candidate_state.setdefault(tid, dict(cand_gid=None, count=0, last_bind_fid=0))
            time_since_last_new = fid - self.new_gid_state.get(tid, {}).get("last_new_fid", -1)
            ng_state = self.new_gid_state.setdefault(tid,
                                                     dict(count=0, last_new_fid=-NEW_GID_TIME_WINDOW, ambig_count=0))
            if cand_gid and cand_score >= MATCH_THR:
                ng_state["ambig_count"] = 0
                state["count"] = state["count"] + 1 if state["cand_gid"] == cand_gid else 1;
                state["cand_gid"] = cand_gid
                if state["count"] >= CANDIDATE_FRAMES and self.gid_mgr.can_update_proto(cand_gid, face_feat,
                                                                                        body_feat) == 0:
                    self.gid_mgr.bind(cand_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp);
                    self.tid2gid[tid] = cand_gid
                    state["last_bind_fid"] = fid
                    n_tid = len(self.gid_mgr.tid_hist[cand_gid])
                    if n_tid >= ALARM_CNT_TH: self.trigger_alarm(cand_gid, agg)
                    ts, tn = tid.split("_")
                    realtime_map.setdefault(ts, {})[int(tn)] = (cand_gid, cand_score, n_tid)
                else:
                    flag = self.gid_mgr.can_update_proto(cand_gid, face_feat, body_feat);
                    ts, tn = tid.split("_")
                    realtime_map.setdefault(ts, {})[int(tn)] = (
                        "-4_ud_f" if flag == -1 else "-4_ud_b" if flag == -2 else "-4_c", -1.0, 0)
            elif len(self.gid_mgr.bank) < 1:
                ng_state["ambig_count"] = 0
                new_gid = self.gid_mgr.new_gid()
                self.gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp);
                self.tid2gid[tid] = new_gid
                state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid);
                ng_state["last_new_fid"] = fid
                n_tid = len(self.gid_mgr.tid_hist[new_gid])
                if n_tid >= ALARM_CNT_TH: self.trigger_alarm(new_gid, agg)
                ts, tn = tid.split("_")
                realtime_map.setdefault(ts, {})[int(tn)] = (new_gid, cand_score, n_tid)
            elif cand_gid and THR_NEW_GID <= cand_score < MATCH_THR:
                ng_state["ambig_count"] += 1
                ts, tn = tid.split("_")
                if ng_state["ambig_count"] >= WAIT_FRAMES_AMBIGUOUS and time_since_last_new >= NEW_GID_TIME_WINDOW:
                    new_gid = self.gid_mgr.new_gid()
                    self.gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp);
                    self.tid2gid[tid] = new_gid
                    state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid);
                    ng_state.update(last_new_fid=fid, count=0, ambig_count=0)
                    n_tid = len(self.gid_mgr.tid_hist[new_gid])
                    if n_tid >= ALARM_CNT_TH: self.trigger_alarm(new_gid, agg)
                    realtime_map.setdefault(ts, {})[int(tn)] = (new_gid, cand_score, n_tid)
                else:
                    realtime_map.setdefault(ts, {})[int(tn)] = ("-7", cand_score, 0)
            else:
                ng_state["ambig_count"] = 0
                ts, tn = tid.split("_")
                if time_since_last_new >= NEW_GID_TIME_WINDOW:
                    ng_state["count"] += 1
                    if ng_state["count"] >= NEW_GID_MIN_FRAMES:
                        new_gid = self.gid_mgr.new_gid()
                        self.gid_mgr.bind(new_gid, face_feat, body_feat, agg, tid=tid, current_ts=now_stamp)
                        self.tid2gid[tid] = new_gid
                        state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid)
                        ng_state.update(last_new_fid=fid, count=0)
                        n_tid = len(self.gid_mgr.tid_hist[new_gid])
                        if n_tid >= ALARM_CNT_TH: self.trigger_alarm(new_gid, agg)
                        realtime_map.setdefault(ts, {})[int(tn)] = (new_gid, cand_score, n_tid)
                    else:
                        realtime_map.setdefault(ts, {})[int(tn)] = ("-5", -1.0, 0)
                else:
                    realtime_map.setdefault(ts, {})[int(tn)] = ("-6", -1.0, 0)

        # ---------------- 4. 清理超时 tid (保持不变) ----------------
        for tid in list(self.last_seen.keys()):
            if now_stamp - self.last_seen[tid] >= max_tid_idle:
                self.last_seen.pop(tid, None)
                self.candidate_state.pop(tid, None)
                self.tid2gid.pop(tid, None)
                self.new_gid_state.pop(tid, None)
                self.agg_pool.pop(tid, None)

        # ---------------- 5. 清理超时 gid (保持不变) ----------------
        to_delete = [gid for gid, t in self.gid_mgr.last_update.items() if now_stamp - t >= gid_max_idle]
        for gid in to_delete:
            tids_linked = [tid for tid, g in self.tid2gid.items() if g == gid]
            for tid in tids_linked:
                self.tid2gid.pop(tid, None)
                self.candidate_state.pop(tid, None)
                self.new_gid_state.pop(tid, None)
                self.agg_pool.pop(tid, None)
                self.last_seen.pop(tid, None)
            self.gid_mgr.bank.pop(gid, None)
            self.gid_mgr.tid_hist.pop(gid, None)
            self.gid_mgr.last_update.pop(gid, None)
            try:
                shutil.rmtree(os.path.join(SAVE_DIR, gid))
            except Exception as e:
                logger.warning(f"删除 GID 目录失败: {e}")

        return realtime_map
