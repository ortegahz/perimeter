from __future__ import annotations

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

from cores.faceSearcher import FaceSearcher
from cores.personReid import PersonReid
# ------------ 内部模块 ------------
from utils_peri.general_funcs import make_dirs
from utils_peri.macros import DIR_REID_MODEL

# ---------------------------------

# ===============================================================
# 常量区
# ===============================================================
SHOW_SCALE = 0.5
SENTINEL = None
MIN_HW_RATIO = 1.5
MIN_BODY4GID = 8
MIN_FACE4GID = 8
W_FACE, W_BODY = 0.6, 0.4
MATCH_THR = 0.5
THR_NEW_GID = 0.3
FACE_DET_MIN_SCORE = 0.60

SAVE_DIR = "/home/manu/tmp/perimeter"  # 原始数据目录
os.makedirs(SAVE_DIR, exist_ok=True)
make_dirs(SAVE_DIR, reset=True)

ALARM_DIR = "/home/manu/tmp/perimeter_alarm"  # 报警目录
os.makedirs(ALARM_DIR, exist_ok=True)
make_dirs(ALARM_DIR, reset=True)

UPDATE_THR = 0.65
FACE_THR_STRICT = 0.5
BODY_THR_STRICT = 0.4
NEW_GID_MIN_FRAMES = 3
NEW_GID_TIME_WINDOW = 50
BIND_LOCK_FRAMES = 15
CANDIDATE_FRAMES = 2
MAX_TID_GAP = 256  # 以帧为单位
GID_MAX_IDLE = 25 / 2 * 60 * 5
WAIT_FRAMES_AMBIGUOUS = 10

FPS = 25 / 2
MAX_TID_IDLE_SEC = MAX_TID_GAP / FPS
GID_MAX_IDLE_SEC = GID_MAX_IDLE / FPS


# ===============================================================

def is_long_patch(patch: np.ndarray, thr: float = MIN_HW_RATIO) -> bool:
    if patch is None or patch.size == 0:
        return False
    h, w = patch.shape[:2]
    return h / (w + 1e-9) >= thr


# ===============================================================
# TrackAgg
# ===============================================================
class TrackAgg:
    """
    聚合单个 track 在多帧上的 body/face 特征
    body 存储 (feat, score, patch)；face 存储 (feat, patch)
    """

    def __init__(self, max_body=MIN_BODY4GID, max_face=MIN_FACE4GID):
        self.body: deque = deque(maxlen=max_body)
        self.face: deque = deque(maxlen=max_face)
        self.last_fid = -1

    # ------------ representation & consistency -----------------
    @staticmethod
    def _check_consistency(feats, thr=0.35):
        if len(feats) < 2:
            return True
        sims = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                sims.append(float(feats[i] @ feats[j]))
        mean_diff = 1.0 - np.mean(sims)
        return mean_diff <= thr

    def _main_representation(self, feats, patches, outlier_thr=1.5):
        if len(feats) == 0:
            return None, None
        arr = np.stack(feats)
        mean_f = arr.mean(axis=0)
        mean_f /= (np.linalg.norm(mean_f) + 1e-9)
        dists = np.linalg.norm(arr - mean_f, axis=1)
        keep = dists < (dists.mean() + outlier_thr * dists.std())
        kept_arr = arr[keep]
        kept_patches = [p for k, p in zip(keep, patches) if k]
        if kept_arr.shape[0] == 0:
            kept_arr = arr
            kept_patches = patches
        mean_f = kept_arr.mean(axis=0)
        mean_f /= (np.linalg.norm(mean_f) + 1e-9)
        sims = kept_arr @ mean_f
        idx = int(np.argmax(sims))
        return kept_arr[idx], kept_patches[idx]

    # ------------ add & get interfaces -------------------------
    def add_body(self, feat, scr, fid, patch):
        if feat is None:
            return
        self.body.append((np.asarray(feat, dtype=np.float32), scr, patch))
        self.last_fid = fid

    def add_face(self, feat, fid, patch):
        if feat is None:
            return
        self.face.append((np.asarray(feat, dtype=np.float32), patch))
        self.last_fid = fid

    def main_body_feat_and_patch(self):
        if not self.body:
            return None, None
        feats, scores, patches = zip(*self.body)
        if not self._check_consistency(list(feats), thr=0.5):
            return None, None
        return self._main_representation(feats, patches, outlier_thr=1.5)

    def main_face_feat_and_patch(self):
        if not self.face:
            return None, None
        feats, patches = zip(*self.face)
        if not self._check_consistency(list(feats), thr=0.5):
            return None, None
        return self._main_representation(feats, patches, outlier_thr=1.5)

    def body_feat(self):
        if not self.body:
            return None
        feats, scores, _ = zip(*self.body)
        if not self._check_consistency(list(feats), thr=0.5):
            return None
        w = np.clip(np.float32(scores), 1e-2, None)
        w /= w.sum()
        rep = (np.stack(feats) * w[:, None]).sum(0)
        norm = np.linalg.norm(rep)
        if norm < 1e-9:
            return None
        return rep / norm

    def face_feat(self):
        if not self.face:
            return None
        feats, _ = zip(*self.face)
        if not self._check_consistency(list(feats), thr=0.5):
            return None
        rep = np.mean(np.stack(feats, axis=0), axis=0)
        norm = np.linalg.norm(rep)
        if norm < 1e-9:
            return None
        return rep / norm

    def body_patches(self):
        return [p for *_r, p in self.body]

    def face_patches(self):
        return [p for _f, p in self.face]


# ===============================================================
# 一些工具函数
# ===============================================================
@torch.inference_mode()
def prep_patch(patch: np.ndarray) -> torch.Tensor:
    im = cv2.resize(patch, (128, 256))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean, std = np.array([.485, .456, .406], dtype=np.float32), np.array([.229, .224, .225], dtype=np.float32)
    im = (im - mean) / std
    return torch.from_numpy(im.transpose(2, 0, 1))


def normv(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ===============================================================
# GlobalID
# ===============================================================
class GlobalID:
    """
    管理全局身份库
    """

    def __init__(self, max_proto=8, w_face=0.6, w_body=0.4, thr=0.5, outlier_thresh=3.0):
        self.max_proto = max_proto
        self.w_face = w_face
        self.w_body = w_body
        self.thr = thr
        self.outlier_thresh = outlier_thresh
        self.bank: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.tid_hist: Dict[str, List[str]] = {}
        self.last_update: Dict[str, float] = {}
        self.gid_next = 1

    # ---------- 静态工具 ----------
    @staticmethod
    def _sim(a, b):
        return float(a @ b)

    @staticmethod
    def _avg(feats):
        rep = np.mean(np.stack(feats, axis=0), 0)
        return rep / (np.linalg.norm(rep) + 1e-9)

    @staticmethod
    def remove_outliers(embeddings: list[np.ndarray], thresh: float = 1.2):
        if len(embeddings) < 3:
            return embeddings, [True] * len(embeddings)
        arr = np.stack(embeddings, axis=0)
        mean_vec = arr.mean(axis=0)
        dist = np.linalg.norm(arr - mean_vec, axis=1)
        z_scores = (dist - dist.mean()) / (dist.std() + 1e-8)
        keep_mask = np.abs(z_scores) < thresh
        new_list = [embeddings[i] for i in range(len(embeddings)) if keep_mask[i]]
        return new_list, keep_mask.tolist()

    # ---------- 接口 ----------
    def can_update_proto(self, gid, face_feat, body_feat):
        pool = self.bank[gid]
        if pool['faces'] and self._sim(face_feat, self._avg(pool['faces'])) < FACE_THR_STRICT:
            return -1
        if pool['bodies'] and self._sim(body_feat, self._avg(pool['bodies'])) < BODY_THR_STRICT:
            return -2
        return 0

    def _add(self, lst, feat, patch, dir_path):
        if feat is None or patch is None:
            return
        if lst and max(self._sim(feat, x) for x in lst) < UPDATE_THR:
            return
        if len(lst) < self.max_proto:
            idx = len(lst)
            lst.append(feat)
        else:
            idx = int(np.argmax([self._sim(feat, x) for x in lst]))
            lst[idx] = 0.7 * lst[idx] + 0.3 * feat
            lst[idx] /= np.linalg.norm(lst[idx]) + 1e-9
        cv2.imwrite(os.path.join(dir_path, f"{idx:02d}.jpg"), patch)

        new_lst, keep_mask = self.remove_outliers(lst, self.outlier_thresh)
        if len(new_lst) != len(lst):
            logger.info(f"[GlobalID] Outlier removed: {len(lst) - len(new_lst)} from {dir_path}")
            lst.clear()
            lst.extend(new_lst)
            img_files = sorted(Path(dir_path).glob("*.jpg"))
            for i, img_path in enumerate(img_files):
                if i >= len(keep_mask) or not keep_mask[i]:
                    img_path.unlink(missing_ok=True)

    def _best_match(self, face, body):
        gid_best, score_best = None, -1.0
        for gid, pool in self.bank.items():
            if not pool['faces'] or not pool['bodies']:
                continue
            score = self.w_face * self._sim(face, self._avg(pool['faces'])) + \
                    self.w_body * self._sim(body, self._avg(pool['bodies']))
            if score > score_best:
                gid_best, score_best = gid, score
        return gid_best, score_best

    def probe(self, face, body):
        return self._best_match(face, body)

    def bind(self, gid, face, body, agg=None, tid=None, current_ts=None):
        """
        更新库并可绑定 tid
        """
        root = os.path.join(SAVE_DIR, gid)
        if agg is not None:
            f_feat, f_patch = agg.main_face_feat_and_patch()
            b_feat, b_patch = agg.main_body_feat_and_patch()
        else:
            f_feat, f_patch = face, None
            b_feat, b_patch = body, None

        self._add(self.bank[gid]['faces'], f_feat, f_patch, os.path.join(root, "faces"))
        self._add(self.bank[gid]['bodies'], b_feat, b_patch, os.path.join(root, "bodies"))

        if tid:
            self.tid_hist.setdefault(gid, [])
            if tid not in self.tid_hist[gid]:
                self.tid_hist[gid].append(tid)

        if current_ts is not None:
            self.last_update[gid] = current_ts

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
    核心逻辑：检测 → 提取特征 → 绑定 / 新建 GID → 报警 & 清理
    """

    def __init__(self, device="cuda"):
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.reid = PersonReid(DIR_REID_MODEL, which_epoch="last",
                               gpu="0" if dev.type == "cuda" else "")
        self.face_app = FaceSearcher(provider="CUDAExecutionProvider").app

        self.gid_mgr = GlobalID()
        self.agg_pool: Dict[str, TrackAgg] = {}
        self.last_seen: Dict[str, float] = {}
        self.tid2gid: Dict[str, str] = {}
        self.candidate_state: Dict[str, dict] = {}
        self.new_gid_state: Dict[str, dict] = {}

        # 报警去重
        self.alarmed: set[str] = set()

    # ---------------- 报警函数 ----------------
    def trigger_alarm(self, gid: str, agg: TrackAgg):
        """
        将目录拷贝到报警目录，并把当前 agg 内的 patch 保存到 agg_sequence
        """
        if gid in self.alarmed:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        src_dir = os.path.join(SAVE_DIR, gid)
        dst_dir = os.path.join(ALARM_DIR, f"{gid}_{ts}")

        try:
            # 1) 拷贝已有数据
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

            # 2) 保存 agg 内的 patch 序列
            seq_face_dir = os.path.join(dst_dir, "agg_sequence", "face")
            seq_body_dir = os.path.join(dst_dir, "agg_sequence", "body")
            os.makedirs(seq_face_dir, exist_ok=True)
            os.makedirs(seq_body_dir, exist_ok=True)

            for i, img in enumerate(agg.face_patches()):
                cv2.imwrite(os.path.join(seq_face_dir, f"{i:03d}.jpg"), img)
            for i, img in enumerate(agg.body_patches()):
                cv2.imwrite(os.path.join(seq_body_dir, f"{i:03d}.jpg"), img)

            logger.warning(f"[ALARM] GID {gid} 关联 tid ≥ 2，已备份至 {dst_dir}")
            self.alarmed.add(gid)
        except Exception as e:
            logger.error(f"[ALARM] 处理 {gid} 失败: {e}")

    # ---------------- 主处理函数 ----------------
    def process_packet(self, pkt):
        """
        pkt = (stream_id, fid, patches, dets)
        返回 realtime_map 以供显示进程使用
        """
        stream_id, fid, patches, dets = pkt
        now_ts = time.time()

        # 1 -------- 行人 ReID ----------
        tensors, metas, keep_patches = [], [], []
        for det, patch in zip(dets, patches):
            if not is_long_patch(patch):
                continue
            tensors.append(prep_patch(patch))
            metas.append((f"{stream_id}_{det['id']}", det["score"]))
            keep_patches.append(patch)

        if tensors:
            batch = torch.stack(tensors).to(self.device).float()
            with torch.no_grad():
                out = self.reid.model(batch)
                out = torch.nn.functional.normalize(out, dim=1)
            feats = out.cpu().numpy()
            for (tid, scr), f, p in zip(metas, feats, keep_patches):
                agg = self.agg_pool.setdefault(tid, TrackAgg())
                agg.add_body(f, scr, fid, p)
                self.last_seen[tid] = now_ts

        # 2 -------- 人脸特征 ----------
        for det, patch in zip(dets, patches):
            faces = self.face_app.get(patch)
            if len(faces) != 1:
                continue
            face_obj = faces[0]
            if getattr(face_obj, "det_score", 1.0) < FACE_DET_MIN_SCORE:
                continue
            if patch.shape[0] < 120 or patch.shape[1] < 120:
                continue
            if cv2.Laplacian(patch, cv2.CV_64F).var() < 100:
                continue
            f_emb = normv(face_obj.embedding)
            tid = f"{stream_id}_{det['id']}"
            agg = self.agg_pool.setdefault(tid, TrackAgg())
            agg.add_face(f_emb, fid, patch)
            self.last_seen[tid] = now_ts

        # 3 -------- GID 绑定 / 新建 ----------
        realtime_map: Dict[str, Dict[int, Tuple[str, float, int]]] = {}
        for tid, agg in list(self.agg_pool.items()):
            # 3-0 前置检查
            if len(agg.body) < MIN_BODY4GID or len(agg.face) < MIN_FACE4GID:
                tid_stream, tid_num = tid.split("_", 1)
                flag = f"{tid_num}_-1_b_{len(agg.body)}" if len(agg.body) < MIN_BODY4GID else \
                    f"{tid_num}_-1_f_{len(agg.face)}"
                realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (flag, -1.0, 0)
                continue

            face_feat, _ = agg.main_face_feat_and_patch()
            body_feat, _ = agg.main_body_feat_and_patch()
            if face_feat is None or body_feat is None:
                tid_stream, tid_num = tid.split("_", 1)
                realtime_map.setdefault(tid_stream, {})[int(tid_num)] = \
                    ("-2_f", -1.0, 0) if face_feat is None else ("-2_b", -1.0, 0)
                continue

            cand_gid, cand_score = self.gid_mgr.probe(face_feat, body_feat)

            # 3-1 锁定检测
            if tid in self.tid2gid:
                bound_gid = self.tid2gid[tid]
                lock_elapsed = fid - self.candidate_state.get(tid, {}).get("last_bind_fid", 0)
                if cand_gid != bound_gid and lock_elapsed < BIND_LOCK_FRAMES:
                    n_tid = len(self.gid_mgr.tid_hist[bound_gid])
                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-3", cand_score, n_tid)
                    continue

            # --- 状态字典
            state = self.candidate_state.setdefault(tid, {"cand_gid": None, "count": 0, "last_bind_fid": 0})
            time_since_last_new = fid - self.new_gid_state.get(tid, {}).get("last_new_fid", -1)
            ng_state = self.new_gid_state.setdefault(tid, {"count": 0, "last_new_fid": -NEW_GID_TIME_WINDOW,
                                                           "ambig_count": 0})

            # 3-2 匹配到已有 GID
            if cand_gid and cand_score >= MATCH_THR:
                ng_state["ambig_count"] = 0
                if state["cand_gid"] == cand_gid:
                    state["count"] += 1
                else:
                    state["cand_gid"] = cand_gid
                    state["count"] = 1

                if state["count"] >= CANDIDATE_FRAMES and \
                        self.gid_mgr.can_update_proto(cand_gid, face_feat, body_feat) == 0:
                    self.gid_mgr.bind(cand_gid, face_feat, body_feat, agg,
                                      tid=tid, current_ts=now_ts)
                    self.tid2gid[tid] = cand_gid
                    state["last_bind_fid"] = fid
                    n_tid = len(self.gid_mgr.tid_hist[cand_gid])

                    if n_tid >= 2:
                        self.trigger_alarm(cand_gid, agg)

                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (cand_gid, cand_score, n_tid)
                else:
                    flag = self.gid_mgr.can_update_proto(cand_gid, face_feat, body_feat)
                    tid_stream, tid_num = tid.split("_", 1)
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = \
                        ("-4_ud_f", -1.0, 0) if flag == -1 else \
                            ("-4_ud_b", -1.0, 0) if flag == -2 else ("-4_c", -1.0, 0)

            # 3-3 身份库空，直接新建
            elif len(self.gid_mgr.bank) < 1:
                ng_state["ambig_count"] = 0
                new_gid = self.gid_mgr.new_gid()
                self.gid_mgr.bind(new_gid, face_feat, body_feat, agg,
                                  tid=tid, current_ts=now_ts)
                self.tid2gid[tid] = new_gid
                state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid)
                ng_state["last_new_fid"] = fid
                n_tid = len(self.gid_mgr.tid_hist[new_gid])

                if n_tid >= 2:
                    self.trigger_alarm(new_gid, agg)

                tid_stream, tid_num = tid.split("_", 1)
                realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (new_gid, cand_score, n_tid)

            # 3-4 模糊匹配
            elif cand_gid and THR_NEW_GID <= cand_score < MATCH_THR:
                ng_state["ambig_count"] += 1
                tid_stream, tid_num = tid.split("_", 1)
                if ng_state["ambig_count"] >= WAIT_FRAMES_AMBIGUOUS and \
                        time_since_last_new >= NEW_GID_TIME_WINDOW:
                    new_gid = self.gid_mgr.new_gid()
                    self.gid_mgr.bind(new_gid, face_feat, body_feat, agg,
                                      tid=tid, current_ts=now_ts)
                    self.tid2gid[tid] = new_gid
                    state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid)
                    ng_state.update(last_new_fid=fid, count=0, ambig_count=0)
                    n_tid = len(self.gid_mgr.tid_hist[new_gid])

                    if n_tid >= 2:
                        self.trigger_alarm(new_gid, agg)

                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (new_gid, cand_score, n_tid)
                else:
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-7", cand_score, 0)

            # 3-5 完全没有匹配
            else:
                ng_state["ambig_count"] = 0
                tid_stream, tid_num = tid.split("_", 1)
                if time_since_last_new >= NEW_GID_TIME_WINDOW:
                    ng_state["count"] += 1
                    if ng_state["count"] >= NEW_GID_MIN_FRAMES:
                        new_gid = self.gid_mgr.new_gid()
                        self.gid_mgr.bind(new_gid, face_feat, body_feat, agg,
                                          tid=tid, current_ts=now_ts)
                        self.tid2gid[tid] = new_gid
                        state.update(cand_gid=new_gid, count=CANDIDATE_FRAMES, last_bind_fid=fid)
                        ng_state.update(last_new_fid=fid, count=0)
                        n_tid = len(self.gid_mgr.tid_hist[new_gid])

                        if n_tid >= 2:
                            self.trigger_alarm(new_gid, agg)

                        realtime_map.setdefault(tid_stream, {})[int(tid_num)] = (new_gid, cand_score, n_tid)
                    else:
                        realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-5", -1.0, 0)
                else:
                    realtime_map.setdefault(tid_stream, {})[int(tid_num)] = ("-6", -1.0, 0)

        # 4 -------- 清理超时 tid ----------
        for tid in list(self.last_seen.keys()):
            if now_ts - self.last_seen[tid] >= MAX_TID_IDLE_SEC:
                self.last_seen.pop(tid, None)
                self.candidate_state.pop(tid, None)
                self.tid2gid.pop(tid, None)
                self.new_gid_state.pop(tid, None)
                self.agg_pool.pop(tid, None)

        # 5 -------- 清理超时 gid ----------
        to_delete = [gid for gid, t in self.gid_mgr.last_update.items()
                     if now_ts - t >= GID_MAX_IDLE_SEC]
        for gid in to_delete:
            tids_linked = [tid for tid, g in self.tid2gid.items() if g == gid]
            if tids_linked:
                logger.warning(f"GID {gid} 将被删除，但仍绑定 tid: {tids_linked}")
                for tid in tids_linked:
                    self.tid2gid.pop(tid, None)
                    self.candidate_state.pop(tid, None)
                    self.new_gid_state.pop(tid, None)
                    self.agg_pool.pop(tid, None)
                    self.last_seen.pop(tid, None)

            logger.info(f"[GlobalID] GID {gid} 因空闲超时被删除")
            self.gid_mgr.bank.pop(gid, None)
            self.gid_mgr.tid_hist.pop(gid, None)
            self.gid_mgr.last_update.pop(gid, None)
            dir_path = os.path.join(SAVE_DIR, gid)
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"删除目录 {dir_path} 失败: {e}")

        return realtime_map
