import os
import pickle
from typing import List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceSearcher:
    """
    1. 支持把指定目录的人脸照片批量建库（或增量添加）
    2. 支持把 embedding 缓存到磁盘，后续秒级加载
    3. 输入 query 图片，一次性返回 top-k 最相似的人脸与分数
    """

    def __init__(
            self,
            model_name: str = "buffalo_l",
            provider: str = "CPUExecutionProvider",  # 或 "CUDAExecutionProvider"
            threshold: float = 0.45,
            cache_path: str = "/home/manu/tmp/face_db.pkl",
    ):
        self.threshold = threshold
        self.cache_path = cache_path

        # 初始化 insightface
        self.app = FaceAnalysis(name=model_name, providers=[provider])
        # ctx_id: -1 用 CPU；0、1…用 GPU 对应编号
        self.app.prepare(ctx_id=-1 if provider == "CPUExecutionProvider" else 0)

        # 数据结构
        self.embeddings: List[np.ndarray] = []  # shape=(N, 512)
        self.img_paths: List[str] = []  # 与 embeddings 同序
        if os.path.exists(self.cache_path):
            self._load_cache()

    # ============== 对外 API ==============

    def build_db(self, face_dir: str):
        """
        指定目录批量建库；只处理 .jpg/.png/.jpeg/.bmp
        """
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_files = [
            os.path.join(dp, fn)
            for dp, _, fns in os.walk(face_dir)
            for fn in fns
            if os.path.splitext(fn.lower())[1] in exts
        ]
        if not img_files:
            raise ValueError(f"No images found under {face_dir}")

        for img_path in img_files:
            if img_path in self.img_paths:  # 已在库里
                continue
            try:
                emb = self._get_embedding(img_path)
                self.embeddings.append(emb)
                self.img_paths.append(img_path)
                print(f"[ADD] {img_path}")
            except Exception as e:
                print(f"[WARN] {img_path} skipped: {e}")

        self._save_cache()

    def add(self, img_path: str):
        """
        新增单张人脸到库
        """
        emb = self._get_embedding(img_path)
        self.embeddings.append(emb)
        self.img_paths.append(img_path)
        self._save_cache()

    def search(
            self, query_img_path: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        检索并返回 top-k
        return: List[(img_path, similarity_score)]，按分数降序
        """
        if not self.embeddings:
            raise RuntimeError("Face database is empty, please call build_db() first.")

        query_emb = self._get_embedding(query_img_path)  # shape=(512,)

        # 转成矩阵方便一次性向量化计算
        db_embs = np.stack(self.embeddings, axis=0)  # shape=(N, 512)

        # 余弦相似度： (q·db) / (||q||*||db||)
        scores = (db_embs @ query_emb) / (
                np.linalg.norm(db_embs, axis=1) * np.linalg.norm(query_emb) + 1e-12
        )

        # 取 top-k
        idxs = np.argsort(-scores)[:top_k]  # 降序
        results = [
            (self.img_paths[i], float(scores[i]))
            for i in idxs
            if scores[i] >= self.threshold
        ]

        return results

    # ============== 内部工具函数 ==============

    def _get_embedding(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        faces = self.app.get(img)
        if len(faces) < 1:
            raise ValueError("No face detected")
        if len(faces) > 1:
            print(f"[INFO] Multiple faces in {img_path}, using the first one")
        return faces[0].embedding.astype(np.float32)

    def _save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump({"paths": self.img_paths, "embs": self.embeddings}, f)
        print(f"[SAVE] DB cached to {self.cache_path} (size={len(self.img_paths)})")

    def _load_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.img_paths = data["paths"]
                self.embeddings = data["embs"]
            print(f"[LOAD] Loaded {len(self.img_paths)} faces from cache")
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")
            self.img_paths, self.embeddings = [], []
