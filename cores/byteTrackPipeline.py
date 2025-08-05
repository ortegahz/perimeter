# tracker_wrapper.py
from typing import List, Dict

import torch
from loguru import logger
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info

from cores.predictor import Predictor

__all__ = ["ByteTrackPipeline"]


class ByteTrackPipeline:
    """
    目标检测 (YOLOX) + 多目标跟踪 (BYTETracker) 统一封装。
    """

    def __init__(
            self,
            exp_file: str,
            ckpt: str,
            name: str = None,
            device: str = "cuda",
            fp16: bool = True,
            fuse: bool = True,
            # ---------- BYTETracker 超参数 ----------
            track_thresh: float = 0.5,
            track_buffer: int = 30,
            match_thresh: float = 0.8,
            aspect_ratio_thresh: float = 1.6,
            min_box_area: float = 10,
            fps: int = 30,
            mot20: bool = False,
    ):
        # 1. Experiment / Device ------------------------------------------------
        self.exp = get_exp(exp_file, name)
        self.device = torch.device("cuda" if device in ("cuda", "gpu") else "cpu")
        logger.info(f"[ByteTrackPipeline] using device: {self.device}")

        # 2. Build model --------------------------------------------------------
        model = self.exp.get_model().to(self.device).eval()
        logger.info("[ByteTrackPipeline] model summary: "
                    f"{get_model_info(model, self.exp.test_size)}")

        ckpt_dict = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ckpt_dict["model"], strict=True)
        logger.info("[ByteTrackPipeline] checkpoint loaded.")

        if fuse:
            logger.info("[ByteTrackPipeline] fusing Conv & BN ...")
            model = fuse_model(model)
        if fp16:
            model = model.half()

        # 3. Predictor ----------------------------------------------------------
        self.predictor = Predictor(
            model=model,
            exp=self.exp,
            trt_file=None,
            decoder=None,
            device=self.device,
            fp16=fp16,
        )

        # 4. BYTETracker --------------------------------------------------------
        class _Args:
            pass

        _byte_args = _Args()
        _byte_args.track_thresh = track_thresh
        _byte_args.track_buffer = track_buffer
        _byte_args.match_thresh = match_thresh
        _byte_args.aspect_ratio_thresh = aspect_ratio_thresh
        _byte_args.min_box_area = min_box_area
        _byte_args.mot20 = mot20

        self.tracker = BYTETracker(_byte_args, frame_rate=fps)

        # 5. 记录阈值到 Pipeline 自身 ------------------------------------------
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area

        # 6. 其它 --------------------------------------------------------------
        self.frame_id = 0
        self.timer = Timer()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def update(self, frame_bgr) -> List[Dict]:
        """
        输入 : frame_bgr (np.ndarray, BGR)
        输出 : List[dict]，形如
               {"id": 3, "tlwh": (x, y, w, h), "score": 0.87}
        """
        self.frame_id += 1

        # 1. detector 推理
        self.timer.tic()
        outputs, img_info = self.predictor.inference(frame_bgr, self.timer)
        self.timer.toc()

        # 2. tracker 更新
        online_targets = []
        if outputs[0] is not None:
            online_targets = self.tracker.update(
                outputs[0],
                [img_info["height"], img_info["width"]],
                self.exp.test_size,
            )

        # 3. 整理结果
        results: List[Dict] = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            score = t.score
            # 过滤
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] <= self.min_box_area or vertical:
                continue
            results.append(
                {
                    "id": int(tid),
                    "tlwh": (float(tlwh[0]), float(tlwh[1]),
                             float(tlwh[2]), float(tlwh[3])),
                    "score": float(score),
                }
            )

        return results
