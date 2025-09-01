#!/usr/bin/env python3
"""
FastReID PyTorch OSNet Market1501集成系统 - 修复版 V4.2
基于DeepStream 7.1 + osnet_ibn_x1_0_market1501 + 512维特征提取
支持RTSP流和本地视频文件

【修复内容】：
1. 修复ALPHA_EMA错误 - 使用局部变量alpha而不是self.ALPHA_EMA
2. 修复person_match_count未定义错误 - 添加初始化和递增逻辑
3. 修复prototype字段访问错误 - 使用正确的reid_prototype字段
4. 优化PyTorch推理流程 - 使用inference_mode提高性能
5. 清理重复导入和未定义变量
6. 增强错误处理和统计功能
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import cv2
import gi


def numpy_encoder(obj):
    """自定义JSON编码器，处理NumPy数组和其他非JSON原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_encoder(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_encoder(item) for item in obj]
    return obj


gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GLib

import pyds
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import onnxruntime  # 【新增】导入 onnxruntime

# 添加FastReID路径
sys.path.insert(0, '/home/fanrrrrrrr/workspace/aihub/yunhe/fast-reid')
from osnet import OSNet as OSNetIBN, OSBlock

# --- 【新增】人脸识别配置 ---
FACE_MODEL_PATH = "/home/fanrrrrrrr/下载/buffalo_l/w600k_r50.onnx"  # Buffalo_L模型路径
FACE_FEATURE_DIM = 512  # 人脸特征维度
FACE_SIMILARITY_THRESHOLD = 0.5  # 人脸相似度阈值
FACE_WEIGHT = 0.7  # 人脸特征权重
REID_WEIGHT = 0.3  # ReID特征权重
HYBRID_SCORE_THRESHOLD = 0.6  # 混合得分阈值

# DeepStream 7.1 特定常量
# 参考 deepstream-test2：使用统一分辨率，避免复杂的坐标转换
MUXER_OUTPUT_WIDTH = 1280  # 匹配输入视频宽度
MUXER_OUTPUT_HEIGHT = 720  # 匹配输入视频高度
TILER_OUTPUT_WIDTH = 2560  # 2个1280x720视频并排显示的总宽度
TILER_OUTPUT_HEIGHT = 720  # 保持高度不变
MODEL_INPUT_WIDTH = 960  # 模型原始输入宽度
MODEL_INPUT_HEIGHT = 544  # 模型原始输入高度


def create_default_config(config_path):
    """创建与模型文件匹配的配置文件"""

    # 验证模型文件是否存在
    model_dir = "./models"
    # 尝试使用现有的引擎文件
    engine_file = f"{model_dir}/resnet34_peoplenet.onnx_b2_gpu0_fp16.engine"
    onnx_file = f"{model_dir}/resnet34_peoplenet.onnx"
    labels_file = f"{model_dir}/labels.txt"

    # 检查文件是否存在
    engine_exists = os.path.exists(engine_file)
    onnx_exists = os.path.exists(onnx_file)
    labels_exists = os.path.exists(labels_file)

    if not (engine_exists or onnx_exists):
        print(f"❌ 警告: 模型文件不存在！")
        print(f"   引擎文件: {engine_exists}")
        print(f"   ONNX文件: {onnx_exists}")
        print(f"   标签文件: {labels_exists}")

    config_content = f'''[property]
# 模型配置 - 使用实际存在的文件
model-engine-file={engine_file}
onnx-file={onnx_file}
labelfile-path={labels_file}

# 输入配置 - 使用模型原始分辨率544x960 (注意顺序：高度;宽度)
infer-dims=3;544;960
batch-size=1
network-mode=2  # FP16模式
network-type=0  # 检测网络
num-detected-classes=3
interval=0
cluster-mode=1  # NMS聚类

# 修复边界框过大的关键配置
maintain-aspect-ratio=1
symmetric-padding=1
scaling-filter=1  # 使用双线性插值
scaling-compute-hw=1  # 使用GPU加速缩放

# 输出配置
gie-unique-id=1
output-tensor-meta=1

# 类别0 - person (主要检测目标)
[class-attrs-0]
pre-cluster-threshold=0.25
topk=100
nms-iou-threshold=0.45
detected-min-w=30
detected-min-h=60
detected-max-w=960
detected-max-h=544

# 类别1 - bag (降低检测，主要关注person)
[class-attrs-1]
pre-cluster-threshold=0.8
group-threshold=2
eps=0.5
minBoxes=2
roi-top-offset=0
roi-bottom-offset=0
detected-min-w=50
detected-min-h=50
detected-max-w=200
detected-max-h=200

# 类别2 - face (降低检测，主要关注person)
[class-attrs-2]
pre-cluster-threshold=0.8
group-threshold=2
eps=0.5
minBoxes=2
roi-top-offset=0
roi-bottom-offset=0
detected-min-w=30
detected-min-h=30
detected-max-w=150
detected-max-h=150
'''

    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"✅ 已创建/更新配置文件: {config_path}")
    print(f"   引擎文件: {engine_file}")
    print(f"   ONNX文件: {onnx_file}")
    print(f"   标签文件: {labels_file}")


# 检查模型文件是否存在
def check_model_files():
    detection_model_dir = "./models"
    detection_model_path = "./models/resnet34_peoplenet.onnx"
    # 尝试使用现有的引擎文件
    detection_engine_path = "./models/resnet34_peoplenet.onnx_b2_gpu0_fp16.engine"

    # 检测模型检查 - 优先使用引擎文件
    if os.path.exists(detection_engine_path):
        print(f"✅ 使用引擎文件: {detection_engine_path}")
        return True
    elif os.path.exists(detection_model_path):
        print(f"✅ 使用 ONNX 文件: {detection_model_path}")
        print("📝 将自动生成引擎文件")
        return True
    else:
        print(f"❌ 缺少检测模型:")
        print(f"   ONNX文件: {detection_model_path}")
        print(f"   引擎文件: {detection_engine_path}")
        print("📝 建议：")
        print("   1. 确保 ONNX 模型文件存在")
        print("   2. 或运行 deepstream-test3-app 生成引擎文件")
        print("   3. 检查模型文件权限")
        return False

    # 标签文件检查
    labels_file = "./models/labels.txt"
    if not os.path.exists(labels_file):
        print(f"❌ 缺少标签文件: {labels_file}")
        return False

    print("✅ 检测模型文件已找到")
    return True


# REID配置常量
REID_FEATURE_DIM = 512
REID_SIMILARITY_THRESHOLD = 0.85
ALPHA_TRACK = 0.2  # EMA系数，用于track级特征聚合

# 双层阈值配置 - 改善跨摄像头识别
REID_REACQUISITION_THRESHOLD = 0.75  # 【新增】用于跨摄像头寻回的较低阈值
REID_CONFIRMATION_THRESHOLD = 0.88  # 【修改】用于已锁定track持续确认的较高阈值 (可以比原来0.85还高一点)

PATROL_TIME_WINDOW = 3600
PATROL_MIN_OCCURRENCES = 3
PATROL_MIN_DURATION = 300
MAX_TRACK_HISTORY = 100


# --- 【新增】FaceFeatureExtractor类 ---
class FaceFeatureExtractor:
    """封装 Buffalo_L ONNX 模型的人脸特征提取功能"""

    def __init__(self, model_path=FACE_MODEL_PATH):
        self.model_path = model_path
        self.input_shape = (112, 112)  # Buffalo_L输入尺寸

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ 人脸模型文件未找到: {self.model_path}")

        try:
            # 使用GPU进行推理
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
            print(f"✅ 成功加载人脸模型: {self.model_path} on {self.session.get_providers()[0]}")

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            print(f"❌ 初始化人脸模型失败: {e}")
            self.session = None

    def preprocess(self, face_crop):
        """对人脸图像进行预处理"""
        # 调整大小并转换为RGB
        resized_face = cv2.resize(face_crop, self.input_shape)
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        # 归一化并调整维度 HWC -> CHW
        tensor = (rgb_face.astype(np.float32) - 127.5) / 128.0
        tensor = tensor.transpose(2, 0, 1)

        # 增加批次维度 CHW -> NCHW
        return np.expand_dims(tensor, axis=0)

    def extract_face_feature(self, face_crop):
        """
        提取人脸特征向量
        face_crop: BGR格式的numpy数组
        """
        if self.session is None or face_crop is None or face_crop.size == 0:
            return None

        try:
            # 使用优化的预处理流程
            input_tensor = self.preprocess(face_crop)

            # 使用ONNX Runtime进行高效推理
            feature = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]

            # L2 归一化，得到单位向量
            norm = np.linalg.norm(feature)
            if norm == 0:
                return None
            normalized_feature = feature / norm

            # 确保数据类型正确
            result = normalized_feature.astype(np.float32)
            return result
        except Exception as e:
            print(f"❌ 人脸特征提取推理失败: {e}")
            return None


class FastReIDDatabase:
    def __init__(self):
        # --- 保留所有旧的属性(V3及以前) ---
        self.track_cache = {}
        self.person_prototypes = {}
        self.recently_disappeared_tracks = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- 模型与推理相关 ---
        self.face_extractor = FaceFeatureExtractor()  # 假设FaceFeatureExtractor类已定义
        self.reid_model = None  # 将在initialize_reid_model中被赋值
        self.initialize_reid_model()
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- 调试与统计计数器 ---
        self.person_creation_count = 0
        self.person_match_count = 0  # 添加匹配次数计数器
        self.reid_inference_count = 0
        self.face_inference_count = 0
        self.total_frames_processed = 0

        # 【性能优化】周期性日志计数器
        self.log_counter = 0
        self.log_interval = 300  # 每300帧打印一次详细日志（约10秒）

        # --- 功能参数 ---
        self.EXTRACT_INTERVAL = 3  # 降低间隔，使ReID处理更频繁
        self.DISAPPEAR_TIMEOUT = 5.0
        self.CONFIRMATION_COUNT_THRESHOLD = 3

        # 【新增】跨摄像头ID一致性优化 - 动态阈值设置
        self.REID_REACQUISITION_THRESHOLD = 0.7  # 跨摄像头寻回阈值（降低至0.7，原0.75）
        self.REID_CONFIRMATION_THRESHOLD = 0.85  # 同一摄像头确认阈值（降低至0.85，保持较高避免误关联）

        # --- 【V4 & V4.1 核心参数整合】动态EMA 与 分层重置 ---
        # 动态EMA参数
        self.ALPHA_HIGH = 0.7  # 初始学习阶段的高alpha值
        self.ALPHA_LOW = 0.2  # 稳定阶段的低alpha值
        self.ALPHA_DECAY_UPDATES = 15  # 从高alpha衰减到低alpha所需的更新次数

        # --- 【改进1】跨摄像头域自适应（Domain Adaptation Matrix）---
        # 初始化一个N×N相机转换矩阵，记录不同摄像头之间的均值差异
        # Key格式: (cam_a, cam_b) 表示cam_a到cam_b的相似度调整因子
        self.domain_adaptation_matrix = {
            # 为常见摄像头对设置初始调整因子（可在运行时根据实际情况调整）
            (0, 1): 1.05,  # cam0 到 cam1 相似度乘 1.05
            (1, 0): 1.03,  # cam1 到 cam0 相似度乘 1.03
            (0, 2): 0.98,  # cam0 到 cam2 相似度乘 0.98
            (2, 0): 1.02,  # cam2 到 cam0 相似度乘 1.02
            (1, 2): 1.01,  # cam1 到 cam2 相似度乘 1.01
            (2, 1): 0.99,  # cam2 到 cam1 相似度乘 0.99
        }
        self.successful_domain_matches = defaultdict(list)  # 记录成功的跨摄像头匹配
        self.learning_rate = 0.01  # 域自适应学习率
        self.update_interval = 10  # 每10帧更新一次域自适应矩阵

        # 分层重置/清理参数
        self.CONFIDENCE_LEVELS = {
            "TRANSIENT": 0,  # 瞬时/游客: 仅靠几次ReID确认
            "CONFIRMED": 1,  # 已确认: 靠多次稳定ReID确认
            "FACE_VERIFIED": 2  # 人脸已验证: 最高可信度
        }
        self.INACTIVE_THRESHOLDS = {
            self.CONFIDENCE_LEVELS["TRANSIENT"]: 6 * 3600,  # 6小时后清除
            self.CONFIDENCE_LEVELS["CONFIRMED"]: 24 * 3600,  # 24小时后重置学习率
            self.CONFIDENCE_LEVELS["FACE_VERIFIED"]: float('inf')  # 永不过期
        }

    #
    # --- 【V4.1 新增/修改的核心函数】 ---
    #

    def _calculate_dynamic_alpha(self, update_count):
        """
        【V4 功能】根据更新次数计算动态的ALPHA_EMA值。
        """
        # 阶段1: 快速学习 (刚创建或刚重置后)
        if update_count < 5:
            return self.ALPHA_HIGH

        # 阶段2: 线性衰减
        if update_count < (5 + self.ALPHA_DECAY_UPDATES):
            progress = (update_count - 5) / self.ALPHA_DECAY_UPDATES
            alpha = self.ALPHA_HIGH - (self.ALPHA_HIGH - self.ALPHA_LOW) * progress
            return alpha

        # 阶段3: 稳定学习
        return self.ALPHA_LOW

    def _calculate_adaptive_alpha(self, camera_id, person_data):
        """
        【V5.0】为每个摄像头计算自适应的ALPHA值
        """
        # 基于摄像头的更新频率和质量调整alpha
        camera_updates = person_data['camera_weights'].get(camera_id, 1.0)

        # 高频更新的摄像头使用较小的alpha（更稳定）
        # 低频更新的摄像头使用较大的alpha（快速适应）
        if camera_updates > 10:
            return self.ALPHA_LOW
        elif camera_updates > 5:
            return (self.ALPHA_HIGH + self.ALPHA_LOW) / 2
        else:
            return self.ALPHA_HIGH

    def _update_global_prototype(self, person_id):
        """
        计算全局原型（新增辅助方法）
        """
        data = self.person_prototypes[person_id]
        camera_protos = data.get('camera_prototypes', {})
        weights = data.get('camera_weights', {})

        if not camera_protos:
            return

        # 加权平均
        weighted_sum = np.zeros(REID_FEATURE_DIM, dtype=np.float32)
        total_weight = 0.0

        for cam_id, proto in camera_protos.items():
            weight = weights.get(cam_id, 1.0)
            weighted_sum += proto * weight
            total_weight += weight

        if total_weight > 0:
            global_proto = weighted_sum / total_weight
            data['global_prototype'] = global_proto / np.linalg.norm(global_proto)
            print(f"   更新全局原型，整合 {len(camera_protos)} 个摄像头")

    def cleanup_inactive_prototypes(self):
        """
        【V4.1 功能】执行分层清理：
        - FACE_VERIFIED (Lvl 2): 永不处理
        - CONFIRMED (Lvl 1): 长时间不活跃则重置学习率
        - TRANSIENT (Lvl 0): 较短时间不活跃则直接删除
        """
        current_time = time.time()
        ids_to_delete = []
        reset_count = 0

        # 使用 list(self.person_prototypes.items()) 来允许在循环中删除
        for person_id, data in list(self.person_prototypes.items()):
            confidence = data.get('confidence_level', self.CONFIDENCE_LEVELS["TRANSIENT"])
            last_update = data.get('last_update_time', current_time)
            inactive_duration = current_time - last_update

            threshold = self.INACTIVE_THRESHOLDS.get(confidence, float('inf'))

            if inactive_duration > threshold:
                if confidence == self.CONFIDENCE_LEVELS["TRANSIENT"]:
                    ids_to_delete.append(person_id)
                elif confidence == self.CONFIDENCE_LEVELS["CONFIRMED"]:
                    data['update_count'] = 0
                    data['current_alpha'] = self.ALPHA_HIGH
                    reset_count += 1
                    print(f"🕒 原型重置: {person_id}(Lvl:{confidence}) 因不活跃而被重置学习率。")

        for person_id in ids_to_delete:
            if person_id in self.person_prototypes:
                del self.person_prototypes[person_id]
                print(f"🗑️ 访客ID删除: {person_id} 因长期不活跃被清除。")

        if reset_count > 0 or len(ids_to_delete) > 0:
            print(f"🧹 本次清理: 重置了 {reset_count} 个已确认ID, 删除了 {len(ids_to_delete)} 个访客ID。")

    def update_prototypes(self, person_id, track_id, reid_feature, face_feature=None, camera_id=-1):
        """
        【V5.0 优化版】支持多摄像头独立原型管理
        """
        current_time = time.time()

        if person_id not in self.person_prototypes:
            # 创建新人员（兼容两种数据结构）
            confidence = self.CONFIDENCE_LEVELS["TRANSIENT"]
            if face_feature is not None:
                confidence = self.CONFIDENCE_LEVELS["FACE_VERIFIED"]

            self.person_prototypes[person_id] = {
                'camera_prototypes': {camera_id: reid_feature} if reid_feature is not None else {},  # 新字段
                'global_prototype': reid_feature,  # 新字段：初始时等于第一个reid特征
                'camera_weights': {camera_id: 1.0} if reid_feature is not None else {},  # 新字段
                'prototypes': {camera_id: reid_feature} if reid_feature is not None else {},  # 保留旧字段兼容性
                'face_prototype': face_feature,
                'camera_id': camera_id,
                'locked_by': track_id,
                'last_update_time': current_time,
                'update_count': 1,
                'current_alpha': self._calculate_dynamic_alpha(1),
                'confidence_level': confidence,
                'history': deque(maxlen=20),
                # 新增统计字段
                'first_camera_id': camera_id,
                'camera_appearance_count': {camera_id: 1},
                'last_seen_camera': camera_id
            }
            print(f"🆕 创建人员: {person_id}, Camera:{camera_id}")
        else:
            # 更新已有人员
            data = self.person_prototypes[person_id]
            update_count = data.get('update_count', 0) + 1
            alpha = self._calculate_dynamic_alpha(update_count)

            data.update({
                'update_count': update_count,
                'current_alpha': alpha,
                'last_update_time': current_time,
                'locked_by': track_id,
                'last_seen_camera': camera_id
            })

            # 更新摄像头出现统计
            if 'camera_appearance_count' not in data:
                data['camera_appearance_count'] = {}
            data['camera_appearance_count'][camera_id] = data['camera_appearance_count'].get(camera_id, 0) + 1

            # 更新ReID原型
            if reid_feature is not None:
                # 确保camera_prototypes字段存在（兼容旧数据）
                if 'camera_prototypes' not in data:
                    data['camera_prototypes'] = {}
                if 'camera_weights' not in data:
                    data['camera_weights'] = {}

                # 更新当前摄像头的原型
                if camera_id not in data['camera_prototypes']:
                    data['camera_prototypes'][camera_id] = reid_feature
                    data['camera_weights'][camera_id] = 1.0
                    print(f"   为摄像头 {camera_id} 创建新原型")
                else:
                    # EMA更新
                    old_proto = data['camera_prototypes'][camera_id]
                    new_proto = alpha * reid_feature + (1 - alpha) * old_proto
                    data['camera_prototypes'][camera_id] = new_proto / np.linalg.norm(new_proto)
                    # 增加权重（最多到2.0）
                    data['camera_weights'][camera_id] = min(data['camera_weights'][camera_id] + 0.05, 2.0)
                    print(f"   更新摄像头 {camera_id} 原型，Alpha:{alpha:.3f}")

                # 更新全局原型（所有摄像头原型的加权平均）
                self._update_global_prototype(person_id)

                # 同时更新旧的prototypes字段（保持兼容性）
                data['prototypes'] = data['camera_prototypes'].copy()

            # 更新人脸原型
            if face_feature is not None:
                if data.get('face_prototype') is None:
                    data['face_prototype'] = face_feature
                    print(f"   添加人脸特征")
                else:
                    face_alpha = 0.3
                    old_face = data['face_prototype']
                    new_face = face_alpha * face_feature + (1 - face_alpha) * old_face
                    data['face_prototype'] = new_face / np.linalg.norm(new_face)
                    print(f"   更新人脸特征，Alpha:{face_alpha:.3f}")

            # 检查是否实现跨摄像头确认
            if len(data.get('camera_appearance_count', {})) >= 2:
                data['cross_camera_confirmed'] = True
                print(f"   ✅ 跨摄像头确认！出现在 {len(data['camera_appearance_count'])} 个摄像头")

    #
    # --- 【保留并适配 V3 的核心处理逻辑】 ---
    #

    def initialize_reid_model(self):
        """初始化真正的OSNet-IBN模型"""
        try:
            # 设置模型路径
            self.model_path = "/home/fanrrrrrrr/workspace/aihub/yunhe/reid_models"
            self.model_file = "osnet_ibn_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
            self.full_path = os.path.join(self.model_path, self.model_file)

            if not os.path.exists(self.full_path):
                print(f"⚠️ 模型文件未找到: {self.full_path}")
                print("将使用备用特征提取")
                self.reid_model = None
                return

            # 创建真正的OSNet-x1.0模型
            self.reid_model = OSNetIBN(
                blocks=[OSBlock, OSBlock],
                layers=[2, 2, 2],
                channels=[64, 256, 384, 512],
                feature_dim=REID_FEATURE_DIM
            )

            # 加载模型权重
            checkpoint = torch.load(self.full_path, map_location=self.device)

            # 处理权重格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # 创建新的状态字典，匹配我们的模型结构
            new_state_dict = {}

            # 处理层名称映射
            for key, value in state_dict.items():
                new_key = key

                # 移除module前缀
                new_key = new_key.replace('module.', '')

                # 转换键名格式
                if new_key.startswith('conv'):
                    # 处理conv层
                    parts = new_key.split('.')
                    if len(parts) >= 3:
                        layer_num = parts[0]  # conv2, conv3, etc.
                        block_idx = parts[1]  # 0, 1, 2, etc.

                        if layer_num in ['conv2', 'conv3', 'conv4']:
                            # 处理OSBlock内部的权重
                            if 'conv' in parts[2]:
                                # 转换格式
                                new_parts = [layer_num, block_idx]

                                # 处理conv内部的结构
                                if 'conv1' in parts[2]:
                                    new_parts.extend(['0', '0'])  # conv1.conv
                                elif 'conv2a' in parts[2]:
                                    new_parts.extend(['0', '1'])  # conv2a
                                elif 'conv2b' in parts[2]:
                                    new_parts.extend(['0', '2'])  # conv2b
                                elif 'conv2c' in parts[2]:
                                    new_parts.extend(['0', '3'])  # conv2c
                                elif 'conv2d' in parts[2]:
                                    new_parts.extend(['0', '4'])  # conv2d
                                elif 'conv3' in parts[2]:
                                    new_parts.extend(['0', '5'])  # conv3

                                # 添加剩余部分
                                new_parts.extend(parts[3:])
                                new_key = '.'.join(new_parts)

                # 处理fc层
                if new_key.startswith('fc'):
                    new_key = new_key.replace('fc.', 'fc.')

                # 只保存匹配的权重
                if 'classifier' not in new_key:  # 跳过分类器
                    new_state_dict[new_key] = value

            # 加载匹配的权重
            model_dict = self.reid_model.state_dict()

            # 过滤出形状匹配的权重
            matched_dict = {}
            for key, value in new_state_dict.items():
                if key in model_dict and value.shape == model_dict[key].shape:
                    matched_dict[key] = value

            # 加载权重
            model_dict.update(matched_dict)
            missing, unexpected = self.reid_model.load_state_dict(model_dict, strict=False)

            if missing:
                print(f"⚠️ 缺失权重: {len(missing)}个")
            if unexpected:
                print(f"⚠️ 意外权重: {len(unexpected)}个")

            # 保留分类层，用于完整的OSNet-IBN模型

            self.reid_model.eval()
            self.reid_model.to(self.device)

            print(f"✅ 成功加载官方OSNet-IBN-x1.0模型")
            print(f"   模型: {self.model_file}")
            print(f"   输出维度: {REID_FEATURE_DIM}")
            print(f"   成功加载权重: {len(matched_dict)}个")

        except Exception as e:
            print(f"⚠️ 初始化REID模型失败: {e}")
            import traceback
            traceback.print_exc()
            print("将使用备用特征提取")
            self.reid_model = None

    def extract_person_feature(self, frame, bbox):
        """
        从单帧里提取 512 维 OSNet 特征（TTA + 优化版过滤）
        frame : BGR ndarray
        bbox  : [x, y, w, h]  —— 已确保 bbox 在画面范围内
        """
        # 1. 基本裁剪合法性
        x, y, w, h = map(int, bbox)
        H, W = frame.shape[:2]

        if w <= 10 or h <= 20 or x >= W or y >= H:  # 过滤掉极小的框
            return None

        # 2. 【核心修改】放宽对"过宽 / 过瘦"的框的限制
        ratio = w / (h + 1e-6)
        # 原始限制: if ratio > 0.75 or ratio < 0.30:
        # 新的、更宽松的限制，允许更广泛的姿态和检测误差
        if ratio > 1.2 or ratio < 0.2:
            return None

        # 3. 上下各留 10 % PAD，避免裁掉头/脚
        pad_h = int(0.10 * h)
        y0 = max(0, y - pad_h)
        y1 = min(H, y + h + pad_h)

        # 左右也增加一点padding，防止身体被截断
        pad_w = int(0.05 * w)
        x0 = max(0, x - pad_w)
        x1 = min(W, x + w + pad_w)

        person_crop = frame[y0:y1, x0:x1]

        if person_crop.size == 0:
            return None

        # 4. OSNet 前向推理（含水平翻转 TTA）
        try:
            if self.reid_model is None:
                return None

            # 正式 OSNet - 优化推理流程
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 使用PyTorch的高效推理模式
                if hasattr(torch, 'inference_mode'):
                    with torch.inference_mode():
                        feat = self.reid_model(tensor)
                        feat_flip = self.reid_model(torch.flip(tensor, dims=[3]))
                        feat = F.normalize(feat + feat_flip, p=2, dim=1)
                else:
                    # 兼容旧版本PyTorch
                    feat = self.reid_model(tensor)
                    feat_flip = self.reid_model(torch.flip(tensor, dims=[3]))
                    feat = F.normalize(feat + feat_flip, p=2, dim=1)

            # 将特征转换为NumPy数组并确保数据类型正确
            feature = feat.cpu().numpy()[0].astype(np.float32)
            self.reid_inference_count += 1  # 增加推理计数器
            return feature

        except Exception as e:
            print(f"❌ ReID 特征提取推理失败: {e}")
            return None

    def compute_cosine_similarity(self, feature1, feature2):
        """计算两个特征的余弦相似度"""
        if feature1 is None or feature2 is None:
            return 0.0

        # 计算余弦相似度：点积 / (范数1 * 范数2)
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)

        # 避免除零错误
        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_similarity = dot_product / (norm1 * norm2)

        # 确保相似度在[-1, 1]范围内
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        return cosine_similarity

    def find_best_matches(self, reid_feature, face_feature, current_track_id=None, active_track_ids=None, camera_id=-1):
        """
        【V5.0 优化版】改进的跨摄像头匹配策略
        保持原有接口，增强跨摄像头识别能力
        """
        if active_track_ids is None:
            active_track_ids = set()

        # 获取当前track的信息
        track_info = self.track_cache.get(current_track_id, {})
        current_camera_id = camera_id  # 直接使用传入的camera_id

        print(f"🔍 [MATCH-V5] Track {current_track_id}: 开始匹配查询")
        print(f"   摄像头: {current_camera_id}")
        print(f"   原型数量: {len(self.person_prototypes)}")
        print(f"   活跃Tracks: {len(active_track_ids)}")

        if not self.person_prototypes:
            return (None, 0.0), (None, 0.0)

        # --- 人脸匹配（保持原有逻辑）---
        best_face_match_id = None
        best_face_sim = 0.0
        if face_feature is not None:
            for person_id, data in self.person_prototypes.items():
                face_proto = data.get('face_prototype')
                if face_proto is not None:
                    sim = self.compute_cosine_similarity(face_feature, face_proto)
                    if sim > best_face_sim:
                        best_face_sim = sim
                        best_face_match_id = person_id

        # --- ReID匹配（改进版）---
        best_reid_match_id = None
        best_reid_sim = 0.0
        best_match_type = None  # 记录匹配类型

        if reid_feature is not None:
            for person_id, data in self.person_prototypes.items():
                # 检查原型锁定（保持原有逻辑）
                locked_by_track = data.get('locked_by')
                is_locked_by_another = (
                        locked_by_track is not None and
                        locked_by_track in active_track_ids and
                        locked_by_track != current_track_id
                )

                if is_locked_by_another:
                    print(f"🚫 [LOCK] 跳过原型 {person_id}，已被Track {locked_by_track} 锁定")
                    continue

                # 【新增】计算多种相似度
                similarities = []

                # 1. 检查同摄像头原型（最可靠）
                camera_prototypes = data.get('camera_prototypes', {})
                if not camera_prototypes:
                    # 兼容旧版本：如果没有camera_prototypes，尝试使用prototypes
                    old_prototypes = data.get('prototypes', {})
                    if old_prototypes:
                        camera_prototypes = old_prototypes

                if current_camera_id in camera_prototypes:
                    same_cam_proto = camera_prototypes[current_camera_id]
                    same_cam_sim = self.compute_cosine_similarity(reid_feature, same_cam_proto)
                    similarities.append(('same_camera', same_cam_sim, 1.0))  # 权重1.0
                    print(f"   同摄像头匹配 {person_id}: {same_cam_sim:.3f}")

                # 2. 使用全局原型（如果存在）
                global_proto = data.get('global_prototype')
                if global_proto is not None:
                    global_sim = self.compute_cosine_similarity(reid_feature, global_proto)
                    similarities.append(('global', global_sim, 0.85))  # 权重0.85
                    print(f"   全局原型匹配 {person_id}: {global_sim:.3f}")

                # 3. 跨摄像头原型匹配（带域自适应调整）
                for cam_id, proto in camera_prototypes.items():
                    if cam_id != current_camera_id:
                        cross_sim = self.compute_cosine_similarity(reid_feature, proto)
                        # 【改进1】应用域自适应调整
                        adapt_key = (cam_id, current_camera_id)  # 从cam_id到current_camera_id的转换
                        adjust = self.domain_adaptation_matrix.get(adapt_key, 1.0)
                        cross_sim_adjusted = cross_sim * adjust
                        similarities.append(('cross_camera', cross_sim_adjusted, 0.7))  # 权重0.7
                        print(
                            f"   跨摄像头匹配 {person_id} (cam{cam_id}): {cross_sim:.3f} -> {cross_sim_adjusted:.3f} (调整:{adjust:.3f})")

                        # 【改进1】记录跨摄像头匹配的成功信息，用于后续学习
                        if cross_sim_adjusted > self.REID_CONFIRMATION_THRESHOLD * 0.9:  # 较高的匹配阈值
                            self.successful_domain_matches[adapt_key].append(cross_sim)

                # 【新增】计算加权平均相似度
                if similarities:
                    # 选择最佳匹配方式
                    best_match = max(similarities, key=lambda x: x[1] * x[2])  # 考虑权重
                    match_type, sim_value, weight = best_match

                    # 如果人脸也匹配，提升置信度
                    if best_face_match_id == person_id and best_face_sim > FACE_SIMILARITY_THRESHOLD:
                        sim_value = sim_value * 0.9 + 0.1  # 轻微提升
                        print(f"   ✨ 人脸增强 {person_id}: +0.1")

                    if sim_value > best_reid_sim:
                        best_reid_sim = sim_value
                        best_reid_match_id = person_id
                        best_match_type = match_type

        # 【新增】根据匹配类型调整阈值判断
        if best_match_type == 'same_camera':
            # 同摄像头使用标准阈值
            effective_threshold = self.REID_CONFIRMATION_THRESHOLD
        elif best_match_type == 'global':
            # 全局原型使用稍低阈值
            effective_threshold = self.REID_CONFIRMATION_THRESHOLD * 0.95
        else:  # cross_camera
            # 跨摄像头使用更低阈值
            effective_threshold = self.REID_REACQUISITION_THRESHOLD

        print(f"✅ [MATCH-V5] 结果:")
        print(f"   人脸: {best_face_match_id} ({best_face_sim:.3f})")
        print(
            f"   ReID: {best_reid_match_id} ({best_reid_sim:.3f}, 类型:{best_match_type}, 阈值:{effective_threshold:.3f})")

        # 【改进1】定期更新域自适应矩阵
        if self.total_frames_processed % self.update_interval == 0:
            self.update_domain_adaptation_matrix()

        # 返回时保持原有接口
        return (best_face_match_id, best_face_sim), (best_reid_match_id, best_reid_sim)

    def update_domain_adaptation_matrix(self):
        """【改进1】更新域自适应矩阵 - 根据成功匹配学习调整因子"""
        for adapt_key, similarity_scores in self.successful_domain_matches.items():
            if len(similarity_scores) >= 3:  # 至少需要3个成功样本才更新
                # 计算平均相似度
                avg_sim = np.mean(similarity_scores)
                # 如果平均相似度较低，说明该摄像头对需要补偿
                if avg_sim < 0.8:  # 较低的匹配阈值
                    # 学习调整方向：如果匹配度低，增加调整因子
                    current_adjust = self.domain_adaptation_matrix.get(adapt_key, 1.0)
                    # 使用启发式方法：匹配度越低，调整因子越大
                    target_adjust = min(1.5, 1.0 + (0.8 - avg_sim) * 0.5)  # 最大不超过1.5
                    # 平滑更新
                    new_adjust = current_adjust + self.learning_rate * (target_adjust - current_adjust)
                    self.domain_adaptation_matrix[adapt_key] = new_adjust
                    print(
                        f"📊 [DOMAIN-ADAPT] 更新域自适应矩阵 {adapt_key}: {current_adjust:.3f} -> {new_adjust:.3f} (基于{len(similarity_scores)}个匹配，平均相似度:{avg_sim:.3f})")
                # 清空已处理的记录
                self.successful_domain_matches[adapt_key] = []

        # 定期打印域自适应矩阵状态
        if self.total_frames_processed % 100 == 0 and self.domain_adaptation_matrix:
            print(f"📊 [DOMAIN-ADAPT] 当前域自适应矩阵状态:")
            for (cam_a, cam_b), adjust in sorted(self.domain_adaptation_matrix.items()):
                print(f"   cam{cam_a}→cam{cam_b}: {adjust:.3f}")

    def check_patrol_behavior(self, person_id, current_time):
        """检查徘徊行为"""
        if person_id not in self.person_prototypes:
            return False

        # 【修改】从所有摄像头的原型中获取特征
        prototypes = self.person_prototypes[person_id].get('prototypes', {})
        features = list(prototypes.values()) if prototypes else []
        if len(features) < PATROL_MIN_OCCURRENCES:
            return False

        recent_detections = [(current_time, features[0], None)]

        if len(recent_detections) < PATROL_MIN_OCCURRENCES:
            return False

        start_time = current_time - 300  # 假设5分钟前开始
        end_time = current_time
        duration = end_time - start_time

        if duration >= PATROL_MIN_DURATION:
            bboxes = [None]  # 临时适配
            if self._calculate_patrol_area(bboxes) > 0.1:
                return True

        return False

    def _calculate_patrol_area(self, bboxes):
        """计算人员活动区域范围"""
        if not bboxes:
            return 0.0

        x_coords = [b[0] + b[2] / 2 for b in bboxes]
        y_coords = [b[1] + b[3] / 2 for b in bboxes]

        if len(x_coords) < 2:
            return 0.0

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        return (x_range * y_range) / (1280 * 720)

    def cleanup_stale_tracks(self, alive_track_ids):
        """【优化版】清理track，并将消失的track信息存入"短期记忆"以应对遮挡"""
        stale_tracks = []
        current_time = time.time()

        # 找出消失的track
        for track_id in self.track_cache.keys():
            if track_id not in alive_track_ids:
                stale_tracks.append(track_id)

        # 处理消失的track
        for track_id in stale_tracks:
            track_info = self.track_cache.get(track_id)
            if track_info:
                person_id = track_info.get('person_id')
                # 只有被成功识别过的人，才有价值被记入短期记忆
                if person_id and person_id in self.person_prototypes:
                    print(f"🧠 Track {track_id} ({person_id}) 消失，存入短期记忆...")
                    self.recently_disappeared_tracks[track_id] = {
                        'person_id': person_id,
                        'disappeared_time': current_time,
                        # 保存消失前的最后一个原型，用于优先比对
                        'last_prototype': list(self.person_prototypes[person_id].get('prototypes', {}).values())[0] if
                        self.person_prototypes[person_id].get('prototypes') else None
                    }

            # 从活跃缓存中删除
            del self.track_cache[track_id]

        # 清理"短期记忆"中超时的track
        timeout_tracks = []
        for track_id, data in self.recently_disappeared_tracks.items():
            if current_time - data['disappeared_time'] > self.DISAPPEAR_TIMEOUT:
                timeout_tracks.append(track_id)

        for track_id in timeout_tracks:
            print(f"💭 Track {track_id} 从短期记忆中清除 (超时)。")
            del self.recently_disappeared_tracks[track_id]

    def cleanup_person_registry(self):
        """清理人员注册表中的过期数据 - 已弃用，使用新结构"""
        # 这个方法现在只保留兼容性，不再执行实际操作
        # 原型由cleanup_stale_tracks和系统逻辑自动管理
        pass

    def find_in_short_term_memory(self, feature):
        """在短期记忆中查找匹配项，用于处理遮挡重现"""
        if feature is None or not self.recently_disappeared_tracks:
            print(
                f"🔍 [MEMORY] 短期记忆检查跳过: feature={feature is not None}, 记录数={len(self.recently_disappeared_tracks)}")
            return None, 0.0

        best_match_person_id = None
        best_similarity = 0.0
        matched_disappeared_track_id = None

        print(f"🔍 [MEMORY] 开始在短期记忆中搜索，记录数={len(self.recently_disappeared_tracks)}")

        for track_id, data in self.recently_disappeared_tracks.items():
            last_prototype = data.get('last_prototype')
            if last_prototype is None:
                print(f"🔍 [MEMORY] 跳过记录 {track_id}: 无原型数据")
                continue

            similarity = self.compute_cosine_similarity(feature, last_prototype)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_person_id = data.get('person_id')
                matched_disappeared_track_id = track_id

        if best_similarity >= self.REID_REACQUISITION_THRESHOLD:
            # 成功在短期记忆中找到匹配！
            # 从短期记忆中移除，因为它已经被"认领"了
            del self.recently_disappeared_tracks[matched_disappeared_track_id]
            return best_match_person_id, best_similarity
        else:
            return None, 0.0

    def save_database(self, filename=None, create_backup=True):
        """
        【V5.0 适配】保存数据库时，支持多摄像头原型结构
        """
        if filename is None:
            filename = "fastreid_database.json"

        temp_filename = filename + ".tmp"

        try:
            # 准备要保存的数据
            save_data = {
                'version': '5.0',  # 更新版本号，标识新的数据结构
                'timestamp': time.time(),
                'person_prototypes': {},
                'track_cache': {},
                'person_creation_count': self.person_creation_count,
                'total_frames_processed': getattr(self, 'total_frames_processed', 0),
                'reid_inference_count': getattr(self, 'reid_inference_count', 0),
                'face_inference_count': getattr(self, 'face_inference_count', 0),
                # 新增摄像头相关统计
                'camera_statistics': getattr(self, 'camera_statistics', {}),
                'domain_adaptation_data': getattr(self, 'domain_adaptation_matrix', {})
            }

            # 序列化person_prototypes（支持V5.0多摄像头原型）
            for person_id, data in self.person_prototypes.items():
                # 处理多摄像头原型
                camera_prototypes_serialized = {}
                camera_protos = data.get('camera_prototypes', {})

                # 序列化每个摄像头的原型
                for cam_id, proto in camera_protos.items():
                    if proto is not None:
                        norm = np.linalg.norm(proto)
                        if norm > 0:
                            proto = proto / norm
                        camera_prototypes_serialized[str(cam_id)] = proto.tolist()

                # 序列化全局原型
                global_proto = data.get('global_prototype')
                if global_proto is not None:
                    norm = np.linalg.norm(global_proto)
                    if norm > 0:
                        global_proto = global_proto / norm
                    global_proto_serialized = global_proto.tolist()
                else:
                    global_proto_serialized = None

                # 序列化人脸原型
                face_proto = data.get('face_prototype')
                if face_proto is not None:
                    norm = np.linalg.norm(face_proto)
                    if norm > 0:
                        face_proto = face_proto / norm
                    face_proto_serialized = face_proto.tolist()
                else:
                    face_proto_serialized = None

                # 序列化摄像头权重
                camera_weights_serialized = {}
                for cam_id, weight in data.get('camera_weights', {}).items():
                    camera_weights_serialized[str(cam_id)] = float(weight)

                save_data['person_prototypes'][person_id] = {
                    'camera_prototypes': camera_prototypes_serialized,  # 多摄像头原型
                    'global_prototype': global_proto_serialized,  # 全局融合原型
                    'face_prototype': face_proto_serialized,
                    'camera_weights': camera_weights_serialized,  # 摄像头权重
                    'locked_by': data.get('locked_by'),
                    'last_update_time': data.get('last_update_time'),
                    'update_count': data.get('update_count', 0),
                    'current_alpha': data.get('current_alpha', self.ALPHA_LOW),
                    'confidence_level': data.get('confidence_level', self.CONFIDENCE_LEVELS["TRANSIENT"]),
                    # 新增字段
                    'first_camera_id': data.get('first_camera_id', -1),  # 首次出现的摄像头
                    'camera_appearance_count': data.get('camera_appearance_count', {}),  # 各摄像头出现次数
                    'last_seen_camera': data.get('last_seen_camera', -1),  # 最后出现的摄像头
                    'cross_camera_confirmed': data.get('cross_camera_confirmed', False)  # 是否跨摄像头确认
                }

            # 序列化track_cache
            for track_id, cache_data in self.track_cache.items():
                # 确保track_id是字符串
                save_data['track_cache'][str(track_id)] = {
                    'person_id': cache_data.get('person_id'),
                    'status': cache_data.get('status'),
                    'camera_id': cache_data.get('camera_id', -1),
                    'last_feat_frame': cache_data.get('last_feat_frame', -1),
                    'confirming_id': cache_data.get('confirming_id'),
                    'confirming_count': cache_data.get('confirming_count', 0)
                }

            # 写入临时文件
            with open(temp_filename, 'w') as f:
                json.dump(save_data, f, indent=2, default=numpy_encoder)

            # 原子化操作
            os.replace(temp_filename, filename)

            # 创建备份
            if create_backup:
                backup_filename = filename.replace('.json', f'_backup_{int(time.time())}.json')
                import shutil
                shutil.copyfile(filename, backup_filename)

            # 统计信息
            total_cameras = set()
            for person_data in save_data['person_prototypes'].values():
                total_cameras.update(person_data.get('camera_prototypes', {}).keys())

            print(f"💾 V5.0数据库已保存至 {filename}")
            print(f"   人员数: {len(save_data['person_prototypes'])}")
            print(f"   活跃Track: {len(save_data['track_cache'])}")
            print(f"   涉及摄像头: {len(total_cameras)}个")

            return True

        except Exception as e:
            print(f"❌ 保存数据库到 {filename} 失败: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return False

    def load_database(self, filename=None):
        """
        【V5.0 适配】加载数据库，支持向后兼容
        """
        if filename is None:
            filename = "fastreid_database.json"

        if not os.path.exists(filename):
            print(f"ℹ️ 数据库文件 {filename} 不存在")
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # 检查版本以确定加载策略
            version = data.get('version', '1.0')
            print(f"📂 加载数据库版本: {version}")

            self.person_prototypes.clear()
            self.track_cache.clear()

            # 处理person_prototypes
            for person_id, proto_data in data.get('person_prototypes', {}).items():

                if version >= '5.0':
                    # V5.0格式：多摄像头原型
                    camera_prototypes = {}
                    for cam_id_str, proto_list in proto_data.get('camera_prototypes', {}).items():
                        if proto_list:
                            camera_prototypes[int(cam_id_str)] = np.array(proto_list, dtype=np.float32)

                    global_proto = None
                    if proto_data.get('global_prototype'):
                        global_proto = np.array(proto_data['global_prototype'], dtype=np.float32)

                    face_proto = None
                    if proto_data.get('face_prototype'):
                        face_proto = np.array(proto_data['face_prototype'], dtype=np.float32)

                    camera_weights = {}
                    for cam_id_str, weight in proto_data.get('camera_weights', {}).items():
                        camera_weights[int(cam_id_str)] = float(weight)

                    self.person_prototypes[person_id] = {
                        'camera_prototypes': camera_prototypes,
                        'global_prototype': global_proto,
                        'face_prototype': face_proto,
                        'camera_weights': camera_weights,
                        'locked_by': proto_data.get('locked_by'),
                        'last_update_time': proto_data.get('last_update_time', time.time()),
                        'update_count': proto_data.get('update_count', 0),
                        'current_alpha': proto_data.get('current_alpha', self.ALPHA_LOW),
                        'confidence_level': proto_data.get('confidence_level', self.CONFIDENCE_LEVELS["TRANSIENT"]),
                        'history': deque(maxlen=20),
                        'first_camera_id': proto_data.get('first_camera_id', -1),
                        'camera_appearance_count': proto_data.get('camera_appearance_count', {}),
                        'last_seen_camera': proto_data.get('last_seen_camera', -1),
                        'cross_camera_confirmed': proto_data.get('cross_camera_confirmed', False)
                    }

                else:
                    # 旧版本格式兼容（V4.x及以下）
                    print(f"⚠️ 检测到旧版本数据，进行格式转换...")

                    # 处理旧版本的单一原型或prototypes字段
                    reid_proto = None
                    if 'reid_prototype' in proto_data and proto_data['reid_prototype']:
                        reid_proto = np.array(proto_data['reid_prototype'], dtype=np.float32)
                    elif 'prototypes' in proto_data:
                        # V4.3格式
                        prototypes = proto_data['prototypes']
                        if isinstance(prototypes, dict) and prototypes:
                            # 取第一个原型作为默认
                            first_proto = list(prototypes.values())[0]
                            if first_proto:
                                reid_proto = np.array(first_proto, dtype=np.float32)

                    face_proto = None
                    if proto_data.get('face_prototype'):
                        face_proto = np.array(proto_data['face_prototype'], dtype=np.float32)

                    # 转换为新格式
                    camera_prototypes = {}
                    if reid_proto is not None:
                        # 假设原始摄像头ID为0
                        camera_prototypes[0] = reid_proto

                    self.person_prototypes[person_id] = {
                        'camera_prototypes': camera_prototypes,
                        'global_prototype': reid_proto,  # 使用单一原型作为全局原型
                        'face_prototype': face_proto,
                        'camera_weights': {0: 1.0} if camera_prototypes else {},
                        'locked_by': proto_data.get('locked_by'),
                        'last_update_time': proto_data.get('last_update_time', time.time()),
                        'update_count': proto_data.get('update_count', 20),
                        'current_alpha': proto_data.get('current_alpha', self.ALPHA_LOW),
                        'confidence_level': proto_data.get('confidence_level', self.CONFIDENCE_LEVELS["CONFIRMED"]),
                        'history': deque(maxlen=20),
                        'first_camera_id': proto_data.get('camera_id', 0),
                        'camera_appearance_count': {},
                        'last_seen_camera': 0,
                        'cross_camera_confirmed': False
                    }

            # 处理track_cache
            for track_id_str, cache_data in data.get('track_cache', {}).items():
                # 转换track_id为整数
                try:
                    track_id = int(track_id_str)
                except:
                    track_id = track_id_str

                self.track_cache[track_id] = cache_data

            # 更新统计数据
            self.person_creation_count = data.get('person_creation_count', len(self.person_prototypes))
            self.total_frames_processed = data.get('total_frames_processed', 0)
            self.reid_inference_count = data.get('reid_inference_count', 0)
            self.face_inference_count = data.get('face_inference_count', 0)

            # 加载新增的摄像头相关数据（如果有）
            if version >= '5.0':
                self.camera_statistics = data.get('camera_statistics', {})
                self.domain_adaptation_matrix = data.get('domain_adaptation_data', {})

            # 更新person_creation_count
            if self.person_prototypes:
                max_id = max([int(pid.split('_')[-1]) for pid in self.person_prototypes.keys()
                              if pid.startswith('person_')], default=0)
                self.person_creation_count = max(self.person_creation_count, max_id)

            # 统计信息
            total_persons = len(self.person_prototypes)
            total_cameras = set()
            for person_data in self.person_prototypes.values():
                total_cameras.update(person_data.get('camera_prototypes', {}).keys())

            print(f"✅ V5.0数据库加载完成: {filename}")
            print(f"   人员数: {total_persons}")
            print(f"   涉及摄像头: {len(total_cameras)}个")
            if version < '5.0':
                print(f"   ⚠️ 已从版本 {version} 自动升级到 5.0")

            return True

        except Exception as e:
            print(f"❌ 数据库加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def export_database_summary(self, filename="database_summary.txt"):
        """导出数据库摘要"""
        try:
            summary_lines = [
                "FastReID 数据库摘要",
                "=" * 50,
                f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"数据库版本: 2.0",
                f"人员原型: {len(self.person_prototypes)}",
                f"Track缓存: {len(self.track_cache)}",
                f"创建次数: {self.person_creation_count}",
                f"匹配次数: {getattr(self, 'person_match_count', 0)}",  # 使用getattr避免未定义错误
                f"处理帧数: {self.total_frames_processed}",
                f"推理次数: {self.reid_inference_count}",
                "",
                "人员原型列表:"
            ]

            for person_id in self.person_prototypes.keys():
                locked_by = self.person_prototypes[person_id].get('locked_by', 'None')
                summary_lines.append(f"  - {person_id} (锁定者: {locked_by})")

            summary_text = "\n".join(summary_lines)

            with open(filename, 'w') as f:
                f.write(summary_text)

            print(f"✅ 数据库摘要已导出: {filename}")
            print(summary_text)

        except Exception as e:
            print(f"❌ 导出数据库摘要失败: {e}")

    def cleanup_database(self, time_window=3600):
        """清理数据库中的过期数据 - 新结构"""
        try:
            current_time = time.time()
            cutoff_time = current_time - time_window

            # 清理过期的track缓存
            cleaned_tracks = 0
            for track_id in list(self.track_cache.keys()):
                last_frame = self.track_cache[track_id]['last_feat_frame']
                # 如果超过一定时间没有更新，清理track缓存
                if current_time - last_frame > time_window:
                    # 释放锁定的原型
                    person_id = self.track_cache[track_id].get('person_id')
                    if person_id and person_id in self.person_prototypes:
                        if self.person_prototypes[person_id].get('locked_by') == track_id:
                            self.person_prototypes[person_id]['locked_by'] = None
                            print(f"🔓 清理时解锁原型: {person_id} (track_id={track_id})")

                    del self.track_cache[track_id]
                    cleaned_tracks += 1
                    print(f"🧹 清理过期track缓存: {track_id}")

            # 清理过期的原型（可选，但通常由系统逻辑自动管理）
            print(f"✅ 数据库清理完成")
            print(f"   清理track缓存: {cleaned_tracks} 个")
            print(f"   剩余人员原型: {len(self.person_prototypes)} 个")
            print(f"   剩余track缓存: {len(self.track_cache)} 个")

            return True

        except Exception as e:
            print(f"❌ 清理数据库失败: {e}")
            return False

    def get_person_registry_stats(self):
        """获取人员注册表统计信息 - 新结构"""
        total_prototypes = len(self.person_prototypes)
        locked_prototypes = sum(1 for data in self.person_prototypes.values()
                                if data.get('locked_by') is not None)

        return {
            'total_persons': total_prototypes,
            'total_features': total_prototypes,  # 每个原型就是一个特征
            'avg_features_per_person': 1.0,  # 每个原型就是一个特征
            'person_creation_count': self.person_creation_count,
            'person_match_count': getattr(self, 'person_match_count', 0),  # 使用getattr避免未定义错误
            'locked_prototypes': locked_prototypes,
            'unlocked_prototypes': total_prototypes - locked_prototypes
        }

    def is_bbox_valid(self, bbox):
        """检查bbox是否有效（过滤过小的bbox）"""
        x, y, w, h = bbox
        # 最小尺寸要求：10x20像素
        min_area = 10 * 20
        bbox_area = w * h
        return bbox_area >= min_area

    def process_track_v3(self, track_id, person_bbox, face_bbox, frame_num, frame_data, active_track_ids=None,
                         camera_id=-1):
        """
        【最终版 V3，替换 process_track_final】
        引入 "确认-更新 分离" 和 "置信度累积" 机制，以对抗不稳定的ReID特征。
        """
        print(f"🔍 PROCESS_TRACK_V3: Track {track_id} 开始处理，frame_num={frame_num}")

        # 1. 初始化或获取track缓存
        if track_id not in self.track_cache:
            self.track_cache[track_id] = {
                'person_id': None,
                'status': 'unconfirmed',
                'confirming_id': None,
                'confirming_count': 0,
                'last_feat_frame': -1,
                'last_display_status': 'new',
                'last_reid_sim': 0.0,
                'last_face_sim': 0.0,
                'camera_id': -1  # 新增：记录track所属的摄像头ID
            }
        track_info = self.track_cache[track_id]
        # 设置track的摄像头ID
        track_info['camera_id'] = camera_id

        # 【新增】智能节流逻辑 - 根据目标状态决定提取频率
        status = track_info['status']
        last_frame = track_info.get('last_feat_frame', -1)
        frame_diff = frame_num - last_frame if last_frame != -1 else float('inf')

        # 根据状态决定提取间隔 - 新目标更频繁，已确认目标降低频率
        if status in ['unconfirmed', 'confirming']:
            interval = 5  # 新目标每5帧提取一次，尽快确认身份
        else:
            interval = 15  # 已确认目标每15帧更新一次，减少计算量

        # 智能节流逻辑
        if frame_diff < interval:
            throttled_reid_feature = track_info.get('last_reid_feature', None)
            return (track_info['person_id'], track_info['last_reid_sim'],
                    track_info['last_face_sim'], 'throttled', throttled_reid_feature)
        else:
            # --- 特征提取 ---
            track_info['last_feat_frame'] = frame_num

            reid_feature = self.extract_person_feature(frame_data, person_bbox)
            face_feature = None
            if face_bbox:
                x, y, w, h = map(int, face_bbox)
                face_crop = frame_data[y:y + h, x:x + w]
                face_feature = self.face_extractor.extract_face_feature(face_crop)
                if face_feature is not None:
                    self.face_inference_count += 1  # 增加人脸推理计数器
            if reid_feature is None and face_feature is None:
                return (track_info['person_id'], 0.0, 0.0, 'no_feature', None)

        # 【新增】优先检查短期记忆（遮挡消失的目标）
        person_id_from_memory, sim_from_memory = self.find_in_short_term_memory(reid_feature)
        if person_id_from_memory and sim_from_memory >= self.REID_REACQUISITION_THRESHOLD:
            matched_id = person_id_from_memory
            match_source = 'memory'  # 标记为来自短期记忆
            reid_sim = sim_from_memory
            face_id = None
            face_sim = 0.0
        else:
            # 如果短期记忆没有匹配，再进行常规的全局匹配
            if active_track_ids is None:
                active_track_ids = set()
            (face_id, face_sim), (reid_id, reid_sim) = self.find_best_matches(reid_feature, face_feature, track_id,
                                                                              active_track_ids)

            matched_id = None
            match_source = 'none'

            # 【修改决策逻辑】使用双层阈值
            # 决策1：人脸匹配成功，直接确认 (权威性最高)
            if face_id and face_sim >= FACE_SIMILARITY_THRESHOLD:
                matched_id = face_id
                match_source = 'face'

            # 决策2：如果track已经有关联ID，使用高阈值进行"持续确认"
            elif track_info['person_id'] and reid_id == track_info[
                'person_id'] and reid_sim >= self.REID_CONFIRMATION_THRESHOLD:
                matched_id = reid_id
                match_source = 'reid_confirm'

            # 决策3：如果track是新的，使用较低的"寻回阈值"在全局库中匹配
            elif not track_info['person_id'] and reid_id and reid_sim >= self.REID_REACQUISITION_THRESHOLD:
                matched_id = reid_id
                match_source = 'reid_reacquire'
            else:
                # 没有达到任何匹配阈值
                matched_id = None
                match_source = 'none'

        # --- 状态机流转 ---
        current_person_id = track_info['person_id']

        # 场景1：找到了一个匹配 (无论是脸还是ReID)
        if matched_id:

            # 【新增】人脸覆盖逻辑
            # 如果当前track已有关联的ID，但新的人脸匹配指向了另一个ID
            current_person_id = track_info.get('person_id')
            if match_source == 'face' and current_person_id and current_person_id != matched_id:
                # 简单处理：直接覆盖ID。未来可实现原型合并。
                track_info['person_id'] = matched_id
                track_info['confirming_id'] = matched_id
                track_info['confirming_count'] = self.CONFIRMATION_COUNT_THRESHOLD  # 人脸直接确认
                # 解锁旧ID的原型
                if current_person_id in self.person_prototypes:
                    self.person_prototypes[current_person_id]['locked_by'] = None

            # 如果匹配到的ID和我们正在确认的是同一个ID
            if matched_id == track_info['confirming_id']:
                track_info['confirming_count'] += 1
                print(f"🔍 PROCESS_TRACK_V3: Track {track_id} 确认计数增加: {track_info['confirming_count']}")
            # 如果匹配到了一个新ID，或者之前没有在确认的ID
            else:
                track_info['confirming_id'] = matched_id
                track_info['confirming_count'] = 1
                print(f"🔍 PROCESS_TRACK_V3: Track {track_id} 开始确认新ID: {matched_id}, 计数重置为1")

            # 检查是否达到确认阈值
            is_confirmed_by_count = track_info['confirming_count'] >= self.CONFIRMATION_COUNT_THRESHOLD
            is_confirmed_by_face = (match_source == 'face')
            print(
                f"🔍 PROCESS_TRACK_V3: Track {track_id} 确认检查: count_threshold={is_confirmed_by_count}, face_source={is_confirmed_by_face}")

            # 只要满足任一确认条件
            if is_confirmed_by_count or is_confirmed_by_face:
                # 身份正式确认！
                confirmed_person_id = track_info['confirming_id']
                # 防止ID在确认期间被其他逻辑改变
                if track_info.get('person_id') != confirmed_person_id:
                    print(f"✅ [ID-CONFIRMED] Track {track_id} -> {confirmed_person_id} by {match_source}")

                track_info['person_id'] = confirmed_person_id
                track_info['status'] = 'confirmed'

                # 【关键】只有在确认后，才更新原型以防止污染
                self.update_prototypes(confirmed_person_id, track_id, reid_feature, face_feature, camera_id)
                self.person_match_count += 1  # 增加匹配计数器

            # 还没达到确认阈值，只是在"确认中"
            else:
                track_info['status'] = 'confirming'
                # 暂时将person_id指向正在确认的id，用于显示，但不更新原型
                current_person_id = track_info['confirming_id']

        # 场景2：没有找到任何匹配
        else:
            # 重置确认过程
            track_info['confirming_id'] = None
            track_info['confirming_count'] = 0

            # 如果之前是已确认状态，现在跟丢了，可以暂时保持旧ID一段时间
            if track_info['status'] == 'confirmed':
                pass  # 保持 current_person_id
            # 如果之前就没确认，现在也没匹配上，那就是新人
            else:
                if current_person_id is None:  # 避免覆盖一个正在确认中的ID
                    self.person_creation_count += 1
                    current_person_id = f"person_{self.person_creation_count:04d}"
                    self.update_prototypes(current_person_id, track_id, reid_feature, face_feature,
                                           camera_id)  # 新人直接创建并更新
                    track_info['person_id'] = current_person_id
                    track_info['status'] = 'confirmed'  # 新人直接就是确认状态

        # --- 更新用于显示的状态 ---
        track_info['last_display_status'] = track_info['status']
        track_info['last_reid_sim'] = reid_sim
        track_info['last_face_sim'] = face_sim
        track_info['last_reid_feature'] = reid_feature  # 保存reid特征供节流时使用

        return current_person_id, reid_sim, face_sim, track_info['status'], reid_feature

    def periodic_log(self, frame_num, active_track_ids):
        """
        【性能优化】周期性日志输出
        - 减少高频print带来的I/O阻塞
        - 提供系统状态概览
        """
        self.log_counter += 1

        if self.log_counter >= self.log_interval:
            self.log_counter = 0

            print(
                f"📊 系统状态(帧#{frame_num}): "
                f"活动目标={len(active_track_ids) if active_track_ids else 0}, "
                f"已注册人员={len(self.person_prototypes)}, "
                f"ReID/Face推理={self.reid_inference_count}/{self.face_inference_count}, "
                f"创建/匹配={self.person_creation_count}/{self.person_match_count}, "
                f"短期记忆={len(self.recently_disappeared_tracks)}, "
                f"缓存Tracks={len(self.track_cache)}"
            )


# 全局变量跟踪REID状态
reid_status = {}  # track_id -> {'person_id': str, 'confirmed': bool, 'similarity': float}

# 全局FastReID实例
fastreid_system = FastReIDDatabase()


def migrate_database(self, old_filename, new_filename=None):
    """
    数据库迁移工具：将旧版本数据库升级到V5.0格式
    """
    if new_filename is None:
        new_filename = old_filename.replace('.json', '_v5.json')

    print(f"🔄 开始数据库迁移: {old_filename} -> {new_filename}")

    # 加载旧数据库
    if not self.load_database(old_filename):
        print("❌ 无法加载源数据库")
        return False

    # 保存为新格式
    if not self.save_database(new_filename):
        print("❌ 无法保存新格式数据库")
        return False

    print(f"✅ 数据库迁移完成！")
    print(f"   新数据库: {new_filename}")


# 数据库初始化函数
def initialize_database(args):
    """
    【完整最终版】初始化数据库功能，正确处理清理、导出等命令行操作。
    """

    # --- 步骤 1: 检查是否是特殊操作命令（清理或导出） ---
    # 这些命令需要先加载数据库，执行操作，然后退出程序。

    if args.export_summary or args.cleanup_db:

        print("──────────────────────────────────────────────────")
        print("⚙️  执行数据库维护命令...")
        print("──────────────────────────────────────────────────")

        # 首先，检查数据库文件是否存在，如果不存在则无法操作。
        if not os.path.exists(args.db_file):
            print(f"❌ 错误: 数据库文件 '{args.db_file}' 不存在，无法执行操作。")
            return False  # 返回False，让主程序退出

        # 加载数据库到内存
        print(f"📂 正在从 '{args.db_file}' 加载数据库...")
        if not fastreid_system.load_database(args.db_file):
            print(f"❌ 错误: 加载数据库 '{args.db_file}' 失败，无法继续。")
            return False  # 返回False，让主程序退出

        print("✅ 数据库加载成功。")
        print("──────────────────────────────────────────────────")

        # --- 根据命令执行具体操作 ---

        if args.export_summary:
            print("✍️  正在导出数据库摘要...")
            fastreid_system.export_database_summary("database_summary.txt")
            print("✅ 摘要已成功导出到 'database_summary.txt'。")

        if args.cleanup_db:
            print("🧹  正在清理数据库中的过期数据...")
            # 执行清理操作，可以指定时间窗口（例如，只保留最近1小时的数据）
            fastreid_system.cleanup_database(time_window=3600)

            print("\n💾  正在保存清理后的数据库...")
            # 清理后自动保存，以持久化更改
            if fastreid_system.save_database(args.db_file):
                print("✅ 清理后的数据库已成功保存。")
            else:
                print("❌ 错误: 保存清理后的数据库失败！")

        print("──────────────────────────────────────────────────")
        print("✅ 数据库维护操作完成，程序将退出。")

        # 返回False，告诉主程序不需要启动GStreamer管线，直接退出即可
        return False

    # --- 步骤 2: 正常的启动流程 ---
    # 如果没有特殊命令，则执行常规的程序启动加载。

    print("──────────────────────────────────────────────────")
    print("🚀  系统正常启动...")
    print("──────────────────────────────────────────────────")

    # 尝试加载现有数据库（如果存在或被命令行指定）
    if args.load_db or os.path.exists(args.db_file):
        print(f"📂 正在尝试从 '{args.db_file}' 加载数据库...")
        if fastreid_system.load_database(args.db_file):
            print("✅ 数据库加载成功，准备运行管线。")
        else:
            # 文件不存在或加载失败都不是致命错误，程序可以继续并创建新数据库
            print(f"⚠️  警告: 数据库 '{args.db_file}' 不存在或加载失败，将在运行时创建新数据库。")
    else:
        print("ℹ️  信息: 未找到现有数据库，将在运行时创建新数据库。")

    # 返回True，告诉主程序继续执行，启动GStreamer管线
    return True


def cleanup_database(args):
    """清理数据库功能"""
    if not args.no_save:
        print("💾 保存REID数据库...")
        if fastreid_system.save_database(args.db_file):
            print("✅ REID数据库保存完成")
        else:
            print("❌ REID数据库保存失败")

    # 打印最终统计信息
    print("\n📊 数据库统计信息:")
    print(f"   人员总数: {len(fastreid_system.person_prototypes)}")
    print(f"   Track缓存: {len(fastreid_system.track_cache)}")
    print(f"   创建次数: {fastreid_system.person_creation_count}")
    print(f"   匹配次数: {getattr(fastreid_system, 'person_match_count', 0)}")  # 使用getattr避免未定义错误
    print(f"   处理帧数: {fastreid_system.total_frames_processed}")
    print(f"   推理次数: {fastreid_system.reid_inference_count}")
    print(f"   数据库文件: {args.db_file}")


def normalize_bbox_to_target(bbox, frame_width, frame_height, target_width, target_height):
    """将边界框坐标标准化到目标分辨率 - 关键修复"""
    # 正确理解：模型推理在 960x544 上进行，得到 960x544 坐标
    # 显示在 1280x720 上，需要将坐标从 960x544 映射到 1280x720
    # 但是映射应该是按比例缩放，不是放大！

    if frame_width == target_width and frame_height == target_height:
        # 分辨率相同，不需要转换
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    # 正确的缩放：从推理分辨率到显示分辨率
    # 模型推理在 960x544，显示在 1280x720
    # 所以需要将坐标从 960x544 映射到 1280x720
    scale_x = target_width / frame_width
    scale_y = target_height / frame_height

    left = int(bbox[0] * scale_x)
    top = int(bbox[1] * scale_y)
    width = int(bbox[2] * scale_x)
    height = int(bbox[3] * scale_y)

    # 边界检查
    left = max(0, min(left, target_width - width))
    top = max(0, min(top, target_height - height))
    width = max(1, min(width, target_width - left))
    height = max(1, min(height, target_height - top))

    return [left, top, width, height]


def fix_bbox_coordinates(raw_bbox, display_width, display_height):
    """修复边界框坐标 - 关键修复"""
    # 关键修复：根据模型推理分辨率(960x544)和显示分辨率(1280x720)的比例进行缩放
    # 模型在960x544上推理，但显示在1280x720上

    # 计算缩放比例
    scale_x = display_width / 960  # 1280/960 = 1.333
    scale_y = display_height / 544  # 720/544 = 1.324

    # 从模型推理坐标缩放到显示坐标
    left = int(raw_bbox[0] * scale_x)
    top = int(raw_bbox[1] * scale_y)
    width = int(raw_bbox[2] * scale_x)
    height = int(raw_bbox[3] * scale_y)

    # 安全限制：确保边界框不会过大
    max_display_width = display_width * 0.5  # 最大50%显示宽度
    max_display_height = display_height * 0.8  # 最大80%显示高度

    if width > max_display_width:
        width = int(max_display_width)
    if height > max_display_height:
        height = int(max_display_height)

    # 边界检查
    left = max(0, min(left, display_width - width))
    top = max(0, min(top, display_height - height))
    width = max(1, min(width, display_width - left))
    height = max(1, min(height, display_height - top))

    return [left, top, width, height]


def fix_bbox_coordinates_multi(raw_bbox, display_width, display_height, source_index, num_sources):
    """多路版本边界框坐标修复 - 考虑tiler布局"""
    # 首先按单路方式计算坐标
    bbox = fix_bbox_coordinates(raw_bbox, display_width, display_height)

    # 计算tiler布局偏移
    # 假设tiler是1行N列布局
    tile_width = TILER_OUTPUT_WIDTH // num_sources  # 每个tile的宽度
    tile_height = TILER_OUTPUT_HEIGHT  # 每个tile的高度

    # 计算在tiler中的偏移
    offset_x = source_index * tile_width
    offset_y = 0  # 单行布局，Y偏移为0

    # 应用tiler偏移
    left = bbox[0] + offset_x
    top = bbox[1] + offset_y
    width = bbox[2]
    height = bbox[3]

    return [left, top, width, height]


def is_bbox_inside(inner_bbox, outer_bbox, tolerance=0.95):
    """检查内框是否完全在外框内（带容忍度）"""
    ix, iy, iw, ih = inner_bbox
    ox, oy, ow, oh = outer_bbox

    # 检查内框的四个角是否都在外框内
    inner_left = ix
    inner_right = ix + iw
    inner_top = iy
    inner_bottom = iy + ih

    outer_left = ox
    outer_right = ox + ow
    outer_top = oy
    outer_bottom = oy + oh

    # 计算内框在外框内的面积比例
    x_overlap_left = max(inner_left, outer_left)
    x_overlap_right = min(inner_right, outer_right)
    y_overlap_top = max(inner_top, outer_top)
    y_overlap_bottom = min(inner_bottom, outer_bottom)

    if x_overlap_right < x_overlap_left or y_overlap_bottom < y_overlap_top:
        return False

    overlap_area = (x_overlap_right - x_overlap_left) * (y_overlap_bottom - y_overlap_top)
    inner_area = iw * ih

    # 重叠面积必须占内框面积的一定比例
    return (overlap_area / inner_area) >= tolerance if inner_area > 0 else False


def get_true_frame_resolution(pipeline):
    """获取真正的帧分辨率"""
    # DeepStream 7.1 中，真正的帧分辨率通常由 streammux 决定
    # 或者从解码器输出决定
    # 返回默认的推理分辨率
    return MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT


def tiler_src_pad_buffer_probe(pad, info, u_data):
    """
    【DeepStream 7.1 性能优化版】
    - 去除高频print语句，减少I/O阻塞
    - 移除不必要的sorted()操作
    - 保留核心的"帧内ID锁"逻辑
    """
    uris = u_data if u_data else []

    try:
        gst_buffer = info.get_buffer()
        if not gst_buffer: return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta: return Gst.PadProbeReturn.OK

    except Exception as e:
        print(f"❌ [PROBE-CRITICAL] 获取buffer或batch metadata失败: {e}")
        return Gst.PadProbeReturn.OK

    if not hasattr(tiler_src_pad_buffer_probe, 'frame_counter'):
        tiler_src_pad_buffer_probe.frame_counter = 0
    else:
        tiler_src_pad_buffer_probe.frame_counter += 1
    current_frame_num = tiler_src_pad_buffer_probe.frame_counter

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            fastreid_system.total_frames_processed += 1

            frame_data = None
            try:
                surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                if surface is not None:
                    arr = np.array(surface, copy=True, order='C')
                    if len(arr.shape) == 3 and arr.shape[2] == 4:
                        frame_data = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    elif len(arr.shape) == 3 and arr.shape[2] == 3:
                        frame_data = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    else:
                        l_frame = l_frame.next
                        continue
            except RuntimeError:
                l_frame = l_frame.next
                continue

            if frame_data is None:
                l_frame = l_frame.next
                continue

            persons_in_frame, faces_in_frame = {}, []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    raw_bbox = [obj_meta.rect_params.left, obj_meta.rect_params.top, obj_meta.rect_params.width,
                                obj_meta.rect_params.height]
                except Exception:
                    l_obj = l_obj.next
                    continue

                src_idx = frame_meta.pad_index
                batch_size = len(uris)
                bbox = fix_bbox_coordinates_multi(raw_bbox, MUXER_OUTPUT_WIDTH, MUXER_OUTPUT_HEIGHT, src_idx,
                                                  batch_size)

                if obj_meta.class_id == 0:
                    persons_in_frame[obj_meta.object_id] = (bbox, obj_meta)
                elif obj_meta.class_id == 2:
                    faces_in_frame.append(bbox)
                l_obj = l_obj.next

            person_to_face_map = {}
            for track_id, (person_bbox, _) in persons_in_frame.items():
                for face_bbox in faces_in_frame:
                    if is_bbox_inside(face_bbox, person_bbox):
                        person_to_face_map[track_id] = face_bbox
                        break

            active_track_ids = set(persons_in_frame.keys())

            # 【性能优化】用于防止同一帧内一个person_id被多个track占用的"帧内锁"
            claimed_person_ids_in_frame = set()

            # 【性能优化】移除 sorted()，直接迭代字典
            for track_id, (person_bbox, obj_meta) in persons_in_frame.items():
                face_bbox = person_to_face_map.get(track_id)

                src_idx = frame_meta.pad_index
                local_id = obj_meta.object_id
                global_track_id = (src_idx << 32) | local_id

                person_id, reid_sim, face_sim, status, reid_feature = fastreid_system.process_track_v3(
                    global_track_id, person_bbox, face_bbox, current_frame_num,
                    frame_data, active_track_ids, src_idx
                )

                # --- 【核心修复逻辑 - 保留】 ---
                if person_id and person_id in claimed_person_ids_in_frame:
                    # 【性能优化】只在发生关键冲突时打印，而不是每一帧都打印
                    print(
                        f"🚨 [ID-CONFLICT] Frame {current_frame_num}: Track {global_track_id} 试图认领已被占用的 ID {person_id}。强制重置！")
                    if global_track_id in fastreid_system.track_cache:
                        track_info = fastreid_system.track_cache[global_track_id]
                        track_info.update({'status': 'unconfirmed', 'confirming_id': None, 'confirming_count': 0})
                    person_id = None  # 清除person_id，后续逻辑会将其作为未识别处理
                elif person_id:
                    claimed_person_ids_in_frame.add(person_id)

                # --- ReID特征注入逻辑 (不变) ---
                if reid_feature is not None:
                    try:
                        user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
                        if user_meta:
                            custom_struct = pyds.alloc_custom_struct(user_meta)

                            custom_struct.structId = reid_feature.shape[0]
                            feature_str = ",".join(map(str, reid_feature.astype(np.float32)))
                            custom_struct.message = feature_str
                            custom_struct.message = pyds.get_string(custom_struct.message)
                            custom_struct.sampleInt = int(reid_feature.shape[0])

                            user_meta.user_meta_data = custom_struct
                            user_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_USER_META

                            pyds.nvds_add_user_meta_to_obj(obj_meta, user_meta)
                    except Exception as e:
                        pass  # 静默处理特征注入错误

                # --- 【V3 显示逻辑 - 性能优化版】---
                display_text = ""
                border_color = (0.0, 1.0, 0.0, 1.0)  # Green (默认未识别)

                if person_id:
                    track_info = fastreid_system.track_cache.get(global_track_id, {})
                    current_status = track_info.get('status', 'unconfirmed')

                    sim_text = ""
                    if face_sim >= FACE_SIMILARITY_THRESHOLD:
                        sim_text = f"F:{face_sim:.2f}"
                    elif reid_sim > 0:
                        sim_text = f"R:{reid_sim:.2f}"

                    if current_status == 'confirmed':
                        border_color = (1.0, 0.0, 0.0, 1.0)  # Red
                        if track_info.get('last_face_sim', 0) >= FACE_SIMILARITY_THRESHOLD:
                            border_color = (1.0, 0.64, 0.0, 1.0)  # Orange
                    elif current_status == 'confirming':
                        border_color = (1.0, 1.0, 0.0, 1.0)  # Yellow
                    else:  # unconfirmed
                        border_color = (0.0, 0.0, 1.0, 1.0)  # Blue

                    status_short = {'unconfirmed': 'U', 'confirming': 'C', 'confirmed': 'OK'}.get(current_status, 'U')
                    display_text = f"{person_id} {sim_text}[{status_short}]"
                else:
                    display_text = f"T:{global_track_id}"

                obj_meta.text_params.display_text = display_text
                obj_meta.text_params.font_params.font_name = "Serif"
                obj_meta.text_params.font_params.font_size = 12
                obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                obj_meta.rect_params.border_color.set(*border_color)
                obj_meta.rect_params.border_width = 3

            fastreid_system.cleanup_stale_tracks(active_track_ids)

            # 【性能优化】使用新的周期性日志系统
            fastreid_system.periodic_log(current_frame_num, active_track_ids)

        except Exception as e:
            import traceback
            print(f"❌ Probe Error in Frame Loop: {e}")
            traceback.print_exc()

        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


def check_pipeline_elements(pipeline):
    """检查管线元素是否正确创建和添加"""
    print("🔍 检查管线元素...")

    try:
        # 方法1: 使用ChildProxy接口
        elements = pipeline.get_by_interface(Gst.ChildProxy)
        if elements:
            print("✅ 使用ChildProxy接口获取元素:")
            try:
                # 尝试不同的方法获取子元素数量
                if hasattr(elements, 'get_children'):
                    children = elements.get_children()
                    num_elements = len(children)
                    print(f"   子元素数量: {num_elements}")
                    for i, child in enumerate(children):
                        print(f"   {i}: {child.get_name()} ({child.get_factory().get_name()}")
                elif hasattr(elements, 'get_property') and elements.get_property('num-children'):
                    num_elements = elements.get_property('num-children')
                    print(f"   子元素数量: {num_elements}")
                    for i in range(num_elements):
                        child = elements.get_property_nth(i)
                        if child:
                            print(f"   {i}: {child.get_name()} ({child.get_factory().get_name()}")
                else:
                    print("   无法获取子元素数量，使用备用方法")
                    # 备用方法：遍历pipeline中的所有元素
                    _dump_pipeline_elements(pipeline)
            except Exception as e:
                print(f"⚠️ ChildProxy方法失败: {e}")
                # 备用方法：遍历pipeline中的所有元素
                _dump_pipeline_elements(pipeline)
        else:
            print("❌ 无法获取ChildProxy接口，使用备用方法")
            # 备用方法：遍历pipeline中的所有元素
            _dump_pipeline_elements(pipeline)

    except Exception as e:
        print(f"⚠️ 检查管线元素时出错: {e}")
        # 备用方法：遍历pipeline中的所有元素
        _dump_pipeline_elements(pipeline)

    # 检查管线状态
    try:
        state = pipeline.get_state(Gst.State.NULL)
        print(f"📊 管线状态: {state}")
    except Exception as e:
        print(f"⚠️ 获取管线状态失败: {e}")


def _dump_pipeline_elements(pipeline):
    """备用方法：遍历管线中的所有元素"""
    print("📦 使用备用方法遍历管线元素:")

    # 获取pipeline中的所有元素
    try:
        elements = []

        def traverse_elements(element, depth=0):
            elements.append((element, depth))
            # 尝试获取子元素
            try:
                if hasattr(element, 'get_children'):
                    children = element.get_children()
                    for child in children:
                        traverse_elements(child, depth + 1)
            except:
                pass

        traverse_elements(pipeline)

        print(f"   总计元素数量: {len(elements)}")
        for i, (element, depth) in enumerate(elements):
            indent = "  " * depth
            name = element.get_name() or "Unknown"
            factory = element.get_factory().get_name() if element.get_factory() else "Unknown"
            print(f"   {i}: {indent}{name} ({factory})")

    except Exception as e:
        print(f"❌ 备用方法也失败: {e}")
        # 最后的备用方法
        print("   🆘 最后备用方法：列出已知元素")
        known_elements = ["source", "h264parser", "decoder", "streammux", "pgie", "tracker",
                          "nvvidconv1", "nvvidconv2", "nvosd", "sink"]
        for i, elem_name in enumerate(known_elements):
            print(f"   {i}: {elem_name} (预期)")


def create_source_bin(index, uri):
    """创建单路源的source bin - 使用DeepStream 7.1官方API"""
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # 使用uridecodebin自动适配格式
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-{index}")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")

    uri_decode_bin.set_property("uri", uri)

    # 为RTSP源添加优化配置
    if uri.startswith("rtsp://"):
        print(f"🔧 为RTSP源{index}应用优化配置")
        uri_decode_bin.set_property("buffer-duration", 2000000)  # 2秒缓冲
        uri_decode_bin.set_property("buffer-size", 0)  # 自动缓冲大小
        uri_decode_bin.set_property("download", False)  # 不下载整个文件
        uri_decode_bin.set_property("use-buffering", True)  # 启用缓冲

    # 连接回调函数
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, None)

    nbin.add(uri_decode_bin)

    # 创建ghost pad，稍后会设置target
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None

    return nbin


def cb_newpad(decodebin, decoder_src_pad, data):
    """处理decodebin的pad添加回调 - 使用官方API"""
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    print("gstname=", gstname)
    if gstname.find("video") != -1:
        print("features=", features)
        if features.contains("memory:NVMM"):
            # 获取source bin的ghost pad并设置target
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("📺 视频播放结束")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"❌ 错误: {err.message}")
        print(f"🔍 详细信息: {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"⚠️ 警告: {err.message}")
    elif t == Gst.MessageType.INFO:
        if message.src:
            src_name = message.src.get_name()
            print(f"ℹ️ 信息: {src_name} - {message.parse_info()}")
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src:
            old, new, pending = message.parse_state_changed()
            src_name = message.src.get_name()
            print(f"🔄 状态变化: {src_name} {old} -> {new} (pending: {pending})")
    elif t == Gst.MessageType.STREAM_STATUS:
        if message.src:
            try:
                type_, owner = message.parse_stream_status()
                print(f"📊 流状态: {owner.get_name()} - {type_}")
            except Exception as e:
                print(f"⚠️ 解析流状态失败: {e}")
    elif t == Gst.MessageType.ELEMENT:
        # 元素特定消息
        structure = message.get_structure()
        if structure and structure.has_field("message"):
            msg_text = structure.get_value("message")
            print(f"📦 元素消息: {msg_text}")
    return True


# RTSP动态pad处理函数
def cb_new_pad(rtspsrc, new_pad, queue):
    """处理rtspsrc的动态pad生成"""
    # 已经连过就返回
    if new_pad.is_linked():
        print("⚠️ Pad已经连接过，跳过")
        return

    # 处理 H264 和 H265 的 rtp 流
    caps = new_pad.get_current_caps()
    if caps:
        s = caps.to_string()
        print(f"🔍 检查caps: {s}")
        if not ("application/x-rtp" in s and "media=(string)video" in s):
            print("⚠️ 非视频RTP流，跳过")
            return

        # 根据编码类型选择不同的depayloader
        if "H265" in s or "h265" in s:
            print("📹 检测到H265视频流")
            # 使用H265处理链路
            try:
                # 链接 queue -> h265_depay -> h265_parse -> decoder
                queue.link(rtsp_h265_depay)
                rtsp_h265_depay.link(rtsp_h265_parse)
                rtsp_h265_parse.link(rtsp_decoder)
                print("✅ H265处理链路链接成功")
            except Exception as e:
                print(f"❌ H265处理链路链接失败: {e}")
                return
        elif "H264" in s or "h264" in s:
            print("📹 检测到H264视频流")
            # 使用H264处理链路
            try:
                # 链接 queue -> h264_depay -> h264_parse -> decoder
                queue.link(rtsp_h264_depay)
                rtsp_h264_depay.link(rtsp_h264_parse)
                rtsp_h264_parse.link(rtsp_decoder)
                print("✅ H264处理链路链接成功")
            except Exception as e:
                print(f"❌ H264处理链路链接失败: {e}")
                return
        else:
            print("⚠️ 未知的视频编码格式，跳过")
            return

    sink_pad = queue.get_static_pad("sink")
    if sink_pad.is_linked():
        print("⚠️ Queue sink pad已连接，跳过")
        return

    ret = new_pad.link(sink_pad)
    if ret == Gst.PadLinkReturn.OK:
        print("✅ rtspsrc pad -> queue.sink 连接成功")
        print(f"🔗 连接详情: {new_pad.get_name()} -> {sink_pad.get_parent().get_name()}")
    else:
        print(f"❌ rtspsrc pad 连接失败: {ret}")


def decodebin_child_added(child_proxy, object, name, user_data):
    """处理decodebin的子元素添加"""
    print(f"🔧 decodebin新增子元素: {name}")
    if name == "source":
        # 对RTSP源进行配置
        try:
            object.set_property('latency', 200)
            object.set_property('drop-on-latency', True)
            print("✅ RTSP源属性设置成功")
        except Exception as e:
            print(f"⚠️ 设置RTSP源属性失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FastReID Market1501集成系统 - 多路版本')
    parser.add_argument("--sources", nargs='+', required=True,
                        help="最多 6 条 uri（rtsp:// 或 文件路径，如 rtsp://用户:密码@IP 视频文件.mp4）")
    parser.add_argument("--test-video", default="sample_720p.h264",
                        help="测试视频文件路径")

    # --- 【修改部分】 ---
    # 数据库相关参数
    parser.add_argument("--load-db", action="store_true",
                        help="启动时加载现有数据库")
    # 删除了 --save-db 参数，因为保存是默认行为
    parser.add_argument("--no-save", action="store_true",
                        help="程序退出时不保存数据库（默认会保存）")
    parser.add_argument("--export-summary", action="store_true",
                        help="导出数据库摘要并退出")
    parser.add_argument("--cleanup-db", action="store_true",
                        help="清理过期数据并退出")
    parser.add_argument("--db-file", default="fastreid_database.json",
                        help="指定数据库文件路径")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 检查输入源数量
    uris = args.sources
    batch_size = len(uris)
    if batch_size > 6:
        print("⚠️  目前最多支持 6 路视频源")
        sys.exit(1)

    # 转换为正确的URI格式
    for i, uri in enumerate(uris):
        if not uri.startswith("rtsp://") and not uri.startswith("file://"):
            # 本地文件需要file://协议
            abs_path = os.path.abspath(uri)
            uris[i] = f"file://{abs_path}"

    print(f"🎯 多路混合识别系统启动 V4.2 [FastReID + Buffalo_L人脸识别] - 多路版")
    print(f"📊 支持 {batch_size} 路视频源:")
    for i, uri in enumerate(uris):
        source_type = "RTSP" if uri.startswith("rtsp://") else "FILE"
        print(f"   源{i}: {uri} ({source_type})")

    # 验证所有文件源
    for i, uri in enumerate(uris):
        if not uri.startswith("rtsp://"):
            # 从file:// URI中提取文件路径
            file_path = uri[7:] if uri.startswith("file://") else uri
            if not os.path.isfile(file_path):
                print(f"❌ 视频文件不存在: {file_path}")
                if i == 0 and os.path.isfile(args.test_video):
                    print(f"🔄 尝试使用测试视频: {args.test_video}")
                    uris[0] = f"file://{os.path.abspath(args.test_video)}"
                    print(f"✅ 使用测试视频: {uris[0]}")
                else:
                    print(f"❌ 测试视频也不存在: {args.test_video}")
                    sys.exit(1)

    # 尝试检测视频分辨率
    def get_video_resolution(video_path):
        """尝试检测视频分辨率"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    cap.release()
                    return width, height
                cap.release()
        except:
            pass
        return MUXER_OUTPUT_WIDTH, MUXER_OUTPUT_HEIGHT  # 使用预设分辨率

    # 检测实际视频分辨率（如果是RTSP，使用默认分辨率）
    has_rtsp = any(uri.startswith("rtsp://") for uri in uris)
    if has_rtsp:
        input_frame_width, input_frame_height = MUXER_OUTPUT_WIDTH, MUXER_OUTPUT_HEIGHT
        print(f"📹 RTSP流使用默认分辨率: {input_frame_width}x{input_frame_height}")
    else:
        input_frame_width, input_frame_height = get_video_resolution(uris[0])
        print(f"📹 检测到视频分辨率: {input_frame_width}x{input_frame_height}")

    print("🎯 混合识别系统启动 V4.2 [FastReID + Buffalo_L人脸识别] - 修复版")
    print("📊 功能：多路人员检测 + 人脸检测 + OSNet ReID + Buffalo_L人脸特征 + 混合匹配")
    print("🚀 V4.2新增：支持多路视频源同时处理，共享同一套ReID数据库")
    print("🚀 V4.2新增：全局track_id确保跨摄像头唯一性")
    print("🚀 V3新增：确认-更新分离机制，对抗不稳定的ReID特征")
    print("🚀 V3新增：置信度累积系统，需要连续3次匹配才确认身份")
    print("🚀 V3新增：人脸一票确认权，强化人脸识别的权威性")
    print("🔧 修复：解决ID漂移和遮挡重识别失败问题")
    print("🔧 修复：帧数据获取问题已修复，使用nvosd sink pad + CUDA统一内存")
    print("🔧 修复：边界框过大问题已修复，参考deepstream-test2的统一分辨率系统")
    print("🔧 修复：采用模型推理分辨率(960x544)和显示分辨率(1280x720)的分离系统")
    print(
        f"📐 分辨率系统: 模型{MODEL_INPUT_WIDTH}x{MODEL_INPUT_HEIGHT} -> 显示{MUXER_OUTPUT_WIDTH}x{MUXER_OUTPUT_HEIGHT}")
    print("⚙️  配置：")
    print(f"   ReID特征维度: {REID_FEATURE_DIM} (官方OSNet-IBN)")
    print(f"   人脸特征维度: {FACE_FEATURE_DIM} (Buffalo_L)")
    print(f"   人脸权重: {FACE_WEIGHT}, ReID权重: {REID_WEIGHT}")
    print(f"   混合阈值: {HYBRID_SCORE_THRESHOLD}, 人脸阈值: {FACE_SIMILARITY_THRESHOLD}")
    print(f"   时间窗口: {PATROL_TIME_WINDOW}秒")
    print(f"   ReID模型: osnet_ibn_x1_0_market1501 (官方预训练)")
    print(f"   人脸模型: Buffalo_L (w600k_r50.onnx)")
    print(f"   视频源数量: {batch_size}")
    print(f"   DeepStream版本: 7.1")
    print(f"   Ubuntu版本: 22.04")
    print("🔍 V4.2版本颜色标识：")
    print("   🔵 蓝色 (unconfirmed): 新目标，未确认")
    print("   🟡 黄色 (confirming): 正在确认中，连续匹配1-2次")
    print("   🔴 红色 (confirmed by ReID): ReID连续确认3次以上")
    print("   🟠 橙色 (confirmed by Face): 人脸一票确认")
    print("   🟢 绿色: 完全未识别的跟踪目标")
    print("🎯 Probe位置: nvosd sink pad + CUDA统一内存，确保CPU可访问")
    print("-" * 50)

    # 初始化GStreamer
    GObject.threads_init()  # <--- NEW: 确保GLib/GObject线程安全
    Gst.init(None)

    # 检查文件
    if not check_model_files():
        print("❌ 模型文件检查失败，程序退出")
        sys.exit(1)

    # 初始化数据库
    if not initialize_database(args):
        print("📊 数据库操作完成，程序退出")
        sys.exit(0)

    # 创建管线
    pipeline = Gst.Pipeline.new("fastreid-market1501")

    # 创建多路source bin
    source_bins = []
    for i, uri in enumerate(uris):
        source_bin = create_source_bin(i, uri)
        pipeline.add(source_bin)
        source_bins.append(source_bin)
        print(f"✅ 创建源{i}: {uri}")

    print("✅ 多路source bin创建成功")

    # 创建元素
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")

    # 添加tiler元素解决多路视频拼接问题
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    if not tiler:
        print("❌ tiler元素创建失败！")
        sys.exit(1)
    print("✅ tiler元素创建成功")

    # 使用DeepStream 7.1推荐结构 - 确保正确的格式转换
    nvvidconv_pretiler = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-pretiler")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv1")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv2")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # 配置nvdsosd以显示自定义标签
    # 注意：DeepStream 7.1中nvdsosd的属性名可能不同
    try:
        nvosd.set_property("display-text", 1)  # 确保显示文本
    except Exception as e:
        print(f"⚠️ 设置display-text失败: {e}")

    try:
        # 尝试常见的文本边距属性名
        possible_padding_props = [
            "text-padding", "bbox-text-padding", "padding",
            "text-margin", "bbox-margin", "margin"
        ]
        for prop in possible_padding_props:
            try:
                nvosd.set_property(prop, 5)
                print(f"✅ 成功设置 {prop} = 5")
                break
            except:
                continue
    except Exception as e:
        print(f"⚠️ 设置文本边距失败: {e}")

    # 显示元素 - 使用标准显示
    if Gst.ElementFactory.find("nveglglessink"):
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        sink = Gst.ElementFactory.make("fakesink", "fake-renderer")

    # 修复管线连接：添加必要的转换元素
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-postosd")
    # nvegltransform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform") # 系统不支持

    # 创建elements列表
    elements = [streammux, pgie, tracker, nvvidconv_pretiler, tiler, nvvidconv1, nvvidconv2, nvosd, nvvidconv_postosd,
                sink]
    check_elements = [streammux, pgie, tracker, nvvidconv_pretiler, tiler, nvvidconv1, nvvidconv2, nvosd,
                      nvvidconv_postosd, sink]
    element_names = ["streammux", "pgie", "tracker", "nvvidconv_pretiler", "tiler", "nvvidconv1", "nvvidconv2", "nvosd",
                     "nvvidconv_postosd", "sink"]

    if not all(elements):
        print("❌ 创建GStreamer元素失败")
        for elem, name in zip(check_elements, element_names):
            if elem is None:
                print(f"   ❌ 元素创建失败: {name}")
        if 'capsfilter' in locals() and capsfilter is None:
            print("   ⚠️  capsfilter创建失败，将不使用capsfilter")
        return

    # 设置元素属性 - 参考 deepstream-test2：使用统一分辨率
    # 注意：RTSP源的属性已经在创建时设置

    # 设置流复用器参数 - 支持多路
    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', batch_size)  # 设置为实际源数量
    streammux.set_property('batched-push-timeout', 2000000)  # 增加超时时间避免卡死
    streammux.set_property('live-source', 1)  # 多路混合，设置为1支持RTSP
    streammux.set_property('attach-sys-ts', True)  # 附加系统时间戳
    streammux.set_property('sync-inputs', True)  # 同步输入
    streammux.set_property('max-latency', 2000000)  # 最大延迟2秒

    # 配置文件 - 使用支持人脸检测的配置文件
    config_file = 'dstest3_tracking_enabled_config.txt'

    # 确保使用正确的配置文件
    print(f"🔧 使用配置文件: {config_file}")
    print(
        "🚨 重要提示：请确保Peoplenet配置文件中 `[class-attrs-2]` (face) 的 `pre-cluster-threshold` 已设为较低值 (如0.4) 以启用人脸检测!")

    # 验证配置文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        sys.exit(1)

    # 显示配置文件内容的关键部分
    try:
        with open(config_file, 'r') as f:
            lines = f.readlines()
            print("📋 配置文件关键参数:")
            for i, line in enumerate(lines):
                if 'pre-cluster-threshold' in line or 'detected-max-w' in line or 'detected-max-h' in line:
                    print(f"   {line.strip()}")
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")

    pgie.set_property('config-file-path', config_file)
    pgie.set_property('batch-size', batch_size)  # 设置为实际源数量

    # 设置输入源和目标分辨率
    input_width = 1280  # 典型的720p视频宽度
    input_height = 720  # 典型的720p视频高度
    target_width = 960  # 模型推理分辨率
    target_height = 544  # 模型推理分辨率

    print(f"📐 分辨率设置: 输入={input_width}x{input_height} -> 目标={target_width}x{target_height}")

    tracker_config_file = 'config_tracker_NvDCF_person.yml'
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', tracker_config_file)
    tracker.set_property('display-tracking-id', 1)  # 启用tracker的自动显示作为备用

    # 设置tiler属性 - 解决多路视频拼接问题，保持16:9比例
    print(f"🔧 配置tiler: {batch_size}路视频, 总尺寸{TILER_OUTPUT_WIDTH}x{TILER_OUTPUT_HEIGHT}")
    tiler.set_property("rows", 1)  # 1行
    tiler.set_property("columns", batch_size)  # 根据源数量设置列数
    tiler.set_property("width", TILER_OUTPUT_WIDTH)  # tiler总输出宽度 (2个1280x720视频)
    tiler.set_property("height", TILER_OUTPUT_HEIGHT)  # tiler总输出高度 (保持16:9)
    print(f"✅ tiler配置完成: 每个视频{(TILER_OUTPUT_WIDTH // batch_size)}x{TILER_OUTPUT_HEIGHT} (16:9)")

    # 配置tiler前的格式转换，确保输出RGBA格式
    nvvidconv_pretiler.set_property('nvbuf-memory-type', 0)  # NVBUF_MEM_DEFAULT

    # 配置nvvidconv2以确保CPU可访问的格式 - 使用CUDA统一内存
    nvvidconv2.set_property('nvbuf-memory-type', 3)  # NVBUF_MEM_CUDA_UNIFIED
    nvvidconv2.set_property('output-buffers', 4)

    # 确保输出格式为RGBA，便于CPU访问 - 使用capsfilter
    capsfilter = Gst.ElementFactory.make("capsfilter", "caps-filter")
    if capsfilter:
        caps = Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), format=RGBA, width={TILER_OUTPUT_WIDTH}, height={TILER_OUTPUT_HEIGHT}")
        capsfilter.set_property("caps", caps)
        elements.append(capsfilter)
    else:
        capsfilter = None

    # 构建管线 - 修复连接问题，确保所有元素都添加到pipeline
    # 基础元素已经添加，现在添加其他元素
    remaining_elements = [streammux, pgie, tracker, nvvidconv_pretiler, tiler, nvvidconv1, nvvidconv2, nvosd,
                          nvvidconv_postosd, sink]
    if capsfilter:
        remaining_elements.append(capsfilter)

    for element in remaining_elements:
        pipeline.add(element)

    # 多路链接到流复用器
    try:
        print("🔗 链接多路源到流复用器...")

        # 逐路创建并挂到 pipeline
        for i, source_bin in enumerate(source_bins):
            sink_pad = streammux.request_pad_simple(f"sink_{i}")
            src_pad = source_bin.get_static_pad("src")

            if src_pad and sink_pad:
                link_result = src_pad.link(sink_pad)
                if link_result != Gst.PadLinkReturn.OK:
                    print(f"❌ 源{i} pad链接失败: {link_result}")
                    sys.exit(1)
                else:
                    print(f"✅ 源{i} 链接成功")
            else:
                print(f"❌ 无法获取源{i}的src pad或streammux的sink pad")
                print(f"   src_pad: {src_pad is not None}")
                print(f"   sink_pad: {sink_pad is not None}")
                sys.exit(1)

    except Exception as e:
        print(f"❌ 多路管线链接失败: {e}")
        sys.exit(1)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv_pretiler)
    nvvidconv_pretiler.link(tiler)
    tiler.link(nvvidconv1)
    nvvidconv1.link(nvvidconv2)

    # 修复nvvidconv2到nvosd的链接 - 使用capsfilter
    if capsfilter:
        nvvidconv2.link(capsfilter)
        capsfilter.link(nvosd)
    else:
        nvvidconv2.link(nvosd)

    # 修复显示链路：nvosd -> nvvidconv_postosd -> sink (nvegltransform不支持)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(sink)

    # 将探针挂载到 nvosd 的 sink pad，这是最稳妥的位置
    # 确保数据是CPU可访问的，避免GPU内存访问问题
    nvosd_sink_pad = nvosd.get_static_pad("sink")
    if not nvosd_sink_pad:
        print("❌ 无法获取 nvosd 的 sink pad！")
        sys.exit(1)

    nvosd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, uris)
    print("✅ 已添加REID处理probe函数到 nvosd sink pad (推荐位置)")

    loop = GLib.MainLoop()

    def shutdown_handler(signum, frame):
        print("\n🛑 收到退出信号，正在优雅地关闭系统...")
        loop.quit()

    # --- 【修改部分：定期保存逻辑】 ---
    def periodic_save_callback():
        # 只有在 --no-save 未被设置时才执行定期保存
        if not args.no_save:
            print("\n⏳ [自动保存] 正在执行定期数据库保存...")
            fastreid_system.save_database(args.db_file, create_backup=False)
        else:
            print("   (根据 --no-save 参数，跳过自动保存)")
        return True  # 返回True让定时器继续运行

    def periodic_cleanup_callback():
        print("\n⏳ [周期性清理] 正在检查不活跃的原型...")
        fastreid_system.cleanup_inactive_prototypes()
        return True  # 返回True让定时器继续运行

    import signal
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 只有在需要保存时才启动定时器 (即没有 --no-save)
    if not args.no_save:
        autosave_interval = 300  # 秒
        print(f"🕒 已启用数据库自动保存，间隔: {autosave_interval} 秒")
        GLib.timeout_add_seconds(autosave_interval, periodic_save_callback)
    else:
        print("🚫 已根据 --no-save 参数禁用数据库自动保存。")

    # 添加原型清理定时器
    cleanup_interval = 600  # 每10分钟(600秒)执行一次清理
    print(f"🕒 已启用不活跃原型清理，间隔: {cleanup_interval} 秒")
    GLib.timeout_add_seconds(cleanup_interval, periodic_cleanup_callback)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    if source_type == "RTSP":
        print("🚀 启动RTSP流Market1501 OSNet FastReID管线...")
    else:
        print("🚀 启动本地视频Market1501 OSNet FastReID管线...")

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("❌ 无法将管线设置为PLAYING状态")
        return

    if has_rtsp:
        print("✅ 多路Market1501 OSNet FastReID管线启动成功（包含RTSP流）")
        print("⏳ 等待RTSP流初始化... (约10-20秒)")

        # 等待管线完全启动
        time.sleep(5)

        # 检查管线状态
        state, pending, _ = pipeline.get_state(Gst.State.NULL)
        print(f"📊 管线状态: {state}")
    else:
        print("✅ 多路Market1501 OSNet FastReID管线启动成功（本地文件）")

    try:
        loop.run()
    except Exception as e:
        print(f"❌ Glib主循环运行时发生错误: {e}")
    finally:
        print("🧹 正在停止管线并清理资源...")
        pipeline.set_state(Gst.State.NULL)

        # --- 【核心修复：最终保存逻辑】 ---
        # 只要没有指定 --no-save，就执行最终保存
        if not args.no_save:
            print("💾 正在执行最终的数据库保存...")
            fastreid_system.save_database(args.db_file, create_backup=True)
        else:
            print("🚫 根据 --no-save 参数，跳过最终的数据库保存。")

        print("✅ Market1501 OSNet系统已完全停止 (V4.2修复版)")

    print("\n📋 使用说明（多路版本）：")
    print("   多路RTSP: python3 fastreid_integration.py --sources rtsp://用户:密码@IP1 rtsp://用户:密码@IP2")
    print("   多路文件: python3 fastreid_integration.py --sources video1.mp4 video2.mp4")
    print("   混合输入: python3 fastreid_integration.py --sources rtsp://用户:密码@IP video1.mp4")
    print("   测试视频: python3 fastreid_integration.py --sources sample_720p.h264")
    print(f"   当前输入源: {uris}")


if __name__ == "__main__":
    main()
