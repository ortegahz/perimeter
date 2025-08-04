from __future__ import print_function, division

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image

from utils_peri.macros import DIR_PERSON_REID

sys.path.append(os.path.abspath(DIR_PERSON_REID))
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, \
    ft_net_swinv2, ft_net_efficient, ft_net_NAS, \
    ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn


class PersonReid:
    """
    使用训练好的 person-ReID 模型，计算两张图之间或一张图与 whole-gallery
    之间的余弦相似度。
    """

    def __init__(self,
                 model_root: str,  # eg.  "model/ft_ResNet50"
                 which_epoch='last',
                 gpu: str = '0'):
        """
        model_root 目录结构：
           model_root/
               opts.yaml
               net_last.pth  (或其它 epoch 名)
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 读取配置文件
        cfg_path = os.path.join(model_root, 'opts.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # 2. 创建 backbone
        self.PCB_flag = cfg.get('PCB', False)
        self.use_dense = cfg.get('use_dense', False)
        self.use_swin = cfg.get('use_swin', False)
        self.use_swinv2 = cfg.get('use_swinv2', False)
        self.use_efficient = cfg.get('use_efficient', False)
        self.use_convnext = cfg.get('use_convnext', False)
        self.use_NAS = cfg.get('use_NAS', False)
        self.use_hr = cfg.get('use_hr', False)
        self.ibn_flag = cfg.get('ibn', False)
        self.usam_flag = cfg.get('usam', False)
        self.stride = cfg.get('stride', 2)
        self.linear_num = cfg.get('linear_num', 512)
        self.nclasses = cfg.get('nclasses', 751)

        self.model = self._build_model()
        # 3. 载入权重
        weight_file = os.path.join(model_root, f'net_{which_epoch}.pth')
        state = torch.load(weight_file, map_location='cpu')
        self.model.load_state_dict(state, strict=False)

        # 4. 去掉分类器 & fuse bn
        if self.PCB_flag:
            self.model = PCB_test(self.model)
        else:
            self.model.classifier.classifier = nn.Sequential()

        self.model.eval().to(self.device)
        self.model = fuse_all_conv_bn(self.model)

        # 5. 输入尺寸 & 预处理
        if self.PCB_flag:
            self.H, self.W = 384, 192
        elif self.use_swin:  # swin backbone
            self.H, self.W = 224, 224
        else:
            self.H, self.W = 256, 128

        self.transform = T.Compose([
            T.Resize((self.H, self.W), interpolation=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        # gallery 特征缓存
        self.gallery_paths = []
        self.gallery_feats = None  # (N, dim)

    # ------------------------------------------------------------------
    def _build_model(self):
        """根据 cfg 构造骨干网络"""
        if self.use_dense:
            model = ft_net_dense(self.nclasses,
                                 stride=self.stride,
                                 linear_num=self.linear_num)
        elif self.use_NAS:
            model = ft_net_NAS(self.nclasses,
                               linear_num=self.linear_num)
        elif self.use_swin:
            model = ft_net_swin(self.nclasses,
                                linear_num=self.linear_num)
        elif self.use_swinv2:
            model = ft_net_swinv2(self.nclasses, (224, 224),
                                  linear_num=self.linear_num)
        elif self.use_convnext:
            model = ft_net_convnext(self.nclasses,
                                    linear_num=self.linear_num)
        elif self.use_efficient:
            model = ft_net_efficient(self.nclasses,
                                     linear_num=self.linear_num)
        elif self.use_hr:
            model = ft_net_hr(self.nclasses,
                              linear_num=self.linear_num)
        else:
            model = ft_net(self.nclasses,
                           stride=self.stride,
                           ibn=self.ibn_flag,
                           linear_num=self.linear_num,
                           usam=self.usam_flag)
        if self.PCB_flag:
            model = PCB(self.nclasses)
        return model

    # ------------------------------------------------------------------
    def _fliplr(self, x):
        inv = torch.arange(x.size(3) - 1, -1, -1,
                           device=x.device).long()
        return x.index_select(3, inv)

    @torch.no_grad()
    def extract_feat(self, img):
        """
        img : str(PIL路径) 或 PIL.Image 或 已经是 tensor (C,H,W)
        return : (1, feat_dim) torch float32, 已 L2 归一化
        """
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        if isinstance(img, Image.Image):
            img = self.transform(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1,C,H,W)
        img = img.to(self.device)

        # 推理 + flip 增强
        if self.linear_num <= 0:
            feat_dim = {True: 1024}.get(self.use_swin or
                                        self.use_swinv2 or
                                        self.use_dense or
                                        self.use_convnext,
                                        2048)
        else:
            feat_dim = self.linear_num
        feat = torch.zeros((1, 2048, 6) if self.PCB_flag
                           else (1, feat_dim),
                           device=self.device)

        for flip in [False, True]:
            inp = self._fliplr(img) if flip else img
            feat += self.model(inp)

        # 归一化
        if self.PCB_flag:
            feat = feat / (feat.norm(p=2, dim=1, keepdim=True)
                           * math.sqrt(6))
            feat = feat.view(1, -1)
        else:
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)

        return feat  # (1, D)

    # ------------------------------------------------------------------
    def build_gallery(self, img_paths: list):
        """
        预先把一批 gallery 图像转成特征并缓存。
        """
        feats_list = []
        tic = time.time()
        for p in img_paths:
            feats_list.append(self.extract_feat(p))
        self.gallery_feats = torch.cat(feats_list, 0)  # (N,D)
        self.gallery_paths = list(img_paths)
        print(f'[ReID] build gallery: {len(img_paths)} images '
              f'in {time.time() - tic:.1f}s')

    # ------------------------------------------------------------------
    def compare(self, query, topk=5):
        """
        query : path / PIL / tensor
        return :
          top_paths  -> list[str]     (length = topk)
          top_score  -> list[float]   余弦相似度(大=相似)
        """
        assert self.gallery_feats is not None, \
            'call build_gallery() first!'
        qf = self.extract_feat(query)  # (1,D)
        sims = torch.mm(self.gallery_feats, qf.t()).squeeze(1)  # (N,)
        score, idx = sims.topk(k=min(topk, sims.size(0)), largest=True)
        top_paths = [self.gallery_paths[i] for i in idx.cpu().tolist()]
        return top_paths, score.cpu().tolist()
