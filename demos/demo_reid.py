from __future__ import print_function, division

import argparse
import math
import os
import sys
import time

import yaml

from utils_peri.macros import DIR_PERSON_REID
sys.path.append(os.path.abspath(DIR_PERSON_REID))

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, \
    ft_net_swinv2, ft_net_efficient, ft_net_NAS, \
    ft_net_convnext, PCB, PCB_test

from utils import fuse_all_conv_bn

######################################################################
#  1. 解析输入
parser = argparse.ArgumentParser(description='pair test')
parser.add_argument('--img1',
                    default="/home/manu/mnt/ST2000DM005-2U91/workspace/Market/bounding_box_test/-1_c1s1_011526_06.jpg")
parser.add_argument('--img2',
                    default="/home/manu/mnt/ST2000DM005-2U91/workspace/Market/bounding_box_test/-1_c1s1_011426_05.jpg")
parser.add_argument('--name', default='ft_ResNet50', type=str,
                    help='保存模型所在的文件夹名(model/<name>/...)')
parser.add_argument('--which_epoch', default='last', type=str,
                    help='net_xxx.pth 中的 xxx')
parser.add_argument('--gpu', default='0', type=str)
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################
#  2. 读取训练时保存的配置（opts.yaml）
cfg_path = os.path.join(os.path.join(DIR_PERSON_REID, "model"), opt.name, 'opts.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# ---------- 读取开关 ----------
use_dense = cfg.get('use_dense', False)
use_efficient = cfg.get('use_efficient', False)
use_hr = cfg.get('use_hr', False)
use_swin = cfg.get('use_swin', False)
use_swinv2 = cfg.get('use_swinv2', False)
use_convnext = cfg.get('use_convnext', False)
use_NAS = cfg.get('use_NAS', False)
PCB_flag = cfg.get('PCB', False)
ibn_flag = cfg.get('ibn', False)
usam_flag = cfg.get('usam', False)
stride = cfg.get('stride', 2)
linear_num = cfg.get('linear_num', 512)
nclasses = cfg.get('nclasses', 751)

######################################################################
#  3. 创建并载入模型
if use_dense:
    model = ft_net_dense(nclasses, stride=stride, linear_num=linear_num)
elif use_NAS:
    model = ft_net_NAS(nclasses, linear_num=linear_num)
elif use_swin:
    model = ft_net_swin(nclasses, linear_num=linear_num)
elif use_swinv2:
    model = ft_net_swinv2(nclasses, (224, 224), linear_num=linear_num)
elif use_convnext:
    model = ft_net_convnext(nclasses, linear_num=linear_num)
elif use_efficient:
    model = ft_net_efficient(nclasses, linear_num=linear_num)
elif use_hr:
    model = ft_net_hr(nclasses, linear_num=linear_num)
else:
    model = ft_net(nclasses, stride=stride, ibn=ibn_flag,
                   linear_num=linear_num, usam=usam_flag)

if PCB_flag:
    model = PCB(nclasses)

weights = os.path.join(os.path.join(DIR_PERSON_REID, "model"), opt.name, 'net_%s.pth' % opt.which_epoch)
state = torch.load(weights, map_location='cpu')
model.load_state_dict(state, strict=False)

# 去掉分类器，只保留特征
if PCB_flag:
    model = PCB_test(model)
else:
    model.classifier.classifier = nn.Sequential()

model.eval()
model = model.to(device)
model = fuse_all_conv_bn(model)  # conv+bn 融合

######################################################################
#  4. 处理单张图片 -> Tensor
if use_swin:  # swin 系列是 224×224
    H, W = 224, 224
else:
    H, W = 256, 128
if PCB_flag:
    H, W = 384, 192

transform = T.Compose([
    T.Resize((H, W), interpolation=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_img(path):
    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0)  # (1,C,H,W)
    return img.to(device)


######################################################################
#  5. 提取单张图片的特征（包含水平翻转增强）
def fliplr(x):
    """水平翻转"""
    inv_idx = torch.arange(x.size(3) - 1, -1, -1, device=x.device).long()
    return x.index_select(3, inv_idx)


@torch.no_grad()
def extract_one(img_tensor):
    """
    img_tensor: (1,C,H,W)
    Return: (1, feat_dim)
    """
    if linear_num <= 0:  # 根据 backbone 决定 dim
        if use_swin or use_swinv2 or use_dense or use_convnext:
            feat_dim = 1024
        elif use_efficient:
            feat_dim = 1792
        elif use_NAS:
            feat_dim = 4032
        else:
            feat_dim = 2048
    else:
        feat_dim = linear_num
    if PCB_flag:
        feat = torch.zeros(1, 2048, 6).to(device)
    else:
        feat = torch.zeros(1, feat_dim).to(device)

    for flip in [False, True]:
        inp = fliplr(img_tensor) if flip else img_tensor
        out = model(inp)
        feat += out
    # 归一化
    if PCB_flag:
        fn = torch.norm(feat, p=2, dim=1, keepdim=True) * math.sqrt(6)
        feat = feat.div(fn.expand_as(feat)).view(1, -1)
    else:
        fn = torch.norm(feat, p=2, dim=1, keepdim=True)
        feat = feat.div(fn.expand_as(feat))
    return feat


######################################################################
#  6. 计算两幅图的相似度
img1 = load_img(opt.img1)
img2 = load_img(opt.img2)

start = time.time()
f1 = extract_one(img1)
f2 = extract_one(img2)
cost = time.time() - start

# 余弦相似度
sim = torch.mm(f1, f2.t()).item()

# 欧氏距离
dist = torch.norm(f1 - f2, p=2).item()

print('------------------------------------------------------')
print('Image 1 :', opt.img1)
print('Image 2 :', opt.img2)
print('Forward time: %.3f s' % cost)
print('Cosine similarity : %.6f  (1.0 越相似)' % sim)
print('L2 / Euclidean  d : %.6f  (0   越相似)' % dist)
print('------------------------------------------------------')
