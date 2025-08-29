# -*- coding: utf-8 -*-
from __future__ import print_function, division

import glob
import os
import sys
import time

import numpy as np

from utils_peri.macros import DIR_PERSON_REID

sys.path.append(os.path.abspath(DIR_PERSON_REID))
from cores.personReid import PersonReid


def extract_features_in_batch(reid_obj, img_paths):
    feats = []
    tic = time.time()
    for i, p in enumerate(img_paths, 1):
        feats.append(reid_obj.extract_feat(p))  # (1, D)
        if i % 200 == 0 or i == len(img_paths):
            print('  [{:d}/{:d}] extracted'.format(i, len(img_paths)))
    print('Feature extraction finished in {:.1f}s'.format(time.time() - tic))
    return np.vstack([f.cpu().numpy() for f in feats])  # (N, D)


if __name__ == '__main__':
    folderA = '/home/manu/tmp/perimeter_v1/G00003/bodies'
    folderB = '/home/manu/tmp/perimeter_v1/G00001/bodies/'

    model_dir = os.path.join(DIR_PERSON_REID, 'model/ft_ResNet50')
    reid = PersonReid(model_dir, which_epoch='last', gpu='0')

    # ------------------- 新增代码开始 -------------------
    # 保存为 onnx (带 simplify)
    # 输入尺寸将根据模型自动设置为 (1, 3, 256, 128)
    onnx_path = os.path.join(model_dir, 'model.onnx')
    print('\n[ONNX] 正在保存 ONNX 模型到 -> {} ...'.format(onnx_path))
    reid.save_onnx(onnx_path)
    print('[ONNX] 模型保存完毕。\n')
    # ------------------- 新增代码结束 -------------------

    imgsA = sorted(glob.glob(os.path.join(folderA, '*')))
    imgsB = sorted(glob.glob(os.path.join(folderB, '*')))
    assert imgsA and imgsB, '两个文件夹都必须含有图片！'

    print('FolderA: {} images, FolderB: {} images'.format(len(imgsA), len(imgsB)))

    featsA = extract_features_in_batch(reid, imgsA)  # (NA, D)
    featsB = extract_features_in_batch(reid, imgsB)  # (NB, D)

    # 每行 L2 归一化
    featsA /= np.linalg.norm(featsA, axis=1, keepdims=True) + 1e-12
    featsB /= np.linalg.norm(featsB, axis=1, keepdims=True) + 1e-12

    # 两两余弦相似度矩阵
    sim_matrix = np.matmul(featsA, featsB.T)
    overall_mean = float(sim_matrix.mean())
    print('\n==============================================')
    print('整体平均余弦相似度 (FolderA × FolderB) : {:.4f}'.format(overall_mean))
    print('==============================================')

    # ------------------------------------------------------------------
    # ➊ 计算两个文件夹的“平均特征向量”
    avg_vec_A = featsA.mean(axis=0)  # (D,)
    avg_vec_B = featsB.mean(axis=0)  # (D,)

    # ➋ 再各自做一次 L2 归一化
    avg_vec_A /= np.linalg.norm(avg_vec_A) + 1e-12
    avg_vec_B /= np.linalg.norm(avg_vec_B) + 1e-12

    # ➌ 两平均向量的余弦相似度
    avg_sim = float(np.dot(avg_vec_A, avg_vec_B))
    print('\n==============================================')
    print('平均向量余弦相似度 (mean-A  vs  mean-B) : {:.4f}'.format(avg_sim))
    print('==============================================')
