# -*- coding: utf-8 -*-
from __future__ import print_function, division

import glob
import os
import sys
import time

import numpy as np
from PIL import Image

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
    # extract_feat 在 ONNX 模式下返回 tensor，所以这个转换仍然有效
    return np.vstack([f.cpu().numpy() for f in feats])  # (N, D)


def save_feats_to_txt(feats, output_path):
    """将特征向量保存到文本文件的辅助函数"""
    print('\n正在将特征保存到 "{}"...'.format(output_path))
    with open(output_path, 'w') as f:
        for i, feat_vec in enumerate(feats):
            # 格式: '图片名(序号) feat1 feat2 feat3 ...'
            feature_str = ' '.join(map(str, feat_vec))
            f.write('{} {}\n'.format(i, feature_str))
    print('特征保存完毕。')


if __name__ == '__main__':
    folderA = '/home/manu/tmp/perimeter_v1/G00003/bodies'
    folderB = '/home/manu/tmp/perimeter_v1/G00001/bodies/'

    model_dir = os.path.join(DIR_PERSON_REID, 'model/ft_ResNet50')
    reid = PersonReid(model_dir, which_epoch='last', gpu='0')

    # ------------------- 任务1: 合并图片并保存为BMP格式 -------------------
    output_dir = '/home/manu/tmp/out_reid'
    os.makedirs(output_dir, exist_ok=True)

    imgsA = sorted(glob.glob(os.path.join(folderA, '*')))
    imgsB = sorted(glob.glob(os.path.join(folderB, '*')))
    all_imgs = imgsA + imgsB
    assert all_imgs, '指定的两个文件夹中都没有找到任何图片！'

    print('总共找到 {} 张图片, 正在合并并以BMP格式保存到 "{}" 文件夹...'.format(len(all_imgs), output_dir))

    for i, img_path in enumerate(all_imgs):
        try:
            # 使用Pillow打开图片并保存为BMP
            img = Image.open(img_path).convert('RGB')  # 确保是RGB格式
            bmp_path = os.path.join(output_dir, '{}.bmp'.format(i))
            img.save(bmp_path)
            if (i + 1) % 200 == 0 or (i + 1) == len(all_imgs):
                print('  [{:d}/{:d}] images saved'.format(i + 1, len(all_imgs)))
        except Exception as e:
            print('警告: 处理 {} 失败. 错误: {}'.format(img_path, e))
    print('所有图片已保存为BMP格式。\n')

    # ------------------- 任务2.1: 使用 PyTorch 提取特征并保存 -------------------
    print('=' * 20, 'PyTorch 推理', '=' * 20)
    print('正在从 "{}" 文件夹读取BMP图片...'.format(output_dir))
    # 获取BMP图片列表并按数字顺序排序
    bmp_paths = glob.glob(os.path.join(output_dir, '*.bmp'))
    bmp_paths_sorted = sorted(bmp_paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    assert bmp_paths_sorted, '"{}" 文件夹中没有找到BMP图片！'.format(output_dir)

    print('开始从 {} 张BMP图片中提取特征 (使用 PyTorch)...'.format(len(bmp_paths_sorted)))
    feats_pt = extract_features_in_batch(reid, bmp_paths_sorted)  # (N, D)

    output_txt_path_pt = '/home/manu/tmp/features_pt.txt'
    save_feats_to_txt(feats_pt, output_txt_path_pt)

    # ------------------- 步骤: 切换到 ONNX 推理模式 -------------------
    onnx_model_path = '/home/manu/tmp/reid_model.onnx'
    print('\n正在准备并切换到 ONNX 推理环境...')
    reid.switch_to_onnx(onnx_model_path)
    print('已切换到 ONNX 推理模式。\n')

    # ------------------- 任务2.2: 使用 ONNX 提取特征并保存 -------------------
    print('=' * 20, 'ONNX 推理', '=' * 20)
    print('开始从 {} 张BMP图片中提取特征 (使用 ONNX)...'.format(len(bmp_paths_sorted)))
    feats_onnx = extract_features_in_batch(reid, bmp_paths_sorted)  # (N, D)

    output_txt_path_onnx = '/home/manu/tmp/features_onnx.txt'
    save_feats_to_txt(feats_onnx, output_txt_path_onnx)

    # ------------------- 任务完成总结 -------------------
    print('\n==============================================')
    print('任务完成!')
    print('  - 合并后的BMP图片位于:  {}'.format(os.path.abspath(output_dir)))
    print('  - PyTorch 特征文件位于: {}'.format(os.path.abspath(output_txt_path_pt)))
    print('  - ONNX 特征文件位于:    {}'.format(os.path.abspath(output_txt_path_onnx)))
    print('==============================================')
