from __future__ import print_function, division

import os
import sys

from cores.personReid import PersonReid
from utils_peri.macros import DIR_PERSON_REID

sys.path.append(os.path.abspath(DIR_PERSON_REID))

if __name__ == '__main__':
    model_dir = os.path.join(DIR_PERSON_REID, 'model/ft_ResNet50')
    reid_obj = PersonReid(model_dir, which_epoch='last', gpu='0')

    # 构建 gallery
    import glob

    gallery_imgs = glob.glob(
        "/home/manu/tmp/perimeter/G00002/bodies/*")
    reid_obj.build_gallery(gallery_imgs)

    # 查询
    query_img = "/home/manu/tmp/perimeter/G00001/bodies/00000.jpg"
    top_paths, scores = reid_obj.compare(query_img, topk=64)

    print('Top-K 相似图：')
    for p, s in zip(top_paths, scores):
        print(f'{s:.4f}  {p}')
