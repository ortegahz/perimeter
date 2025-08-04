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
        "/home/manu/mnt/ST2000DM005-2U91/workspace/Market/bounding_box_test/-1_c1s1_011426_05.*")
    reid_obj.build_gallery(gallery_imgs)

    # 查询
    query_img = "/home/manu/mnt/ST2000DM005-2U91/workspace/Market/bounding_box_test/-1_c1s1_011526_06.jpg"
    top_paths, scores = reid_obj.compare(query_img, topk=10)

    print('Top-K 相似图：')
    for p, s in zip(top_paths, scores):
        print(f'{s:.4f}  {p}')
