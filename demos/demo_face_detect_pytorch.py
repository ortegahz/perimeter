from __future__ import print_function

import argparse
# import csv  # 不再需要
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils_peri.macros import DIR_RETINAFACE_PYTORCH

sys.path.append(os.path.abspath(DIR_RETINAFACE_PYTORCH))

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

parser = argparse.ArgumentParser(description='Retinaface')

# 修改命令行参数
parser.add_argument('--image_folder', type=str, default="/home/manu/tmp/perimeter_v1/G00002/faces/")
parser.add_argument('--output_txt', default='/home/manu/nfs/detections_py.txt', type=str,
                    help='Path for the single output TXT file with all detection results')  # 新增参数
parser.add_argument('-m', '--trained_model',
                    default='/media/manu/ST8000DM004-2U91/insightface/models/retinaface_pytorch_mn025_relu/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True,
                    help='Save detection results with boxes and landmarks')
parser.add_argument('--save_folder', default='/home/manu/tmp/results/', type=str,
                    help='Directory to save annotated images')  # 帮助文本中移除 txt
# parser.add_argument('--save_txt', action="store_true", default=True, # 移除此参数
#                     help='Save results in txt format, one file per image')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # ======================= MODIFICATION START ===========================
    # 原有的处理单张图片的部分被替换为处理整个文件夹的逻辑

    if not args.image_folder or not os.path.isdir(args.image_folder):
        print(f"Error: Please specify a valid folder using --image_folder")
        sys.exit(1)

    # 如果需要保存图片，则创建结果文件夹
    if args.save_image:
        os.makedirs(args.save_folder, exist_ok=True)

    # 打开一个总的TXT文件用于写入所有检测结果
    with open(args.output_txt, 'w') as txtfile:
        # 写入TXT文件的表头 (可选, 以#开头作为注释)
        header = ['filename', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'score',
                  'kps0_x', 'kps0_y', 'kps1_x', 'kps1_y', 'kps2_x', 'kps2_y',
                  'kps3_x', 'kps3_y', 'kps4_x', 'kps4_y']
        txtfile.write(",".join(header) + '\n')

        image_files = [f for f in os.listdir(args.image_folder) if
                       f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        # 对图片列表进行排序
        image_files.sort()
        print(f"Found {len(image_files)} images in '{args.image_folder}'")

        resize = 1

        # 开始遍历文件夹中的图片
        for filename in image_files:
            image_path = os.path.join(args.image_folder, filename)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if img_raw is None:
                print(f"Warning: Could not read image {image_path}, skipping.")
                continue

            # ----------------- MODIFICATION START: Image padding and resizing -----------------
            # Original image dimensions
            orig_h, orig_w, _ = img_raw.shape

            # Target size
            target_size = 640

            # Calculate scaling factor and new size
            scale_ratio = target_size / max(orig_h, orig_w)
            new_w = int(orig_w * scale_ratio)
            new_h = int(orig_h * scale_ratio)

            # Resize image
            resized_img = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create a black canvas and paste the resized image
            padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            padded_img[0:new_h, 0:new_w] = resized_img

            # The network will now process the padded image
            img = np.float32(padded_img)
            # ----------------- MODIFICATION END ------------------------------------------

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            loc, conf, landms = net(img)  # forward pass
            print('Processing {}: net forward time: {:.4f}'.format(filename, time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # 将检测信息保存到总的TXT文件，并绘制标注图
            for b in dets:
                if b[4] < args.vis_thres:
                    continue

                # ----------------- MODIFICATION START: Rescale coordinates -----------------
                # Rescale coordinates from padded 640x640 space to original image space
                b_rescaled = b.copy()
                b_rescaled[0:4] /= scale_ratio  # Rescale bbox
                b_rescaled[5:] /= scale_ratio  # Rescale landmarks
                # ----------------- MODIFICATION END --------------------------------------

                # 1. 准备数据 (使用 b_rescaled)
                file_name_col = filename
                bbox_x = b_rescaled[0]
                bbox_y = b_rescaled[1]
                bbox_width = b_rescaled[2] - b_rescaled[0]
                bbox_height = b_rescaled[3] - b_rescaled[1]
                score_col = b_rescaled[4]
                kps_cols = b_rescaled[5:].tolist()

                # 2. 写入总的TXT文件
                row_data = [file_name_col, bbox_x, bbox_y, bbox_width, bbox_height, score_col] + kps_cols
                line_to_write = ",".join(map(str, row_data)) + "\n"
                txtfile.write(line_to_write)

                # 3. 在图片上绘制检测结果 (使用 b_rescaled)
                if args.save_image:
                    text = "{:.4f}".format(b_rescaled[4])
                    b_int = list(map(int, b_rescaled))
                    cv2.rectangle(img_raw, (b_int[0], b_int[1]), (b_int[2], b_int[3]), (0, 0, 255), 2)
                    cx = b_int[0]
                    cy = b_int[1] + 12
                    cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    # landms
                    cv2.circle(img_raw, (b_int[5], b_int[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b_int[7], b_int[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b_int[9], b_int[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b_int[11], b_int[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b_int[13], b_int[14]), 1, (255, 0, 0), 4)

            # --- 循环结束后，保存标注图片 ---
            if args.save_image:
                save_path = os.path.join(args.save_folder, filename)
                cv2.imwrite(save_path, img_raw)

    print(f"\nProcessing complete. All detections saved to '{args.output_txt}'.")
    if args.save_image:
        print(f"Annotated images are saved in '{args.save_folder}'.")
    # ======================= MODIFICATION END =============================
