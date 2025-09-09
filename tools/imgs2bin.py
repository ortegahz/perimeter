import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 用户需要修改的参数 ---
# (这些参数已根据您的输入保留，无需更改)
IMAGE_DIR = "/home/manu/tmp/WIDER_FACE/retinaface/val/images/"
INPUT_TENSOR_NAME = "input.1"
OUTPUT_BIN_DIR = "/home/manu/nfs/calib_bin_data"
CALIBRATION_LIST_FILE = "/home/manu/nfs/calibration_files.txt"
INPUT_H = 640
INPUT_W = 640


# --- 参数修改结束 ---

def preprocess_image(img_path, target_h, target_w):
    """
    对单个图像进行预处理。
    这个函数中的逻辑与您提供的 Int8Calibrator.get_batch() 中的逻辑完全相同。
    """
    # =======================【核心预处理逻辑 - 与您的脚本完全一致】=======================
    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    im_ratio = float(im_h) / im_w
    mdl_ratio = float(target_h) / target_w

    if im_ratio > mdl_ratio:
        new_h = target_h
        new_w = int(new_h / im_ratio)
    else:
        new_w = target_w
        new_h = int(new_w * im_ratio)

    resized_img = img.resize((new_w, new_h), Image.BILINEAR)

    det_img_np = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    det_img_np[0:new_h, 0:new_w, :] = np.array(resized_img)

    img_data = (det_img_np.astype(np.float32) - 127.5) / 128.0
    img_data = img_data.transpose(2, 0, 1)
    # =======================【预处理逻辑结束】=======================

    return img_data


def main():
    """
    主函数，执行所有步骤。
    """
    print("脚本开始执行...")
    print(f"1. 图片源目录: {IMAGE_DIR}")
    print(f"2. 二进制文件输出目录: {OUTPUT_BIN_DIR}")
    print(f"3. 最终校准列表文件: {CALIBRATION_LIST_FILE}")

    os.makedirs(OUTPUT_BIN_DIR, exist_ok=True)

    # =======================【修改点 1: 递归遍历子文件夹】=======================
    # 旧方法 (os.listdir) 无法处理子文件夹，已替换为 os.walk。
    image_files = []
    print("\n正在扫描所有子文件夹以查找图片...")
    for root, _, files in os.walk(IMAGE_DIR):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, filename))
    # =======================【修改点 1 结束】=======================

    if not image_files:
        print(f"错误: 在目录 '{IMAGE_DIR}' 及其子目录中未找到任何图片文件。请检查路径。")
        return

    print(f"\n找到 {len(image_files)} 张图片，开始预处理并生成 .bin 文件...")

    calibration_file_lines = []

    for img_path in tqdm(image_files, desc="处理图片"):
        preprocessed_data = preprocess_image(img_path, INPUT_H, INPUT_W)

        # =======================【修改点 2: 生成唯一的文件名以防覆盖】=======================
        # 旧方法仅使用 basename，在不同子文件夹同名时会冲突。
        # 新方法使用相对路径替换路径分隔符，生成唯一的文件名。
        relative_path = os.path.relpath(img_path, IMAGE_DIR)
        unique_filename_base = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        output_bin_path = os.path.join(OUTPUT_BIN_DIR, f"{unique_filename_base}.bin")
        # =======================【修改点 2 结束】=======================

        preprocessed_data.astype(np.float32).tofile(output_bin_path)

        line = f"{INPUT_TENSOR_NAME}: {output_bin_path}"
        calibration_file_lines.append(line)

    print(f"\n所有图片处理完成，.bin 文件已保存至 '{OUTPUT_BIN_DIR}'")

    with open(CALIBRATION_LIST_FILE, 'w') as f:
        f.write('\n'.join(calibration_file_lines))

    print(f"成功生成校准列表文件: '{CALIBRATION_LIST_FILE}'")
    print("\n现在您可以使用以下命令运行trtexec进行校准：")

    trtexec_command = (
        f"/usr/src/tensorrt/bin/trtexec \\\n"
        f"    --onnx={ONNX_FILE} \\\n"
        f"    --saveEngine=/path/to/your_engine.engine \\\n"
        f"    --int8 \\\n"
        f"    --calib=/path/to/save/your_calibration.cache \\\n"
        f"    --loadInputs={CALIBRATION_LIST_FILE} \\\n"
        f"    --verbose"
    )
    print("=" * 60)
    print(trtexec_command)
    print("=" * 60)
    print("注意: 请将上面命令中的 ONNX_FILE, saveEngine, calib 路径替换为您的实际路径。")


if __name__ == '__main__':
    ONNX_FILE = "/mnt/nfs/det_10g_simplified.onnx"
    main()
