# debug_preprocess_deep.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from PIL import Image


def pytorch_preprocess(img_path, H=256, W=128):
    """PyTorch预处理流程"""
    transform = T.Compose([
        T.Resize((H, W), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    tensor = transform(img)
    return tensor.numpy()


def opencv_preprocess(img_path, H=256, W=128):
    """模拟C++端的OpenCV预处理流程"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return img


def analyze_difference(img_path):
    """深入分析差异"""
    pt_result = pytorch_preprocess(img_path)
    cv_result = opencv_preprocess(img_path)

    diff = np.abs(pt_result - cv_result)

    print(f"图像: {img_path}")
    print(f"形状: {pt_result.shape}")
    print(f"PyTorch dtype: {pt_result.dtype}, OpenCV dtype: {cv_result.dtype}")
    print(f"\n统计信息:")
    print(f"最大差异: {np.max(diff):.8f}")
    print(f"平均差异: {np.mean(diff):.8f}")
    print(f"中位数差异: {np.median(diff):.8f}")
    print(f"99.9%分位数差异: {np.percentile(diff, 99.9):.8f}")

    # 找出最大差异的位置
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\n最大差异位置: channel={max_diff_idx[0]}, y={max_diff_idx[1]}, x={max_diff_idx[2]}")
    print(f"该位置的值 - PyTorch: {pt_result[max_diff_idx]:.8f}")
    print(f"该位置的值 - OpenCV:  {cv_result[max_diff_idx]:.8f}")

    # 检查是否是边缘像素
    h, w = diff.shape[1], diff.shape[2]
    if max_diff_idx[1] == 0 or max_diff_idx[1] == h - 1 or max_diff_idx[2] == 0 or max_diff_idx[2] == w - 1:
        print("注意：最大差异出现在边缘像素！")

    # 可视化差异
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 显示原始预处理结果和差异
    for i, (data, title) in enumerate([
        (pt_result[0], 'PyTorch Ch0'),
        (cv_result[0], 'OpenCV Ch0'),
        (diff[0], 'Difference Ch0'),
        (diff.max(axis=0), 'Max Diff (all channels)')
    ]):
        im = axes[i].imshow(data, cmap='hot' if 'Diff' in title else 'gray')
        axes[i].set_title(title)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig('/home/manu/tmp/preprocess_diff.png', dpi=150)
    print("\n差异可视化已保存到 /home/manu/tmp/preprocess_diff.png")

    # 测试不同的resize方法
    print("\n测试不同的resize插值方法:")
    for interp_cv, interp_name in [(cv2.INTER_CUBIC, 'CUBIC'),
                                   (cv2.INTER_LINEAR, 'LINEAR'),
                                   (cv2.INTER_LANCZOS4, 'LANCZOS4')]:
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (128, 256), interpolation=interp_cv)
        print(f"  {interp_name}: shape={img_resized.shape}, dtype={img_resized.dtype}")


if __name__ == "__main__":
    test_img = "/home/manu/tmp/out_reid/0.bmp"
    analyze_difference(test_img)

    # 额外测试：直接比较resize的结果
    print("\n=== 直接比较resize结果 ===")

    # PIL resize
    img_pil = Image.open(test_img).convert('RGB')
    img_pil_resized = img_pil.resize((128, 256), Image.BICUBIC)
    arr_pil = np.array(img_pil_resized)

    # OpenCV resize
    img_cv = cv2.imread(test_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv_resized = cv2.resize(img_cv, (128, 256), interpolation=cv2.INTER_CUBIC)

    resize_diff = np.abs(arr_pil.astype(np.float32) - img_cv_resized.astype(np.float32))
    print(f"PIL vs OpenCV resize 最大差异: {np.max(resize_diff):.2f}")
    print(f"PIL vs OpenCV resize 平均差异: {np.mean(resize_diff):.4f}")
