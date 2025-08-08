import sys

import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("未找到 onnx / onnxruntime，请先执行：")
    print("  pip install onnx onnxruntime-gpu")
    sys.exit(1)


def create_identity_model(onnx_path="identity.onnx"):
    """
    生成一个最简单的 y = x 的 ONNX 模型，方便做推理验证
    """
    from onnx import helper, TensorProto

    # 输入输出 tensor
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])

    node = helper.make_node("Identity", ["input"], ["output"])

    graph = helper.make_graph([node], "identity_graph", [X], [Y])
    model = helper.make_model(graph)
    onnx.save(model, onnx_path)
    return onnx_path


def check_gpu_available():
    """
    检测 CUDAExecutionProvider 是否可用
    """
    providers = ort.get_available_providers()
    print("当前可用 ExecutionProvider 列表:", providers)

    if "CUDAExecutionProvider" not in providers:
        print("\nCUDAExecutionProvider 不在列表中，通常说明：")
        print("  1) 没安装 onnxruntime-gpu，或者")
        print("  2) CUDA / cuDNN 运行时环境缺失\n")
        return False
    return True


def run_inference(use_gpu=True):
    """
    实际跑一次推理，验证 GPU 是否能真正 work
    """
    onnx_path = create_identity_model()
    providers_order = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                       if use_gpu else ["CPUExecutionProvider"])

    try:
        sess = ort.InferenceSession(onnx_path, providers=providers_order)
        print(f"\n成功创建 Session, 实际使用的 Provider: {sess.get_providers()[0]}")
    except Exception as e:
        print("\nSession 创建失败:", e)
        return

    dummy_input = {"input": np.random.randn(2, 4).astype(np.float32)}
    output = sess.run(None, dummy_input)[0]
    print("推理输出:\n", output)


if __name__ == "__main__":
    print("===== 步骤 1. 检测 GPU Provider =====")
    gpu_ok = check_gpu_available()

    print("\n===== 步骤 2. 创建 Session 并跑一次推理 =====")
    if gpu_ok:
        run_inference(use_gpu=True)
    else:
        print("改用 CPU 尝试推理……")
        run_inference(use_gpu=False)
