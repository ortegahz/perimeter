import argparse
# 移除了 update_model_dims, 因为我们将直接操作模型
# 新增 traceback 用于更详细的错误输出
import traceback
from typing import List

import onnx
from onnx.shape_inference import infer_shapes


def change_model_input_shape(model_path: str, output_path: str, fixed_shape: List[int]):
    """
    修改 ONNX 模型一个动态输入的形状为固定的值，并更新整个模型。

    Args:
        model_path (str): 原始 ONNX 模型文件路径。
        output_path (str): 修改后模型的保存路径。
        fixed_shape (List[int]): 要设置的固定形状, 例如 [1, 3, 640, 640]。
    """
    try:
        # 加载模型
        model = onnx.load(model_path)
        print(f"模型已从 {model_path} 加载。")

        # --- 使用更直接的方式修改输入形状 ---
        input_to_modify = None
        for inp in model.graph.input:
            shape = inp.type.tensor_type.shape
            if shape and len(shape.dim) == len(fixed_shape):
                is_dynamic = any(not d.dim_value or d.dim_value <= 0 for d in shape.dim)
                if is_dynamic:
                    input_to_modify = inp
                    print(f"\n找到要修改的动态输入: '{inp.name}'")
                    break

        if not input_to_modify:
            print(f"警告: 未找到维度数量为 {len(fixed_shape)} 的动态输入。将尝试直接进行形状推断。")
        else:
            # 直接修改输入张量的维度信息
            shape_proto = input_to_modify.type.tensor_type.shape
            for i, dim_proto in enumerate(shape_proto.dim):
                # 设置新的固定维度值
                dim_proto.dim_value = fixed_shape[i]
                # 清除可能存在的符号名称 (例如 'batch_size')
                if dim_proto.dim_param:
                    dim_proto.ClearField('dim_param')

            new_shape_str = ', '.join(map(str, fixed_shape))
            print(f"已直接将输入 '{input_to_modify.name}' 的形状设置为: [{new_shape_str}]")

        # 使用 ONNX 的形状推断来更新所有中间张量的形状
        # 这是解决 'zero' shapes 问题的关键步骤
        print("\n正在运行形状推断以更新整个模型...")
        model = infer_shapes(model)
        print("形状推断完成。")

        # 检查修改后的模型是否有效
        onnx.checker.check_model(model)
        print("\n模型检查通过。")

        # 保存修改后的模型
        onnx.save(model, output_path)
        print(f"修改后的模型已保存到 {output_path}")

    # --- 增强的错误捕获 ---
    except Exception as e:
        print(f"\n处理模型时发生严重错误: {e}")
        print("=" * 20, "详细追溯信息", "=" * 20)
        traceback.print_exc()  # 打印完整的错误堆栈
        print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        # 描述已更新
        description="将ONNX模型的动态输入形状（如 batch, H, W）修改为固定的值。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --input 和 --output 参数保留了您原来的 default 设置
    parser.add_argument(
        '-i', '--input',
        type=str,
        default="/home/manu/.insightface/models/buffalo_l/det_10g.onnx",
        help="原始 ONNX 模型的路径, e.g., model_dynamic.onnx"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="/home/manu/.insightface/models/buffalo_l/det_10g_simplified.onnx",
        help="修改后模型的保存路径, e.g., model_bs1.onnx"
    )
    # 删除了 --batch_size，新增了 --shape 参数
    parser.add_argument(
        '-s', '--shape',
        type=str,
        default="1x3x640x640",  # 此参数必须提供
        help="要设置的固定输入形状，以逗号或'x'分隔。\n"
             "例如: '1,3,640,640' 或 '1x3x112x112'"
    )

    args = parser.parse_args()

    # 解析 --shape 参数
    try:
        # 支持逗号和'x'作为分隔符
        shape_str = args.shape.replace('x', ',')
        fixed_shape = [int(dim) for dim in shape_str.split(',')]
    except ValueError as e:
        print(f"错误: --shape 参数格式不正确。请使用逗号或'x'分隔的整数。例如: '1,3,640,640'.\n详细信息: {e}")
        exit(1)

    # 调用修改后的函数
    change_model_input_shape(args.input, args.output, fixed_shape)
