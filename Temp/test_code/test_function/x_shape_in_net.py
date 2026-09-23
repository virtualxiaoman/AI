# # from torchinfo import summary
# #
# # net = LeNet5()
# # # col_names 允许你自定义要显示的信息
# # summary(net, input_size=(1, 1, 28, 28), col_names=["input_size", "output_size", "num_params"])
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.fx import symbolic_trace
# from torch.fx.passes.shape_prop import ShapeProp
#
#
#
#
#
# # ==========================================
# # 我们来编写这个分析类
# # ==========================================
# class NetworkShapeAnalyzer:
#     def __init__(self, model):
#         self.model = model
#
#     def analyze(self, dummy_input):
#         print(f"--- 开始分析模型形状流向 ---")
#         print(f"初始输入 shape: {list(dummy_input.shape)}")
#
#         # 1. 追踪模型的 forward 过程
#         traced_model = symbolic_trace(self.model)
#
#         # 2. 形状推导：让 dummy_input 跑一遍追踪后的图，并记录 shape
#         ShapeProp(traced_model).propagate(dummy_input)
#
#         # 3. 遍历图节点并格式化输出
#         for node in traced_model.graph.nodes:
#             # 过滤掉输入节点和输出节点，只看中间操作
#             if node.op not in ('placeholder', 'output'):
#                 if 'tensor_meta' in node.meta:
#                     # 提取 shape 信息
#                     shape = list(node.meta['tensor_meta'].shape)
#
#                     # 格式化操作名称（让它更好读）
#                     op_name = str(node.target)
#                     if hasattr(node.target, '__name__'):  # 处理 F.relu 等函数
#                         op_name = node.target.__name__
#
#                     print(f"经过操作 [{op_name}] 后, x 的 shape 是: {shape}")
#
#
# # 测试运行
# if __name__ == "__main__":
#     net = LeNet5()
#     dummy_x = torch.randn(33, 1, 28, 28)  # Batch size=1, Channel=1, 28x28
#
#     analyzer = NetworkShapeAnalyzer(net)
#     analyzer.analyze(dummy_x)
