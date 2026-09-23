import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from Utils.trainer.train_net import NetTrainerFNN


class NetInspector(NetTrainerFNN):
    """
    专门用于分析网络结构的类，继承自 NetTrainerFNN。
    只需要传入 net 即可使用 view_parameters 功能。
    """

    def __init__(self, net, device=None):
        # 设置设备
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 只初始化 view_parameters 需要的核心属性
        self.net = net.to(self.device)

    #     # 为了防止父类中其他可能被调用的 log 函数报错，可以给基本属性赋空值
    #     self.net_type = "FNN"
    #     self.eval_type = "None"
    #
    # def log_X_y(self):
    #     # 重写此方法为空，因为 view_parameters 不需要打印数据信息
    #     pass


# 使用方法：
# my_net = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
# analyzer = NetAnalyzer(my_net)
# analyzer.view_parameters(view_net_struct=True, view_params_details=True)


class NetworkShapeAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze(self, dummy_input):
        device = next(self.model.parameters()).device
        dummy_input = dummy_input.to(device)

        print(f"--- 开始分析模型形状流向 ---")
        print(f"输入 shape: {list(dummy_input.shape)}")

        # 1. 追踪模型的 forward 过程
        traced_model = symbolic_trace(self.model).to(device)

        # 2. 形状推导：让 dummy_input 跑一遍追踪后的图，并记录 shape
        ShapeProp(traced_model).propagate(dummy_input)
        print(f"{'Name':<25}{'Type':<20}{'Output Shape'}")
        print("-" * 50)
        # 3. 遍历图节点并格式化输出
        for node in traced_model.graph.nodes:
            # 过滤掉输入节点和输出节点，只看中间操作
            if node.op not in ('placeholder', 'output'):
                if 'tensor_meta' in node.meta:
                    # 提取 shape 信息
                    shape = list(node.meta['tensor_meta'].shape)
                    if node.op == "call_module":
                        module = traced_model.get_submodule(node.target)
                        op_name = type(module).__name__
                        name = str(node.target)
                    elif node.op == "call_function":
                        op_name = node.target.__name__
                        name = op_name
                    else:
                        op_name = str(node.target)
                        name = op_name
                    print(
                        f"{name:<25}"
                        f"{op_name:<20}"
                        f"{shape}"
                    )
                    # # 格式化操作名称
                    # op_name = str(node.target)
                    # if hasattr(node.target, '__name__'):  # 处理 F.relu 等函数
                    #     op_name = node.target.__name__
                    #
                    # print(f"经过操作 [{op_name}] 后, x 的 shape 是: {shape}")


if __name__ == "__main__":
    # from Utils.models.cv.classic_net import LeNet5
    from Utils.models.cv.config import CVNetConfig
    from Utils.models.cv.load_net import CVNetFactory

    net = CVNetFactory.create(CVNetConfig(name="resnet50", pretrained=True))
    # 1. 看参数
    analyzer = NetInspector(net)
    analyzer.view_parameters(view_net_struct=True, view_params_details=True, view_params_count=True)
    # 2. 看数据
    dummy_x = torch.randn(33, 3, 28, 28)  # Batch size=33, Channel=3, 28x28
    analyzer = NetworkShapeAnalyzer(net)
    analyzer.analyze(dummy_x)
