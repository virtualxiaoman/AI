import torch
from Utils.train_net import NetTrainerFNN


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
