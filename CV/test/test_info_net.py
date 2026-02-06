from CV.test.test_train_net_1 import SimpleCNN
from Utils.inspect_net import NetInspector


def main():
    net = SimpleCNN()
    analyzer = NetInspector(net)
    analyzer.view_parameters(view_net_struct=True, view_params_details=True, view_params_count=True)


if __name__ == "__main__":
    main()
