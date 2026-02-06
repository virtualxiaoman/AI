import os
import time
import warnings
import torch
from torch import nn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F


class NetTrainerFNN:
    """
    这是一个用于训练FNN的类，支持回归和分类，支持前馈神经网络，不支持RNN这种需要隐藏层的网络。
    """

    def __init__(self, train_loader, test_loader, net, loss_fn, optimizer,  # 必要参数，数据与网络的基本信息
                 scheduler=None,  # 可选参数，学习率调度器
                 epochs=100,  # 可选参数，用于训练
                 eval_type=None,  # 比较重要的参数，用于选择训练的类型（与评估指标有关）
                 eval_during_training=True,  # 可选参数，训练时是否进行评估（与显存、训练时间有关）
                 eval_interval=10,  # 其他参数，训练时的评估间隔
                 device=None,  # 其他参数，设备选择，默认优先cuda
                 **kwargs):
        """
        初始化模型。

        :param data: 数据(全部数据/已经划分好了的元组)或训练集(包含X, y的DataLoader): X or (X_train, X_test) or train_loader
        :param target: 目标(全部数据/已经划分好了的元组)或验证集(包含X, y的DataLoader): y or (y_train, y_test) or test_loader
        :param net: 支持 net=nn.Sequential() or class Net(nn.Module)
        :param loss_fn: 损失函数，例如：
            nn.MSELoss()  # 回归，y的维度应该是(batch,)
            nn.CrossEntropyLoss()  # 分类，y的维度应该是(batch,)，并且网络的最后一层不需要加softmax
            nn.BCELoss()  # 二分类，y的维度应该是(batch,)，并且网络的最后一层需要加sigmoid
        :param optimizer: 优化器
        :param test_size: 测试集大小，支持浮点数或整数。该参数在data和target是tuple时无效
        :param batch_size: 批量大小
        :param epochs: 训练轮数
        :param eval_type: 模型类型，只可以是"loss"(回归-损失)或"acc"(分类-准确率)
        :param eval_interval: 打印间隔，请注意train_loss_list等间隔也是这个
        :param eval_during_training: 训练时是否进行评估，当显存不够时，可以设置为False，等到训练结束之后再进行评估
          设置为False时，不会影响训练集上的Loss的输出，但是无法输出验证集上的loss、训练集与验证集上的acc，此时默认输出"No eval"
        :param rnn_input_size: RNN的输入维度
        :param rnn_seq_len: RNN的序列长度
        :param rnn_hidden_size: RNN的隐藏层大小
          以上三个参数同时设置时，自动判断网络类型为RNN
        :param device: 设备，支持"cuda"或"cpu"，默认为None，自动优先选择"cuda"
        :param kwargs: 其他参数，包括：
            free_memory: 是否在每次迭代后释放显存，默认为False
          # drop_last: 是否丢弃最后一个batch，默认为False，用于DataLoader
        """
        # 设备参数
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"[__init__] 当前设备为{self.device}")

        # self.target_reshape_1D = kwargs.get("target_reshape_1D", True)
        # self.drop_last = kwargs.get("drop_last", False)
        self.free_memory = kwargs.get("free_memory", False)  # 是否在每次迭代后释放显存
        self.net_name = kwargs.get("net_name", '')
        use_amp = kwargs.get("use_amp", False)  # 是否使用自动混合精度
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.scheduler = scheduler  # 学习率调度器

        # 数据参数
        self.train_loader = train_loader  # train_loader
        self.test_loader = test_loader  # test_loader
        # self.test_size = test_size
        # self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        # self.train_loader, self.test_loader = None, None

        # 网络参数
        self.net = net.to(self.device)
        # self.net = torch.compile(self.net)  # RuntimeError: Windows not yet supported for torch.compile 哈哈哈！
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # 训练参数
        # self.batch_size = batch_size
        self.epochs = epochs

        # 训练输出参数
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []
        self.time_list = []
        self.eval_interval = eval_interval  # 打印间隔

        # 使用loss还是acc参数
        self.eval_type = eval_type
        self._init_eval_type()

        # 当前训练epoch的各种参数
        self.loss_epoch = None  # 每个epoch的loss
        self.current_gpu_memory = None  # 当前GPU显存

        # 训练时是否进行评估
        self.eval_during_training = eval_during_training
        self.NO_EVAL_MSG = '"No eval"'  # 不在训练时评估的输出
        self.best_epoch = None  # 最佳epoch
        self.best_loss = None  # 最佳loss
        self.best_acc = None  # 最佳acc
        self.auto_save_best_net = False  # 是否自动保存最佳模型

        self.log_X_y()  # 打印数据信息

    def _init_eval_type(self):
        """
        初始化 eval_type，根据 loss_fn 动态判断任务类型。
        """
        # 定义支持的回归和分类损失函数
        regression_losses = (
            nn.MSELoss,  # 均方误差，用于回归任务
            nn.L1Loss,  # L1 损失（绝对误差），用于回归任务
            nn.SmoothL1Loss,  # 平滑 L1 损失（Huber 损失），用于回归任务
            nn.HuberLoss,  # Huber 损失，用于稳健回归
            nn.PoissonNLLLoss  # 泊松负对数似然，用于回归（例如计数数据）
        )

        classification_losses = (
            nn.CrossEntropyLoss,  # 多分类任务
            nn.NLLLoss,  # 负对数似然，多分类任务
            nn.BCEWithLogitsLoss,  # 二分类（带 logits 的 BCE）
            nn.BCELoss,  # 二分类（输入为概率）
            nn.HingeEmbeddingLoss,  # 用于二分类和嵌入任务
            nn.MarginRankingLoss,  # 用于学习排序
            nn.MultiLabelSoftMarginLoss  # 多标签分类
        )

        if self.eval_type is None:
            # 判断 loss_fn 的类型
            if isinstance(self.loss_fn, regression_losses):
                self.eval_type = "loss"  # 回归任务，用 loss 评估
                print(f"[__init__] 根据loss_fn自动检测到目前为回归任务，eval_type={self.eval_type}")
            elif isinstance(self.loss_fn, classification_losses):
                self.eval_type = "acc"  # 分类任务，用 acc 评估
                print(f"[__init__] 根据loss_fn自动检测到目前为分类任务，eval_type={self.eval_type}")
            else:
                raise ValueError(f"[__init__] 未知的任务类型，请手动设置eval_type参数")
        elif self.eval_type not in ["loss", "acc"]:
            raise ValueError(f"[__init__] eval_type参数只能是 'loss' 或 'acc' ")
        else:
            # print(f"[__init__] eval_type={self.eval_type}")
            pass

    def log_X_y(self):
        """
        输出第一批次的X, y, outputs的shape。以及网络的参数量
        """
        # 获取样本总量
        train_samples = len(self.train_loader.dataset)
        test_samples = len(self.test_loader.dataset) if self.test_loader else 0
        print(f"[log] 数据总量：Train: {train_samples}, Test: {test_samples}")

        X, y = next(iter(self.train_loader))
        X, y = X.to(self.device), y.to(self.device)
        outputs = self.net(X)
        print(f"[log] 第一批次的shape如下：\n"
              f"      X: {X.shape}, y: {y.shape}, outputs: {outputs.shape}")
        if (len(outputs.shape) == 1 or outputs.shape[1] == 1) and self.eval_type == "acc":
            print(f"[log] 请注意：outputs的维度为{outputs.shape}，在计算acc的时候使用的是"
                  f"predictions = (torch.sigmoid(outputs).view(-1) > 0.5).long()")

        print("[log] ", end='')
        self.view_parameters(view_net_struct=False, view_params_count=True, view_params_details=False)

    # [主函数]训练模型
    def train_net(self, net_save_path: str = None) -> None:
        """
        训练模型
        :param net_save_path: 最佳模型保存path，会在每次评估后保存最佳模型，该参数在eval_during_training=True时有效。
            暂不支持选择state_dict的保存除非改代码，只支持整个模型的保存（因为我平时不怎么用state_dict阿巴阿巴）
        :return: None
        """
        print(f">>> [train_net] (^v^)开始训练模型，参数epochs={self.epochs}。"
              f"当前设备为{self.device}，评估类型为{self.eval_type}。")
        self.__check_best_net_save_path(net_save_path)
        self.current_gpu_memory = self._log_gpu_memory()

        for epoch in range(self.epochs):
            self.net.train()  # 确保dropout等在训练时生效
            self.train_epoch()  # 训练的主体部分
            # 打印训练信息
            if epoch % self.eval_interval == 0:
                self.log_and_update_eval_msg(epoch, net_save_path)

        print(f">>> [train_net] (*^w^*) Congratulations！{self.net_name}训练结束，总共花费时间: {sum(self.time_list)}秒")
        if self.eval_during_training:
            if self.eval_type == "loss":
                print(f"[train_net] 最佳结果 epoch = {self.best_epoch + 1}, loss = {self.best_loss}")
            elif self.eval_type == "acc":
                print(f"[train_net] 最佳结果 epoch = {self.best_epoch + 1}, acc = {self.best_acc}")
        self.eval_during_training = True  # 训练完成后，可以进行评估

    # 对某个epoch进行训练，仅在train_net中调用。可以通过复写这个函数来实现自定义的训练
    def train_epoch(self):
        start_time = time.time()
        loss_sum = 0.0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)  # 初始化数据
            self.optimizer.zero_grad()
            # print(f"X: {X.shape}, y: {y.shape}, output: {self.net(X).shape}")
            # outputs = self.net(X).view(y.shape)
            # 加入amp可以提升速度、节省显存
            with autocast(enabled=self.use_amp, device_type=self.device.type, dtype=torch.float16):
                outputs = self.net(X)  # 前向传播
                loss = self.loss_fn(outputs, y)

            self.scaler.scale(loss).backward()  # 替换反向传播loss.backward()
            self.scaler.step(self.optimizer)  # 替换更新参数self.optimizer.step()
            self.scaler.update()  # 新增，更新scaler
            loss_sum += loss.item()  # 计算损失

            self.current_gpu_memory = self._log_gpu_memory()  # 计算当前GPU显存
            if self.free_memory:
                del X, y, outputs, loss  # 释放显存。如果不释放，直到作用域结束时才会释放显存（这部分一直在reserve的显存里面）
                # torch.cuda.empty_cache()  # 该步会影响速度
        if self.scheduler is not None:
            self.scheduler.step()  # 每个epoch学习率调度器更新，暂不支持OneCycleLR这样的需要在每个batch更新的调度器
        self.loss_epoch = loss_sum / len(self.train_loader)
        if torch.isnan(torch.tensor(self.loss_epoch)):
            warnings.warn("请注意Loss是NaN!", RuntimeWarning)
        self.time_list.append(time.time() - start_time)

    # 在训练时评估并保存最佳模型，仅在train_net中调用。此函数会不断存储最佳模型，只是怕后面哪一次意外失败了那就白训练了
    def log_and_update_eval_msg(self, epoch, net_save_path):
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.eval_type == "loss":
            self.train_loss_list.append(self.loss_epoch)
            self.test_loss_list.append(self.evaluate_net())
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {self.loss_epoch:.6f}, '
                  f'Test Loss: {self.test_loss_list[-1]:.6f}, '
                  f'Time: {self.time_list[-1]:.2f}s, '
                  f'LR: {current_lr:.6f}, '
                  f'GPU: {self.current_gpu_memory}')
            if self.eval_during_training:
                # 如果当前loss小于最佳loss，则保存self.epoch和self.loss
                if self.best_loss is None or self.test_loss_list[-1] < self.best_loss:
                    self.best_loss = self.test_loss_list[-1]
                    self.best_epoch = epoch
                    if self.auto_save_best_net:
                        self.__save_net(net_save_path)
        elif self.eval_type == "acc":
            self.train_acc_list.append(self.evaluate_net(eval_type="train"))
            self.test_acc_list.append(self.evaluate_net())
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {self.loss_epoch:.6f}, '
                  f'Train Acc: {self.train_acc_list[-1]:.6f}, '
                  f'Test Acc: {self.test_acc_list[-1]:.6f}, '
                  f'Time: {self.time_list[-1]:.2f}s, '
                  f'LR: {current_lr:.6f}, '
                  f'GPU: {self.current_gpu_memory}')
            if self.eval_during_training:
                # 如果当前acc大于最佳acc，则保存self.epoch和self.acc
                if self.best_acc is None or self.test_acc_list[-1] > self.best_acc:
                    self.best_acc = self.test_acc_list[-1]
                    self.best_epoch = epoch
                    if self.auto_save_best_net:
                        self.__save_net(net_save_path)
        else:
            raise ValueError("eval_type must be 'loss' or 'acc'")

    # [主函数]评估模型
    def evaluate_net(self, eval_type: str = "test") -> float | str:
        """
        评估模型
        :param eval_type: 评估类型，支持"test"和"train"
        :return: 损失或准确率，依据self.net_type而定；在不评估时返回self.NO_EVAL_MSG(默认为'"No eval"')
        """
        # if delete_train:
        #     del self.X_train, self.y_train
        #     torch.cuda.empty_cache()
        # if self.eval_during_training:
        #     self.__original_dataset_to_device()  # 如果要在训练时评估，需要将数据转移到设备上
        # else:
        #     return self.NO_EVAL_MSG  # 不在训练时评估
        if not self.eval_during_training:
            return self.NO_EVAL_MSG  # 不在训练时评估
        self.net.eval()  # 确保评估时不使用dropout等
        with torch.no_grad():  # 在评估时禁用梯度计算，节省内存
            if self.eval_type == "loss":
                if eval_type == "test":
                    loss = self._cal_fnn_loss(data_type="test")
                else:
                    # 事实上一般不调用这个，因为训练集的loss在训练时已经计算了
                    loss = self._cal_fnn_loss(data_type="train")
                return loss
            elif self.eval_type == "acc":
                if eval_type == "test":
                    acc = self._cal_fnn_acc(data_type="test")
                    # predictions = torch.argmax(self.net(self.X_test), dim=1).type(self.y_test.dtype)
                    # correct = (predictions == self.y_test).sum().item()
                    # n = self.y_test.numel()
                    # acc = correct / n
                else:
                    acc = self._cal_fnn_acc(data_type="train")
                    # predictions = torch.argmax(self.net(self.X_train), dim=1).type(self.y_train.dtype)
                    # correct = (predictions == self.y_train).sum().item()
                    # n = self.y_train.numel()
                    # acc = correct / n
                return acc
            else:
                raise ValueError("eval_type must be 'loss' or 'acc'")
        # total, correct = 0, 0
        # with torch.no_grad():
        #     for inputs, labels in self.test_loader:
        #         inputs, labels = inputs.to(self.device), labels.to(self.device)
        #         outputs = self.net(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        # print(f'Accuracy: {100 * correct / total}%')

    # [主函数]查看模型参数。使用Netron(需要安装)可视化更好，这里只是简单的查看
    def view_parameters(self, view_net_struct=False, view_params_count=True, view_params_details=False):
        """
        例如：
        Conv2d的Weight输出为：[32, 1, 3, 3]，表示out_channels=32, in_channels=1, kernel_h=3, kernel_w=3，参数量为32*1*3*3=288。
        Conv2d的Bias输出为：[32]，表示out_channels=32，偏置参数量为32。
        Linear的Weight[128, 3136] 表示out_features=128, in_features=3136，参数数=128*3136=401408。（不难看出这是参数量最大的层）
        Linear的Bias[128]，表示out_features=128，偏置参数量为128。
        :param view_net_struct:
        :param view_params_count:
        :param view_params_details:
        :return:
        """
        # if view_layers:
        #     for layer in self.net.children():
        #         print(layer)
        if view_net_struct:
            print("网络结构如下：")
            print(self.net)
        if view_params_count:
            # 第一步：计算出总参数量，用于后续计算百分比
            total_count = sum(p.numel() for p in self.net.parameters())
            current_count = 0
            # 第二步：使用 named_parameters() 获取名字（name）和参数（p）
            for name, p in self.net.named_parameters():
                numel = p.numel()  # 当前这组参数的总量
                percentage = (numel / total_count) * 100  # 计算占比
                if view_params_details:
                    # 这里的 name 会输出类似 "conv1.weight", "conv1.bias" 等
                    print(f"该层名称: {name:<15} | 形状: {str(list(p.size())):<16} | "
                          f"参数量: {numel:<8} | 占比: {percentage:>5.2f}%")
                current_count += numel
            print(f"网络的总参数量: {total_count}")
        # if view_params_count:
        #     count = 0
        #     for p in self.net.parameters():
        #         if view_params_details:
        #             print("该层的参数：" + str(list(p.size())))
        #         count += p.numel()
        #     print(f"网络的总参数量: {count}")
        # print(f"Total params: {sum(p.numel() for p in self.net.parameters())}")

        # params = list(self.net.parameters())
        # k = 0
        # for i in params:
        #     l = 1
        #     print(f"该层的名称：{i.size()}")
        #     print("该层的结构：" + str(list(i.size())))
        #     for j in i.size():
        #         l *= j
        #     print("该层参数和：" + str(l))
        #     k = k + l
        # print("总参数数量和：" + str(k))

    # [子函数]评估FNN的loss
    def _cal_fnn_loss(self, data_type="test"):
        """计算指定数据集的平均损失。"""
        self.net.eval()  # 设置模型为评估模式

        # 根据数据类型选择对应的数据加载器
        loader = self.train_loader if data_type == "train" else self.test_loader

        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():  # 禁用梯度计算
            for inputs, targets in loader:
                # 将数据移动到模型所在设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播计算输出和损失
                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, targets)

                # 累加损失和样本数
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

        # 返回平均损失
        return total_loss / num_samples if num_samples > 0 else 0.0

    def _cal_fnn_acc(self, data_type="test"):
        """计算指定数据集的准确率。"""
        self.net.eval()  # 设置模型为评估模式
        loader = self.train_loader if data_type == "train" else self.test_loader  # 根据数据类型选择对应的数据加载器
        correct_predictions = 0
        num_samples = 0

        with torch.no_grad():  # 禁用梯度计算
            for inputs, targets in loader:
                # 将数据移动到模型所在设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # 前向传播计算输出并获取预测类别
                outputs = self.net(inputs)
                # 根据输出 shape 判断问题类型
                if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # 二分类问题
                    # 对 logits 应用 sigmoid 并将概率转化为类别标签
                    predictions = (torch.sigmoid(outputs).view(-1) > 0.5).long()
                else:  # 多分类
                    # 获取预测类别
                    predictions = torch.argmax(outputs, dim=1)
                # predictions = torch.argmax(outputs, dim=1)  # 获取预测类别
                # print(f"shape: {predictions.shape} --- {targets.shape}")
                # # print(predictions)
                # # print(targets)
                # # exit(1)

                # 统计正确预测的样本数和总样本数
                correct_predictions += (predictions == targets).sum().item()
                # print(f"shape: {predictions.shape} --- {targets.shape}")
                num_samples += targets.size(0)
                # print(targets.size(0))
                # print(f"correct_predictions={correct_predictions}, num_samples={num_samples}")

        # 返回准确率
        # print(f">> correct_predictions={correct_predictions}, num_samples={num_samples}")
        return correct_predictions / num_samples

    # def _cal_fnn_loss(self, data_type="test"):
    #     self.net.eval()
    #     total_loss = 0.0
    #
    #     with torch.no_grad():
    #         for i in range(0, len(x), self.batch_size):
    #             X_batch = x[i:i + self.batch_size].to(self.device)
    #             y_batch = y[i:i + self.batch_size].to(self.device)
    #             if len(X_batch) == 0:
    #                 warnings.warn(f"[_cal_fn_loss]最后一个batch的长度为0，理论上不会出现这个情况吧")
    #                 continue
    #             outputs = net(X_batch)
    #             loss = criterion(outputs, y_batch)
    #             total_loss += loss.item() * y_batch.size(0)
    #             del X_batch, y_batch, outputs, loss
    #             torch.cuda.empty_cache()
    #
    #     average_loss = total_loss / len(x)
    #     return average_loss
    #
    # # [子函数]评估FNN的acc
    # def _cal_fnn_acc(self, data_type="test"):
    #     net.eval()
    #     correct = 0
    #     total = 0
    #
    #     with torch.no_grad():
    #         for i in range(0, len(x), self.batch_size):
    #             X_batch = x[i:i + self.batch_size].to(self.device)
    #             y_batch = y[i:i + self.batch_size].to(self.device)
    #             outputs = net(X_batch)
    #             predictions = torch.argmax(outputs, dim=1)
    #             # print(X_batch.shape, y_batch.shape, outputs.shape, predictions.shape)
    #             correct += (predictions == y_batch).sum().item()
    #             total += y_batch.size(0)
    #             del X_batch, y_batch, outputs, predictions
    #             torch.cuda.empty_cache()
    #     # print(f"correct={correct}, total={total}")
    #     accuracy = correct / total
    #     return accuracy

    # [log函数]打印GPU显存
    def _log_gpu_memory(self):
        if not self.eval_during_training:
            return self.NO_EVAL_MSG  # 不在训练时评估
        log = None

        # 获取self.device的设备索引
        self_device_index = None
        # 如果是cuda
        if self.device.type == "cuda":
            self_device_index = self.device.index

        # 获取当前设备索引
        current_device_index = torch.cuda.current_device()
        if current_device_index is None:
            log = "当前没有GPU设备"
            return log
        elif self_device_index is not None and current_device_index != self_device_index:
            warnings.warn(f"[_log_gpu_memory]当前设备为{current_device_index}，与{self_device_index}不一致")
        else:
            log = ""

        props = torch.cuda.get_device_properties(current_device_index)  # 获取设备属性
        used_memory = torch.cuda.memory_allocated(current_device_index)  # 已用显存（字节）
        reserved_memory = torch.cuda.memory_reserved(current_device_index)  # 保留显存（字节）
        total_memory = props.total_memory  # 总显存（字节）
        used_memory_gb = used_memory / (1024 ** 3)  # 已用显存（GB）
        reserved_memory_gb = reserved_memory / (1024 ** 3)  # 保留显存（GB）
        total_memory_gb = total_memory / (1024 ** 3)  # 总显存（GB）
        # U: used已用, R: reserved保留, T: total总
        log += (f"设备{current_device_index}："
                f"U{used_memory_gb:.2f}+R{reserved_memory_gb:.2f}/T{total_memory_gb:.2f}GB")

        return log

    # @staticmethod
    # def _dataframe_to_tensor(df, float_dtype=torch.float16, int_dtype=torch.int64):
    #     """
    #     PyTorch's tensors are homogenous, ie, each of the elements are of the same type.
    #     将df转换为tensor，并保持数据类型的一致性
    #     :param df: pd.DataFrame
    #     :param float_dtype: torch.dtype, default=torch.float32
    #     :param int_dtype: torch.dtype, default=torch.int32
    #     :return: torch.Tensor
    #     """
    #     # 先判断df是不是dataframe
    #     if not isinstance(df, pd.DataFrame):
    #         if isinstance(df, torch.Tensor):
    #             return df
    #         elif isinstance(df, np.ndarray):
    #             return torch.tensor(df)
    #         else:
    #             raise ValueError("既不是dataframe又不是tensor")
    #     # 检查df中的数据类型
    #     dtypes = []
    #     for col in df.column:
    #         if pd.api.types.is_float_dtype(df[col]):
    #             dtypes.append(float_dtype)
    #         elif pd.api.types.is_integer_dtype(df[col]):
    #             dtypes.append(int_dtype)
    #         else:
    #             raise ValueError(f"[_dataframe_to_tensor]Unsupported data type in column {col}: {df[col].dtype}")
    #     # print(dtypes)
    #     # 将df中的每一列转换为tensor
    #     # 对于多维的data
    #     if len(dtypes) > 1:
    #         tensors = [torch.as_tensor(df[col].values, dtype=dtype) for col, dtype in zip(df.columns, dtypes)]
    #         return torch.stack(tensors, dim=1)  # 使用torch.stack将多个tensor堆叠在一起
    #     # 对于一维的target
    #     elif len(dtypes) == 1:
    #         return torch.as_tensor(df.values, dtype=dtypes[0])
    #     else:
    #         raise ValueError(f"[_dataframe_to_tensor]数据长度有误{len(dtypes)}")
    #
    # @staticmethod
    # def _dataloader_to_tensor(dataloader):
    #     data_list = []
    #     target_list = []
    #     for data, target in dataloader:
    #         data_list.append(data)
    #         target_list.append(target)
    #     return torch.cat(data_list), torch.cat(target_list)
    # #
    # # # 将y的维度转换为1维
    # # def _target_reshape_1D(self, y):
    # #     """
    # #     将y的维度转换为1维
    # #     :param y: torch.Tensor
    # #     :return: torch.Tensor
    # #     """
    # #     if self.target_reshape_1D and self.eval_type == "acc" and y.dim() > 1:
    # #         warnings.warn(f"[_target_reshape_1D]请注意：y的维度为{y.dim()}: {y.shape}，将被自动转换为1维\n"
    # #                       "如需保持原有维度，请设置 target_reshape_1D=False ")
    # #         return y.view(-1)
    # #     else:
    # #         return y

    # 检查模型路径是否合法，该函数仅在train_net中调用
    def __check_best_net_save_path(self, net_save_path):
        if isinstance(net_save_path, str):
            if not self.eval_during_training:
                self.auto_save_best_net = False
                warnings.warn("net_save_path参数在eval_during_training=False时无效，auto_save_best_net设为False")
            else:
                dir_path = os.path.dirname(net_save_path)
                self.auto_save_best_net = True
                if os.path.exists(dir_path):
                    print(f"[train_net] 最佳模型保存地址net_save_path={net_save_path}")
                else:
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                        warnings.warn(f"[train_net] 最佳模型保存文件夹dir_path='{dir_path}'不存在，已自动创建")
                    except Exception as e:
                        warnings.warn(f"[train_net] 最佳模型保存文件夹dir_path='{dir_path}'创建失败，错误信息：{e}")
                if not net_save_path.endswith(".pth"):
                    print(f"[train_net] 请注意net_save_path='{net_save_path}'未以'.pth'结尾")
        elif net_save_path is None:
            self.auto_save_best_net = False
            print("[train_net] 未设置net_save_path，auto_save_best_net已设置为False")
        else:
            self.auto_save_best_net = False
            warnings.warn(f"[train_net] 请注意net_save_path={net_save_path}不是str类型，auto_save_best_net已设置为False")

    # 保存模型
    def __save_net(self, net_save_path, save_type="net"):
        try:
            if save_type == "net":
                torch.save(self.net, net_save_path)
            elif save_type == "state_dict":
                torch.save(self.net.state_dict(), net_save_path)  # 暂不支持
            else:
                raise ValueError(f"> [train_net] 保存类型save_type={save_type}不合法")
            # print(f"[train_net] 已保存{save_type}模型到{net_save_path}")
        except Exception as e:
            warnings.warn(f"> [train_net] 保存模型失败，错误信息：{e}")
    # 将原始数据转移到设备上，暂被弃用
    # def __original_dataset_to_device(self):
    #     # 暂时不知道只使用self.original_dataset_to_device是否会有问题，或许可以直接检查self.X_train.device(有问题再改吧)
    #     if not self.original_dataset_to_device:
    #         # 将数据转移到设备上
    #         self.X_train, self.X_test, self.y_train, self.y_test = self.X_train.to(self.device), self.X_test.to(
    #             self.device), self.y_train.to(self.device), self.y_test.to(self.device)
    #         self.original_dataset_to_device = True


class NetTrainerPair(NetTrainerFNN):
    """
    专门用于训练孪生网络（双输入x1,x2）的类，继承自 NetTrainerFNN
    """

    def log_X_y(self):
        """重写 log 逻辑"""
        # 获取样本总量
        train_samples = len(self.train_loader.dataset)
        test_samples = len(self.test_loader.dataset) if self.test_loader else 0
        print(f"[log] 数据总量：Train: {train_samples}, Test: {test_samples}")

        # 打印第一个 batch 的 x1.shape 和 x2.shape
        x1, x2, y = next(iter(self.train_loader))
        x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
        outputs1, outputs2 = self.net(x1, x2)
        print(f"[log] 第一批次的shape如下：\n"
              f"      x1: {x1.shape}, x2: {x2.shape}, y: {y.shape}, outputs1: {outputs1.shape}, outputs2: {outputs2.shape}")
        print("[log] ", end='')
        self.view_parameters(view_net_struct=False, view_params_count=True, view_params_details=False)

    def train_epoch(self):
        start_time = time.time()
        loss_sum = 0.0

        for x1, x2, y in self.train_loader:
            # train_loader返回[x1, x2, y]
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp, device_type=self.device.type, dtype=torch.float16):
                outputs1, outputs2 = self.net(x1, x2)
                loss = self.loss_fn(outputs1, outputs2, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += loss.item()
            self.current_gpu_memory = self._log_gpu_memory()
            if self.free_memory:
                del x1, x2, y, outputs1, outputs2, loss
        if self.scheduler is not None:
            self.scheduler.step()
        self.loss_epoch = loss_sum / len(self.train_loader)
        if torch.isnan(torch.tensor(self.loss_epoch)):
            warnings.warn("请注意Loss是NaN!", RuntimeWarning)
        self.time_list.append(time.time() - start_time)

    def _cal_fnn_loss(self, data_type="test"):
        self.net.eval()
        loader = self.train_loader if data_type == "train" else self.test_loader
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for x1, x2, targets in loader:
                x1, x2, targets = x1.to(self.device), x2.to(self.device), targets.to(self.device)
                # 双输入前向传播
                outputs1, outputs2 = self.net(x1, x2)
                loss = self.loss_fn(outputs1, outputs2, targets)
                batch_size = x1.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size
        return total_loss / num_samples

    def _cal_fnn_acc(self, data_type="test"):
        self.net.eval()
        loader = self.train_loader if data_type == "train" else self.test_loader
        correct_predictions = 0
        num_samples = 0

        with torch.no_grad():
            for x1, x2, targets in loader:
                x1, x2, targets = x1.to(self.device), x2.to(self.device), targets.to(self.device)
                # 双输入前向传播
                outputs1, outputs2 = self.net(x1, x2)
                dist = F.pairwise_distance(outputs1, outputs2)
                preds = (dist < 0.5).long()  # 假设阈值为0.5
                correct_predictions += (preds == targets).sum().item()
                num_samples += targets.size(0)
        return correct_predictions / num_samples


class NetTrainerArcFace(NetTrainerFNN):
    """
    扩展自 NetTrainerFNN，支持 ArcFace 风格的训练流程：
      net(X) -> embedding [B, D] (L2 normalized)
      arcface(embeddings, labels) -> (loss, logits)
    主要覆盖点：
      - train_epoch：处理 (loss, logits) 返回，记录 top1
      - _cal_fnn_loss/_cal_fnn_acc：使用 arcface.get_logits 计算评估指标
      - log_X_y：显示 shapes 时兼容 embedding + logits
    """

    def log_X_y(self):
        train_samples = len(self.train_loader.dataset)
        test_samples = len(self.test_loader.dataset) if self.test_loader else 0
        print(f"[log] 数据总量：Train: {train_samples}, Test: {test_samples}")

        X, y = next(iter(self.train_loader))
        X, y = X.to(self.device), y.to(self.device)
        outputs = self.net(X)  # [B, D] normalized
        loss, logits = self.loss_fn(outputs, y)
        print(f"[log] 第一批次的shape如下：\n"
              f"      X: {X.shape}, y: {y.shape}, outputs: {outputs.shape}, "
              f"      loss: {loss.shape if isinstance(loss, torch.Tensor) else type(loss)}, logits: {logits.shape}")
        print("[log] ", end='')
        self.view_parameters(view_net_struct=False, view_params_count=True, view_params_details=False)

    def train_epoch(self):
        start_time = time.time()
        loss_sum = 0.0
        # total = 0
        # correct = 0

        self.net.train()
        self.loss_fn.train()

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device).long()
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp, device_type=self.device.type, dtype=torch.float16):
                outputs = self.net(X)  # [B, D], should be normalized inside model
                loss, logits = self.loss_fn(outputs, y)  # expect (loss, logits)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += loss.item()
            # batch_size = X.size(0)
            # loss_sum += loss.item() * batch_size
            # total += batch_size

            # # compute top1 from logits as training accuracy proxy
            # preds = logits.argmax(dim=1)
            # correct += (preds == y).sum().item()

            if self.free_memory:
                del X, y, outputs, logits, loss
        if self.scheduler is not None:
            self.scheduler.step()
        self.loss_epoch = loss_sum / len(self.train_loader)
        if torch.isnan(torch.tensor(self.loss_epoch)):
            warnings.warn("请注意Loss是NaN!", RuntimeWarning)
        self.time_list.append(time.time() - start_time)

    def _cal_fnn_loss(self, data_type="test"):
        self.net.eval()
        self.loss_fn.eval()
        loader = self.train_loader if data_type == "train" else self.test_loader
        total_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device).long()
                outputs = self.net(X)
                loss, _ = self.loss_fn(outputs, y)
                batch_size = X.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size
        return total_loss / num_samples if num_samples > 0 else 0.0

    def _cal_fnn_acc(self, data_type="test"):
        self.net.eval()
        self.loss_fn.eval()
        loader = self.train_loader if data_type == "train" else self.test_loader
        correct_predictions = 0
        num_samples = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device).long()
                outputs = self.net(X)
                _, logits = self.loss_fn(outputs, y)
                preds = logits.argmax(dim=1)
                # if data_type == "train":
                #     print("Train batch")
                #     print(preds)
                #     print(y)
                # if data_type == "test":
                #     print("Test batch")
                #     print(preds)
                #     print(y)
                correct_predictions += (preds == y).sum().item()
                num_samples += X.size(0)
        return correct_predictions / num_samples if num_samples > 0 else 0.0
