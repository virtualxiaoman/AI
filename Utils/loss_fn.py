import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        label: 标签, 1 代表同类, 0 代表异类(与论文的定义相反)，因为_cal_fnn_acc中实现的是dist < 0.5则为同类(值为1)
        $$L(W, Y, X_1, X_2) =Y * 0.5 * (D_W)^2 + (1-Y) * 0.5 * {max(0, m-D_W)}^2$$
        """
        # 类内损失
        euclidean_distance = F.pairwise_distance(output1, output2)
        within_loss = label * torch.pow(euclidean_distance, 2) * 0.5

        # 类间损失
        between_loss = (1 - label) * F.relu(self.margin - euclidean_distance).pow(2) * 0.5

        loss_contrastive = torch.mean(within_loss + between_loss)
        return loss_contrastive


# class ArcFace(nn.Module):
#     r"""Implement of large margin arc distance: :
#         Args:
#             in_features: size of each input sample
#             out_features: size of each output sample (通常是类别数 num_classes)
#             s: norm of input feature (scale factor)
#             m: margin
#             easy_margin: optimize trick
#     """
#
#     def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
#         super(ArcFace, self).__init__()
#         self.in_features = in_features  # embedding_dim
#         self.out_features = out_features  # num_classes
#         self.s = s
#         self.m = m
#         # 权重 W，代表每一类的中心
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#
#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#
#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         # 1. 归一化输入特征 x 和 权重 W
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#
#         # 2. 计算 phi = cos(theta + m)
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#
#         # --------------------------- convert label to one-hot ---------------------------
#         # ArcFace 的核心：只对 Ground Truth 对应的那个类别加 margin
#         one_hot = torch.zeros(cosine.size(), device=input.device)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#
#         # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         # 上面这行等价于：如果是目标类，用 phi (margin后的值)，否则用原 cosine
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.s
#
#         return output


class ArcFaceLoss(nn.Module):
    """
    shape中B=batch_size,C=num_classes,D=embedding_dim
    ArcFace head + loss (additive angular margin).
    Usage:
      arcface = ArcFace(num_classes, embedding_dim, s=30.0, m=0.5).to(device)
      embeddings = model(inputs)            # embeddings must NOT be normalized here OR you can normalize inside
      embeddings = F.normalize(embeddings, p=2, dim=1)
      loss, logits = arcface(embeddings, labels)
    Forward returns (loss, logits). You can also call arcface.get_logits(embeddings) to obtain logits for eval.
    """

    def __init__(self, num_classes: int, embedding_dim: int, s: float = 30.0, m: float = 0.5,
                 easy_margin: bool = False, eps: float = 1e-7):
        super().__init__()
        self.num_classes = int(num_classes)  # out_features
        self.embedding_dim = int(embedding_dim)  # in_features
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)
        self.eps = float(eps)

        # precompute constants
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)  # threshold
        self.mm = math.sin(math.pi - self.m) * self.m  # margin compensation (paper trick)

        # class weights (proxies). Shape [C, D]
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.LongTensor):
        """
        embeddings: [B, D] -- expected to be L2-normalized (F.normalize(..., dim=1))
        labels: [B] (long)
        returns: (loss (scalar), logits [B, C])
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be shape [B, D]")
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(f"embeddings dim ({embeddings.size(1)}) != embedding_dim ({self.embedding_dim})")

        device = embeddings.device
        labels = labels.to(device)

        W = F.normalize(self.weight, p=2, dim=1)  # [C, D]，L2归一化，保证内积直接等于余弦值

        cos_theta = torch.matmul(embeddings, W.t())  # cos𝜃_j: [B, C]
        cos_theta = cos_theta.clamp(-1.0 + self.eps, 1.0 - self.eps)  # for numerical stability
        sin_theta = torch.clamp(1.0 - cos_theta.pow(2), min=0.0, max=1.0)
        sin_theta = torch.sqrt(sin_theta)  # sin𝜃_i = sqrt(1 - cos^2(𝜃_i))
        # 𝜙 = cos(𝜃+m) = cos𝜃cosm − sin𝜃sinm，对正确类要用𝜙代替cos𝜃
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            phi = torch.where(cos_theta > 0.0, phi, cos_theta)  # easier: if cos𝜃>0(𝜃<90°)use 𝜙, else use cos𝜃
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)  # cos𝜃>cos(π-m) 则用𝜙

        one_hot = torch.zeros_like(cos_theta, device=device)  # [B, C]
        # labels: [B, 1]，labels[i]就是第i个样本的正确类别索引。
        # scatter_是对每一行i，在第labels[i]列，把值设为1.0，表示正确类(j=y_i)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # 在one_hot=1的位置用𝜙=cos(𝜃+m), 其余位置用cos𝜃
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logits *= self.s

        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def get_logits(self, embeddings: torch.Tensor):
        """
        Return logits only (for inference/eval). embeddings expected normalized.
        """
        W = F.normalize(self.weight, p=2, dim=1)
        cos_theta = torch.matmul(embeddings, W.t())
        cos_theta = cos_theta.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # This returns the *baseline* cosine logits scaled by s.
        # Note: evaluation often compares embeddings by cosine directly, not logits.
        return cos_theta * self.s
