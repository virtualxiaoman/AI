"""
Attention Is All You Need

Transformer Base:
    N = 6
    d_model = 512
    d_ff = 2048
    h = 8
    d_k = d_v = 64
    dropout = 0.1

原论文结构：
    Encoder:
        Self-Attention -> Add & Norm -> FFN -> Add & Norm
    Decoder:
        Masked Self-Attention -> Add & Norm
        -> Encoder-Decoder Attention -> Add & Norm
        -> FFN -> Add & Norm

采用 Post-LN：
    LayerNorm(x + Dropout(Sublayer(x)))
而不是现代实现常见的 Pre-LN：
    x + Dropout(Sublayer(LayerNorm(x)))

原论文还采用：
    1. sinusoidal positional encoding
    2. embedding * sqrt(d_model)
    3. source/target embedding 与 pre-softmax projection 权重共享
    4. label smoothing = 0.1
    5. Adam(beta1=0.9, beta2=0.98, eps=1e-9)
    6. warmup = 4000 的 inverse-square-root learning rate
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Scaled Dot-Product Attention
# ============================================================

def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention:

        $$Attention(Q,K,V)
        =
        softmax
        \\left(
            \\frac{QK^T}{\\sqrt{d_k}}
        \\right)V$$

    输入：
        Q: [B, H, Lq, d_k]
        K: [B, H, Lk, d_k]
        V: [B, H, Lk, d_v]

    输出：
        output:  [B, H, Lq, d_v]
        p_attn:  [B, H, Lq, Lk]

    其中：
        B  = batch size
        H  = number of heads
        Lq = query length
        Lk = key/value length
        d_k = key/query dimension
        d_v = value dimension
    """

    # QK^T:
    # [B,H,Lq,d_k] @ [B,H,d_k,Lk] -> [B,H,Lq,Lk]
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # mask == 0 的位置设为 -inf：
    #
    # softmax(-inf) = 0
    #
    # 因此这些位置不会参与 attention。
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # $$P=softmax(\\frac{QK^T}{\\sqrt{d_k}})$$
    p_attn = F.softmax(scores, dim=-1)

    # 原论文对 attention weights 使用 dropout。
    if dropout is not None:
        p_attn = dropout(p_attn)

    # $$Attention(Q,K,V)=PV$$
    # [B,H,Lq,Lk] @ [B,H,Lk,d_v] -> [B,H,Lq,d_v]
    return torch.matmul(p_attn, value), p_attn


# ============================================================
# 2. Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention:

        $$MultiHead(Q,K,V)
        =
        Concat(head_1,...,head_h)W^O$$

    其中：

        $$head_i =
        Attention(QW_i^Q,KW_i^K,VW_i^V)$$

    原论文 Base：
        d_model = 512
        h = 8
        d_k = d_v = 512 / 8 = 64
    """

    def __init__(self, d_model=512, h=8, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        # 输入：
        #     query: [B,Lq,d_model]
        #     key:   [B,Lk,d_model]
        #     value: [B,Lk,d_model]
        #
        # mask:
        #     [B,Lq,Lk] -> [B,1,Lq,Lk]
        #     这样可以广播到所有 heads。
        if mask is not None:
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask

        batch_size = query.size(0)

        # 线性投影：
        #
        # $$Q=XW^Q$$
        # $$K=XW^K$$
        # $$V=XW^V$$
        #
        # [B,L,d_model] -> [B,L,d_model]
        query, key, value = [layer(x) for layer, x in zip(self.linear[:3], (query, key, value))]

        # 拆分多头：
        #
        # [B,L,512]
        #     -> [B,L,8,64]
        #     -> [B,8,L,64]
        #
        # 即：
        #     [B,H,L,d_k]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]

        # 每个 head 独立执行：
        #
        # $$head_i=Attention(Q_i,K_i,V_i)$$
        x, self.attn = attention(query, key, value, mask, self.dropout)

        # 拼接多头：
        #
        # [B,H,L,d_k]
        #     -> [B,L,H,d_k]
        #     -> [B,L,d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性投影：
        #
        # $$MultiHead=Concat(head_1,...,head_h)W^O$$
        return self.linear[3](x)


# ============================================================
# 3. Position-wise Feed-Forward Network
# ============================================================

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise FFN：

        $$FFN(x)=max(0,xW_1+b_1)W_2+b_2$$

    原论文：
        d_model = 512
        d_ff = 2048

    所以：
        512 -> 2048 -> 512

    FFN 对每个 position 独立执行，但所有 position 共享参数。
    """

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # $$xW_1+b_1$$
        x = self.w_1(x)

        # $$ReLU(x)=max(0,x)$$
        x = F.relu(x)

        # 原论文使用 dropout。
        x = self.dropout(x)

        # $$(ReLU(xW_1+b_1))W_2+b_2$$
        return self.w_2(x)


# ============================================================
# 4. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding。

    原论文：

        $$PE_{(pos,2i)}
        =
        \\sin
        \\left(
            \\frac{pos}{10000^{2i/d_{model}}}
        \\right)$$

        $$PE_{(pos,2i+1)}
        =
        \\cos
        \\left(
            \\frac{pos}{10000^{2i/d_{model}}}
        \\right)$$

    输入 embedding 先乘：

        $$\\sqrt{d_{model}}$$

    然后：

        $$X = Embedding(x)\\sqrt{d_{model}} + PE$$

    最后对这个和使用 dropout。
    """

    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # PE:
        # [max_len,d_model]
        pe = torch.zeros(max_len, d_model)

        # position:
        # [max_len,1]
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        # 对应：
        #
        # $$10000^{-2i/d_{model}}$$
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # 偶数维使用 sin：
        #
        # $$PE_{(pos,2i)}=sin(pos/10000^{2i/d_model})$$
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维使用 cos：
        #
        # $$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_model})$$
        pe[:, 1::2] = torch.cos(position * div_term)

        # [max_len,d_model] -> [1,max_len,d_model]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # $$X=Embedding(x)\\sqrt{d_{model}}+PE$$
        x = x + self.pe[:, :x.size(1)]

        # 原论文对 embedding + PE 的和使用 dropout。
        return self.dropout(x)


# ============================================================
# 5. Embedding
# ============================================================

class Embeddings(nn.Module):
    """
    Token Embedding。

    原论文不是直接使用 embedding，而是：

        $$Embedding(x)=E[x]\\sqrt{d_{model}}$$
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ============================================================
# 6. Sublayer Connection
# ============================================================

class SublayerConnection(nn.Module):
    """
    原论文使用 Post-LN：

        $$LayerNorm(
            x + Dropout(Sublayer(x))
        )$$

    顺序必须是：

        Sublayer
            -> Dropout
            -> Residual Add
            -> LayerNorm

    不能改成 Pre-LN。
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # $$Sublayer(x)$$
        y = sublayer(x)

        # $$Dropout(Sublayer(x))$$
        y = self.dropout(y)

        # $$x+Dropout(Sublayer(x))$$
        y = x + y

        # $$LayerNorm(x+Dropout(Sublayer(x)))$$
        return self.norm(y)


# ============================================================
# 7. Encoder Layer
# ============================================================

class EncoderLayer(nn.Module):
    """
    一个 Encoder Layer：

        1. Multi-Head Self-Attention
        2. Position-wise FFN

    即：

        x -> Self-Attention -> Add & Norm
          -> FFN -> Add & Norm
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        # Encoder Self-Attention：
        #
        # $$Q=XW^Q,\quad K=XW^K,\quad V=XW^V$$
        #
        # 所以：
        #
        # $$Attention(X,X,X)$$
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # $$LayerNorm(x+Dropout(FFN(x)))$$
        return self.sublayer[1](x, self.feed_forward)


# ============================================================
# 8. Encoder
# ============================================================

class Encoder(nn.Module):
    """
    Encoder 由 N 个独立参数的 Encoder Layer 堆叠。

    原论文 Base：
        $$N=6$$
    """

    def __init__(self, layer, N):
        super().__init__()

        # deepcopy 保证每层拥有独立参数。
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

        # Encoder 最终 LayerNorm。
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ============================================================
# 9. Decoder Layer
# ============================================================

class DecoderLayer(nn.Module):
    """
    一个 Decoder Layer：

        1. Masked Multi-Head Self-Attention
        2. Encoder-Decoder Attention
        3. Position-wise FFN

    其中第二个 attention：

        $$Q=Y_dW^Q$$
        $$K=MemoryW^K$$
        $$V=MemoryW^V$$

    即：

        Attention(Decoder, Encoder, Encoder)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        # ----------------------------------------------------
        # 1. Masked Self-Attention
        #
        # Q = K = V = Decoder 当前 hidden states
        #
        # 使用 tgt_mask 防止看到未来 token。
        # ----------------------------------------------------
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # ----------------------------------------------------
        # 2. Encoder-Decoder Attention
        #
        # Q 来自 Decoder。
        # K、V 来自 Encoder。
        #
        # $$Attention(Q_{decoder},K_{encoder},V_{encoder})$$
        # ----------------------------------------------------
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))

        # ----------------------------------------------------
        # 3. Feed-Forward Network
        # ----------------------------------------------------
        return self.sublayer[2](x, self.feed_forward)


# ============================================================
# 10. Decoder
# ============================================================

class Decoder(nn.Module):
    """
    Decoder 由 N 个独立参数的 Decoder Layer 堆叠。

    原论文 Base：
        $$N=6$$
    """

    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


# ============================================================
# 11. Generator
# ============================================================

class Generator(nn.Module):
    """
    Decoder 输出映射到 vocabulary。

    $$Z=YW^T+b$$

    $$P(y|x)=softmax(Z)$$

    这里返回 log_softmax，便于训练时使用 NLL/KL 类 loss。
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ============================================================
# 12. Transformer
# ============================================================

class Transformer(nn.Module):
    """
    完整 Transformer：

        Source
          -> Embedding * sqrt(d_model)
          -> Positional Encoding
          -> Encoder
          -> Memory
          -> Decoder
          -> Generator
          -> Vocabulary distribution
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        # $$Memory=Encoder(Embedding(src)+PE)$$
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # $$Y=Decoder(Embedding(tgt)+PE,Memory)$$
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # $$Memory=Encoder(src)$$
        memory = self.encode(src, src_mask)

        # $$P=Generator(Decoder(tgt,Memory))$$
        return self.generator(self.decode(memory, src_mask, tgt, tgt_mask))


# ============================================================
# 13. Build Transformer
# ============================================================

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    创建原论文 Transformer Base。

    默认：
        N = 6
        d_model = 512
        d_ff = 2048
        h = 8
        dropout = 0.1

    注意：
        原论文明确规定上述架构超参数；
        原论文正文没有明确规定某个特定参数初始化公式，
        因此这里不额外强制使用 Xavier/Kaiming 初始化。
    """

    c = copy.deepcopy

    # 一组 Attention 仅作为模板，后续 deepcopy 后参数彼此独立。
    attn = MultiHeadAttention(d_model, h, dropout)

    # 一组 FFN 作为模板。
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # Positional Encoding 也需要复制到 source/target。
    position = PositionalEncoding(d_model, dropout)

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # 原论文：
    #
    # source embedding、target embedding、
    # pre-softmax linear transformation
    # 使用同一个 weight matrix。
    #
    # 这里要求 source / target vocabulary 相同。
    if src_vocab == tgt_vocab:
        model.tgt_embed[0].lut.weight = model.src_embed[0].lut.weight
        model.generator.proj.weight = model.tgt_embed[0].lut.weight

    return model


# ============================================================
# 14. Causal Mask
# ============================================================

def subsequent_mask(size, device=None):
    """
    Decoder 的 subsequent mask：

        $$M_{ij} =
        \\begin{cases}
        1, & j \\le i \\
        0, & j > i
        \\end{cases}$$

    例如：

        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1

    第 i 个位置只能看到当前位置及之前的位置。
    """

    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0)


# ============================================================
# 15. Source Padding Mask
# ============================================================

def make_src_mask(src, pad=0):
    """
    Padding mask：

        src = [token_1, token_2, PAD, PAD]

        mask = [1, 1, 0, 0]

    shape：
        [B,1,S]
    """

    return (src != pad).unsqueeze(-2)


# ============================================================
# 16. Target Mask
# ============================================================

def make_tgt_mask(tgt, pad=0):
    """
    Target mask 同时包含：

        1. padding mask
        2. causal mask

    即：

        $$M=M_{padding}\\land M_{causal}$$
    """

    # [B,1,T]
    tgt_mask = make_src_mask(tgt, pad)

    # [1,T,T]
    causal_mask = subsequent_mask(tgt.size(-1), tgt.device)

    # broadcasting：
    # [B,1,T] & [1,T,T] -> [B,T,T]
    return tgt_mask & causal_mask


# ============================================================
# 17. Label Smoothing
# ============================================================

class LabelSmoothing(nn.Module):
    """
    原论文使用：

        $$\\epsilon_{ls}=0.1$$

    对于 vocabulary size V：

        正确类别：
            $$1-\\epsilon$$

        其他非 PAD/非 target 类别：
            $$\\frac{\\epsilon}{V-2}$$

    使用 KL divergence：
        $$KL(P||Q)$$
    """

    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        # x 是 log probability。
        assert x.size(1) == self.size

        # 构造 target distribution。
        true_dist = x.detach().clone()

        # 非目标类别：
        # $$\\frac{\\epsilon}{V-2}$$
        true_dist.fill_(self.smoothing / (self.size - 2))

        # target 类别：
        # $$1-\\epsilon$$
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # PAD 不参与计算。
        true_dist[:, self.padding_idx] = 0
        true_dist[target == self.padding_idx] = 0

        return self.criterion(x, true_dist)


# ============================================================
# 18. Noam Learning Rate
# ============================================================

class NoamOpt:
    """
    原论文学习率：

        $$lrate=
        d_{model}^{-1/2}
        \\min(
            step^{-1/2},
            step\\cdot warmup^{-3/2}
        )$$

    原论文 Base：
        $$d_{model}=512$$
        $$warmup=4000$$
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()

        # 每一步重新设置 learning rate。
        for p in self.optimizer.param_groups:
            p["lr"] = rate

        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        step = self._step if step is None else step
        return self.factor * self.model_size ** -0.5 * min(step ** -0.5, step * self.warmup ** -1.5)


# ============================================================
# 19. Original Paper Optimizer
# ============================================================

def make_optimizer(model, d_model=512, warmup=4000):
    """
    原论文 Adam：

        $$\\beta_1=0.9$$
        $$\\beta_2=0.98$$
        $$\\epsilon=10^{-9}$$

    学习率使用 Noam schedule。
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, 1, warmup, optimizer)
torch.nn.Transformer