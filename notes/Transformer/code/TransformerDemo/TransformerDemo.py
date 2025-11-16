# https://zhuanlan.zhihu.com/p/648127076
import torch              # PyTorch深度学习框架的核心库，提供张量操作和自动求导功能
import torch.nn as nn     # PyTorch的神经网络模块，包含各种层、激活函数等
import numpy as np        # 数值计算库

def pos_sinusoid_embedding(seq_len, d_model):
    """
    生成正弦余弦位置编码
    Position Encoding用于给序列中的每个位置添加位置信息
    """
    # torch.zeros() 创建一个全零张量，形状为(seq_len, d_model)
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        # 偶数维度用sin，奇数维度用cos
        f = torch.sin if i % 2 == 0 else torch.cos
        # torch.arange(0, seq_len) 创建从0到seq_len-1的序列张量 [0,1,2,...,seq_len-1]
        # embeddings[:, i] 表示选择所有行的第i列
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    # .float() 将张量转换为float32类型
    return embeddings.float()

def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    生成长度掩码，用于屏蔽超出实际序列长度的部分
    """
    # torch.ones() 创建全1张量，形状为(batch_size, max_len, max_len)
    # device参数指定张量存储在哪个设备上(CPU或GPU)
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        # attn_mask[i, :, :feat_lens[i]] 表示第i个样本的所有行，前feat_lens[i]列
        # 将有效长度内的位置设为0(不屏蔽)，其余位置保持1(屏蔽)
        attn_mask[i, :, :feat_lens[i]] = 0
    # .to(torch.bool) 将张量类型转换为布尔型
    return attn_mask.to(torch.bool)

def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """
    生成后续位置掩码，用于Decoder中防止看到未来的信息
    Args:
        b: batch-size.
        max_len: the length of the whole seqeunce.
        device: cuda or cpu.
    """
    # torch.triu() 返回矩阵的上三角部分，diagonal=1表示主对角线上方的元素
    # 这样可以屏蔽当前位置之后的所有位置，实现因果掩码
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)     # or .to(torch.uint8)

def get_enc_dec_mask(
    b: int, max_feat_len: int, feat_lens: torch.Tensor, max_label_len: int, device: torch.device
) -> torch.Tensor:
    """
    生成编码器-解码器交叉注意力掩码
    """
    # torch.zeros() 创建全零张量，形状为(batch_size, decoder_seq_len, encoder_seq_len)
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)       # (b, seq_q, seq_k)
    for i in range(b):
        # 将超出编码器实际长度的位置设为1(屏蔽)
        # feat_lens[i]: 表示第i个样本的编码器序列实际长度
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    nn.Module是PyTorch中所有神经网络模块的基类，继承它才能使用PyTorch的功能
    """
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        """
        Args:
            d_k: dimension of key
            d_v: dimension of value
            d_model: dimension of model
            num_heads: number of heads
            p: dropout rate
        """
        # super().__init__() 调用父类nn.Module的初始化方法，这是必须的
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k  # dimension of key
        self.d_v = d_v  # dimension of value
        self.num_heads = num_heads
        # nn.Dropout(p) 创建一个dropout层，用于在训练时随机将p比例的神经元置零，防止过拟合
        self.dropout = nn.Dropout(p)

        # linear projections - 线性投影层
        # nn.Linear(in_features, out_features) 创建全连接层，实现 y = xW^T + b
        self.W_Q = nn.Linear(d_model, d_k*num_heads)    # Query投影：将输入维度d_model转换为d_k*num_heads
        self.W_K = nn.Linear(d_model, d_k*num_heads)    # Key投影
        self.W_V = nn.Linear(d_model, d_v*num_heads)    # Value投影
        self.W_out = nn.Linear(d_v*num_heads, d_model)  # 输出投影：将多头结果合并回d_model维度

        # Weight Initialization - 权重初始化
        # nn.init.normal_() 用正态分布初始化权重，这是一种Xavier初始化的变种
        nn.init.normal_(self.W_Q.weight,mean=0,std=np.sqrt(2.0/(d_model+d_k)))  # 用均值为0，标准差为sqrt(2.0/(d_model+d_k))的正态分布初始化W_Q的权重
        nn.init.normal_(self.W_K.weight,mean=0,std=np.sqrt(2.0/(d_model+d_k)))  # 可以防止权重过大或过小，避免梯度消失或爆炸
        nn.init.normal_(self.W_V.weight,mean=0,std=np.sqrt(2.0/(d_model+d_v)))
        nn.init.normal_(self.W_out.weight,mean=0,std=np.sqrt(2.0/(d_v+d_model)))

    def forward(self, Q, K, V, attn_mask, **kwargs):
        """
        前向传播函数，定义了数据如何在网络中流动
        forward()是nn.Module必须实现的方法
        """
        # 输入张量的形状说明:
        # Q: [batch_size, seq_len, d_model] - Query张量
        # K: [batch_size, seq_len, d_model] - Key张量  
        # V: [batch_size, seq_len, d_model] - Value张量
        # attn_mask: [batch_size, seq_len, seq_len] - 注意力掩码
        # **kwargs: 其他关键字参数

        # .size(dim) 返回张量在指定维度的大小
        N = Q.size(0)              # batch_size
        q_len, k_len = Q.size(1), K.size(1)    # 序列长度
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # multi_head split - 多头分割
        # self.W_Q(Q) 将Q通过线性层变换，形状变为[N, seq_len, d_k*num_heads]
        # .view() 重新reshape张量的形状，-1表示自动计算该维度的大小
        # .transpose(1,2) 交换第1和第2维度，将头数维度提前
        Q = self.W_Q(Q).view(N,-1, num_heads,d_k).transpose(1,2)  # [N, num_heads, seq_len, d_k]
        K = self.W_K(K).view(N,-1, num_heads,d_k).transpose(1,2)  # [N, num_heads, seq_len, d_k]
        V = self.W_V(V).view(N,-1, num_heads,d_v).transpose(1,2)  # [N, num_heads, seq_len, d_v]

        # pre-process mask - 预处理掩码
        if attn_mask is not None:
            # assert 断言，检查掩码形状是否正确，如果不正确会抛出异常
            assert attn_mask.size() == (N, q_len, k_len)
            # .unsqueeze(1) 在第1维度插入一个大小为1的维度
            # .repeat() 沿指定维度重复张量，这里在头数维度复制num_heads次
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads,1,1)  # [N, 1, q_len, k_len] -> [N, num_heads, q_len, k_len]
            attn_mask = attn_mask.bool()  # 转换为布尔类型

        # calculate attention weight - 计算注意力权重
        # torch.matmul() 执行矩阵乘法，Q @ K^T
        # K.transpose(-1,-2) 交换K的最后两个维度，即转置操作
        scores = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(d_k)  # 缩放点积注意力
        if attn_mask is not None:
            # .masked_fill_() 将掩码为True的位置填充为指定值（这里是-1e4，一个很小的负数）
            scores.masked_fill_(attn_mask, -1e4)
        # torch.softmax() 计算softmax，dim=-1表示在最后一个维度上计算
        attns = torch.softmax(scores, dim=-1)
        attns = self.dropout(attns)  # 应用dropout

        # calculate output - 计算输出
        # 注意力权重与Value相乘得到加权的Value
        output = torch.matmul(attns, V)  # [N, num_heads, seq_len, d_v]

        # merge heads - 合并多头
        # .transpose(1,2) 将头数和序列长度维度交换回来
        # .contiguous() 确保张量在内存中是连续的，这是reshape操作的要求
        # .reshape() 将多头结果合并为一个张量
        output = output.transpose(1,2).contiguous().reshape(N,-1,d_v*num_heads)  # [N, seq_len, d_v*num_heads]
        output = self.W_out(output)  # 通过输出投影层，形状变为[N, seq_len, d_model]

        return output

class PoswiseFFN(nn.Module):
    """
    位置相关的前馈神经网络
    
    这个类实现了Transformer中的Position-wise Feed-Forward Network (FFN)。
    它对序列中的每个位置独立地应用相同的前馈网络，包含两个线性变换和一个ReLU激活函数。
    结构为: Linear -> ReLU -> Linear -> Dropout
    通过1D卷积实现，相当于对每个位置进行逐点的全连接操作。
    """
    def __init__(self, d_model, d_ff, p=0.):
        # d_model: dimension of model
        # d_ff: dimension of feedforward
        # p: dropout rate
        super(PoswiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # nn.Conv1d参数解释:
        # 第1个参数 d_model: 输入通道数，即输入特征的维度
        # 第2个参数 d_ff: 输出通道数，即输出特征的维度  
        # 第3个参数 1: 卷积核大小(kernel_size)，这里是1表示逐点卷积
        # 第4个参数 1: 步长(stride)，每次移动1个位置
        # 第5个参数 0: 填充(padding)，不进行填充
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, 1, 0)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        # X的形状是 [batch_size, seq_len, d_model]
        # Conv1d期望输入形状为 [batch_size, channels, length]
        # 所以需要转置将 d_model 维度移到第2维，seq_len 移到第3维
        out = self.conv1(X.transpose(1,2))
        out = self.relu(out)
        out = self.conv2(out).transpose(1,2)
        out = self.dropout(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: dimension of model
            n: number of layers
            dff: dimension of feedforward
            dropout_posffn: dropout rate of positionwise feedforward network
            dropout_attn: dropout rate of attention
        """
        assert dim % n == 0, "dim must be divisible by n"
        hdim = dim // n # head dimension
        super(EncoderLayer, self).__init__()

        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # MultiHeadAttention
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim,dff,p=dropout_posffn)

    def forward(self,enc_in,attn_mask):
        residual = enc_in
        # MultiHeadAttention
        context = self.multi_head_attn(enc_in,enc_in,enc_in,attn_mask)
        # residual connection and norm
        out = self.norm1(residual + context)
        residual = out
        # Position-wise Feed-Forward Network
        out = self.poswise_ffn(out)
        # residual connection and norm
        out = self.norm2(residual + out)
        return out

class Encoder(nn.Module):
    def __init__(
        self,dropout_emb,dropout_posffn,dropout_attn,
        num_layers,enc_dim,num_heads,dff,tgt_len,
    ):
        """
        args:
            dropout_emb: dropout rate of embedding
            dropout_posffn: dropout rate of positionwise feedforward network
            dropout_attn: dropout rate of attention
            num_layers: number of layers
            enc_dim: dimension of model
            num_heads: number of heads
            dff: dimension of feedforward
            tgt_len: length of target
        """
        super(Encoder,self).__init__()
        # the maximum length of input sequence
        self.tgt_len = tgt_len
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len,enc_dim),freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim,num_heads,dff,dropout_posffn,dropout_attn) for _ in range(num_layers)]
        )

    def forward(self, X, X_lens, mask=None):
        # add position embedding
        batch_size,seq_len,d_model = X.shape
        out = X + self.pos_emb(torch.arange(seq_len,device=X.device))
        out = self.emb_dropout(out)
        #encoder layers
        for layer in self.layers:
            out = layer(out,mask)
        return out

class DecoderLayer(nn.Module):
    def __init__(self,dim,n,dff,dropout_posffn,dropout_attn):
        super(DecoderLayer,self).__init__()
        assert dim % n == 0
        hdim = dim // n
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # Position-wise Feed-Forward networks
        self.poswise_ffn = PoswiseFFN(dim,dff,p=dropout_posffn)
        # MultiHeadAttention,both self-attention and cross-attention
        self.dec_attn = MultiHeadAttention(hdim,hdim,dim, n, dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(hdim,hdim,dim, n, dropout_attn)

    def forward(self,dec_in,enc_out,dec_mask,dec_enc_mask,cache=None,freqs_cis=None):
        # decoder's self-attention
        residual = dec_in
        context = self.dec_attn(dec_in,dec_in,dec_in,dec_mask)
        dec_out = self.norm1(residual + context)
        # encoder-decoder cross-attention
        residual = dec_out
        context = self.enc_dec_attn(dec_out,enc_out,enc_out,dec_enc_mask)
        dec_out = self.norm2(residual + context)
        # position-wise feed-forward networks
        residual = dec_out 
        out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(residual + out)
        return dec_out 

class Decoder(nn.Module):
    def __init__(
        self,dropout_emb,dropout_posffn,dropout_attn,
        num_layers,dec_dim,num_heads,dff,tgt_len,tgt_vocab_size,
    ):
        """
        args:
            dropout_emb: dropout rate of embedding
            dropout_posffn: dropout rate of positionwise feedforward network
            dropout_attn: dropout rate of attention
            num_layers: number of layers
            dec_dim: dimension of model
            num_heads: number of heads
            dff: dimension of feedforward
            tgt_len: length of target
            tgt_vocab_size: size of target vocabulary
        """
        super(Decoder,self).__init__()

        # out embedding
        self.tgt_emb = nn.Embedding(tgt_vocab_size,dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)
        # position embedding
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len,dec_dim),freeze=True)
        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dec_dim,num_heads,dff,dropout_posffn,dropout_attn) for _ in range(num_layers)
            ]
        )

    def forward(self,labels,enc_out,dec_mask,dec_enc_mask,cache=None):
        # output embedding and position embedding
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1),device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)
        #decoder layers
        for layer in self.layers:
            dec_out = layer(dec_out,enc_out,dec_mask,dec_enc_mask)
        return dec_out

class Transformer(nn.Module):
    def __init__(
        self,frontend:nn.Module,encoder:nn.Module,decoder:nn.Module,
        dec_out_dim:int,vocab:int,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim,vocab)

    def forward(self,X:torch.Tensor,X_lens:torch.Tensor,labels:torch.Tensor):
        X_lens,labels = X_lens.long(),labels.long()
        b = X.size(0)
        device = X.device
        # frontend
        out = self.frontend(X)
        max_feat_len = out.size(1)
        max_label_len = labels.size(1)
        #encoder 
        enc_mask = get_len_mask(b,max_feat_len,X_lens,device)
        enc_out = self.encoder(out,X_lens,enc_mask)
        # decoder 
        dec_mask = get_subsequent_mask(b,max_label_len,device)
        dec_enc_mask = get_enc_dec_mask(b,max_feat_len,X_lens,max_label_len,device)
        dec_out = self.decoder(labels,enc_out,dec_mask,dec_enc_mask)
        logits = self.linear(dec_out)

        return logits

if __name__ == "__main__":
    # constants
    batch_size = 16                 # batch size
    max_feat_len = 100              # the maximum length of input sequence
    max_lable_len = 50              # the maximum length of output sequence
    fbank_dim = 80                  # the dimension of input feature
    hidden_dim = 512                # the dimension of hidden layer
    vocab_size = 26                 # the size of vocabulary

    # dummy data
    fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim)        # input sequence
    feat_lens = torch.randint(1, max_feat_len, (batch_size,))               # the length of each input sequence in the batch
    labels = torch.randint(0, vocab_size, (batch_size, max_lable_len))      # output sequence
    label_lens = torch.randint(1, max_lable_len, (batch_size,))             # the length of each output sequence in the batch

    # model
    feature_extractor = nn.Linear(fbank_dim, hidden_dim)                     # alinear layer to simulate the audio feature extractor
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, enc_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, dec_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048, tgt_vocab_size=vocab_size
    )
    transformer = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size)

    # forward check
    logits = transformer(fbank_feature, feat_lens, labels)
    print(f"logits: {logits.shape}")     # (batch_size, max_label_len, vocab_size)

    # output msg
    # logits: torch.Size([16, 100, 26])