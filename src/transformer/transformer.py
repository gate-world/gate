"""
transformer的示例程序，用来展示它是如何训练和工作的
输入是一串数字，输出是每个数字加上一
24.11.14
"""

import torch
import torch.nn as nn
import torch.optim as optim


# 定义简单的 Transformer 模型
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        """
        input_dim: 输入数据的维度，实际上是有多少种类的单词
        model_dim: 嵌入向量的维度
        """
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, model_dim)  # 输入嵌入层
        self.positional_encoding = nn.Parameter(torch.randn(1, 10, model_dim))  # 假设最大序列长度为 10

        # Transformer 层
        self.transformer = nn.Transformer(d_model=model_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)

        self.fc_out = nn.Linear(model_dim, output_dim)  # 输出层

    def forward(self, src, tgt):
        # 将输入序列和目标序列通过嵌入层加上位置编码
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        src = src.transpose(0, 1)  # Transformer 要求的输入格式: (seq_len, batch_size, model_dim)
        tgt = tgt.transpose(0, 1)  # Transformer 要求的输入格式: (seq_len, batch_size, model_dim)

        # 使用 Transformer 模型
        output = self.transformer(src, tgt)

        # 将输出维度变为 (batch_size, seq_len-1, output_dim)
        output = self.fc_out(output)  # 输出维度：(seq_len, batch_size, output_dim)
        output = output.transpose(0, 1)  # 转置为 (batch_size, seq_len, output_dim)

        return output


# 数据生成：输入是一个整数序列，目标是其加 1
def generate_data(batch_size, seq_len, input_dim):
    src = torch.randint(0, input_dim-1, (batch_size, seq_len))  # 输入是 0 到 input_dim-1 之间的整数，因为之后tgt要加上1
    tgt = src + 1  # 目标是输入加 1
    return src, tgt


# 假设输入序列长度为 10，词汇表大小为 5（0, 1, 2, 3, 4），批量大小为 32
batch_size = 32
seq_len = 10
input_dim = 5
output_dim = 5  # 因为目标序列是整数，输出是一个整数

# 创建模型
model = SimpleTransformer(input_dim, model_dim=64, num_heads=2, num_layers=2, output_dim=output_dim)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 对于整数预测，使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    # 生成一个批次的数据
    src, tgt = generate_data(batch_size, seq_len, input_dim)

    # 在目标序列中，将第一个元素作为解码器输入，其余作为标签
    tgt_input = tgt[:, :-1]  # 去掉最后一个元素
    tgt_output = tgt[:, 1:]  # 去掉第一个元素作为标签

    # 前向传播
    output = model(src, tgt_input)

    # 计算损失
    output = output.reshape(-1, output_dim)  # Flatten the output to (batch_size * seq_len-1, output_dim)
    tgt_output = tgt_output.contiguous().view(-1)  # Flatten the target to (batch_size * seq_len-1)

    loss = criterion(output, tgt_output)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
