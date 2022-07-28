'''
    用于meta learn的lstm模型。文章并未提及模型的具体结构。
    问题：
    1. 是否需要全连接层？
    2. 堆叠多少层LSTM？
    3. 梯度从哪里来？（最关键）
    4. 训练数据怎么获得？
'''
from mindspore import nn
import numpy as np
from sympy import sequence


class LSTMLearner(nn.Cell):
    def __init__(self, para_count, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(para_count, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        

    def construct(self, inputs, seq_length):
        outputs, _ = self.lstm(inputs, seq_length=seq_length)
        return outputs


class LSTMQAOA(nn.Cell):
    def __init__(self, para_count, hidden_dim, n_layers, ansatz_op):
        super().__init__()
        self.lstm = nn.LSTM(para_count, hidden_dim, n_layers, bidirectional=False, batch_first=True)
        self.ansatz_op = ansatz_op


    def construct(self, inputs, seq_length):
        outputs, _ = self.lstm(inputs, seq_length=seq_length)
        e = self.ansatz_op(outputs.flatten())
        return e, outputs
