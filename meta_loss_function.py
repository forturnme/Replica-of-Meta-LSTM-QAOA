"""
    LSTM的loss functions.
    L(phi) = 
    E_(f,y){
        sum_t[
            min(
                f(theta_t)-f(best_theta_for_now),
                0
                )
            ]
        }
    但是这个loss无法求导，不能用作训练。
    文中没有提到解决方法。
"""
from mindspore import nn
from mindspore import Tensor, ops
import numpy as np


class MetaLoss(nn.LossBase):
    def __init__(self):
        super(MetaLoss, self).__init__()
        self.min = ops.Minimum()

    def construct(self, base, target):
        """
            我不知道怎么实现文中的Meta Loss，
            所以这里先用正常的qaoa loss了。
        """
        # print(base, target)
        x = base-target
        # x = self.min(base-target,0)
        return self.get_loss(x)
