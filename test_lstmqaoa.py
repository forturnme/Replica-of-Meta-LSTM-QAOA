from lstm_learner import *
from random_qaoa_episode import *
from graph_generator import *
from mindquantum import Simulator
import mindspore as ms
from meta_loss_function import *
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


n = 8
p = 10
G, k = random_graph_instance(n)
draw_img(G)


sim = Simulator('projectq', k)
cir, ham = gen_qaoa_episode(G, p)
op = gen_qaoa_ops(sim, cir, ham)
# print(cir.params_name)
model = LSTMQAOA(len(cir.params_name), hidden_dim=len(cir.params_name), n_layers=6, ansatz_op=op)

seq_len = ms.Tensor([1], ms.int64)
print(seq_len.shape)
init_paras = ms.Tensor(np.random.randn(1, 1, len(cir.params_name))*1.57).astype(ms.float32)
initial_energy, next_paras = model(init_paras, seq_len)
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))


class CustomWithLossCell(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, seqlen, target):
        output, para = self._backbone(data, seqlen)         # 前向计算得到网络输出
        return self._loss_fn(output, target), para          # 得到多标签损失值


class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss, out = self.network(*inputs)                       # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer(grads)                                   # 使用优化器更新权重参数
        return loss, out


loss = MetaLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
target = ms.Tensor([0])
# print(loss(initial_energy, target))
net_with_loss = CustomWithLossCell(model, loss)
train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
train_net.set_train()
history = []
for i in range(2000):
    lo, next_paras = train_net(init_paras, seq_len, target)
    init_paras = ms.Tensor(next_paras.asnumpy()).astype(ms.float32)
    if (i+1)%50==0:
        loss = lo.asnumpy()
        print(f'iter{i+1:04d}: loss {loss}')
        history.append(loss)


import matplotlib.pyplot as plt
plt.plot([50*(i+1) for i in range(len(history))], history, label='LSTM')
plt.savefig('convergence_p=8.png')
