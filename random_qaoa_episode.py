"""
    为随机图生成哈密顿量。
"""
from mindquantum import Circuit, Hamiltonian,\
     ZZ, RX, H, QubitOperator
import numpy as np
import networkx as nx
from mindquantum import Simulator, MQAnsatzOnlyOps


def build_hc(g, para):
    hc = Circuit()                  # 创建量子线路
    for i in g.edges:
        hc += ZZ(para).on(i)        # 对图中的每条边作用ZZ门
    hc.barrier()                    # 添加Barrier以方便展示线路
    return hc


def build_hb(g, para):
    hb = Circuit()                  # 创建量子线路
    for i in g.nodes:
        hb += RX(para).on(i)        # 对每个节点作用RX门
    hb.barrier()                    # 添加Barrier以方便展示线路
    return hb


def gen_qaoa_episode(G, p):
    circ = Circuit()                       # 创建量子线路
    for i in range(p):
        circ += build_hc(G, f'g{i:03d}')       # 添加Uc对应的线路，参数记为g0、g1、g2...
        circ += build_hb(G, f'b{i:03d}')       # 添加Ub对应的线路，参数记为b0、b1、b2...
    ham = QubitOperator()
    for i in G.edges:
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}')  # 生成哈密顿量Hc
    return circ, Hamiltonian(ham)


def gen_qaoa_ops(sim, cir, ham):
    grad_ops = sim.get_expectation_with_grad(ham,
                                         cir)
    qaoa_ops = MQAnsatzOnlyOps(grad_ops)
    return qaoa_ops
