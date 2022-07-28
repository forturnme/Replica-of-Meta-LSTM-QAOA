'''
    按照文章
    A Quantum Approximate Optimization 
    Algorithm with Metalearning for MaxCut Problem 
    and Its Simulation via TensorFlow Quantum
    要求，生成随机图
'''
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def random_graph_instance(n):
    def __rand_edge(G,vi,vj,p):
        prob = np.random.random()#生成随机小数
        if(prob<p):			#如果小于p
            G.add_edge(vi,vj)
    k = np.random.randint(3,n-1)
    p_connect = k/n
    rg = nx.Graph()
    h = nx.path_graph(k)
    rg.add_nodes_from(h)
    for i in range(k):
        for j in range(i+1,k):
            __rand_edge(rg, i, j, p_connect)
    return rg, k


def draw_img(G):
    pos = nx.circular_layout(G)
    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 2000,
        "width": 2
    }
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    G, k = random_graph_instance(8)
    draw_img(G)
