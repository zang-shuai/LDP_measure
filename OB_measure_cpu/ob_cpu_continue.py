import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import time
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import random
from LDP.frequency_ldp.frequency_oracles import HEClient

from itertools import chain

from LDP.mean_ldp.piecewisemechanism import PiecewiseMechanism

# from BlackBox.utils import run_n

# err = []
# linestyle = {
#     '0.3': '-',
#     '0.5': '--',
#     '0.8': 'dotted'
# }


# 连续向量
class countLDP_continue(object):
    def __init__(self, mechanism, h=1, kernel='gaussian', N=10000, min=0.02):
        self.mechanism = mechanism
        # 核密度估计带宽
        self.h = h
        # 核密度估计的核函数（Str）
        self.kernel = kernel

        self.N = N

        self.min = min

    # 将最小值托底
    def delmin(self, n):
        for ix in range(len(n)):
            if n[ix] < math.log(self.min):
                n[ix] = math.log(self.min)
        return n

    def getfx(self, data, C1, C2):
        x_model = self.kernel_density(data)
        sequence = np.linspace(-C1, C2, self.N)
        ell = self.delmin(x_model.score_samples(sequence[:, np.newaxis]))
        return ell

    def kernel_density(self, data):
        x_train = np.array(data)
        model = KernelDensity(bandwidth=self.h, kernel=self.kernel)
        model.fit(x_train[:, np.newaxis])
        return model

    def get_data(self, data, p):
        m = sorted(data)
        return m[int(len(data) * p)]

    def get_max(self, lmax, lmin):
        res = 0
        for i in range(len(lmax)):
            dict_max, dict_min = lmax[i], lmin[i]
            # dddd = np.abs(np.exp(dict_max) - np.exp(dict_min))
            # plt.plot(list(range(0, len(dddd))), np.exp(dddd), 'o')
            # plt.title("Random Floats (0 to 0.02)")
            # plt.xlabel("Index")
            # plt.ylabel("Value")
            # plt.show()

            data = self.get_data(np.abs(dict_max - dict_min), 0.9)
            # err.append(data)
            if data > 0.1:
                res += data
        return res

    def evaluation(self, num=50000, D=None):
        # 输出列表
        outputs = []

        if D is None:
            # 获取长度
            k = self.mechanism.d
            # 获取数据
            datas = range(1, k + 1)
            datas = random.sample(datas, max(math.ceil(np.log(k)), 2))
            k=max(math.ceil(np.log(k)), 2)
            for data in datas:
                outputs.append([self.mechanism.privatise(data) for _ in range(num)])
        else:
            k = max(math.ceil(np.log((-self.mechanism.a+ self.mechanism.b)//D)), 2)
            datas = list(
                range(self.mechanism.a, self.mechanism.b + 1, k))
            k = len(datas )

            for data in datas:
                outputs.append([[self.mechanism.privatise(data),] for _ in range(num)])

        # 输出概率列表，里面存储k个字典，每个字典里有输入为x输出结果的集合与其相应概率
        output_fx = []
        for output in outputs:
            pro = []
            output = np.array(output)
            out = sorted(list(chain.from_iterable(output)))
            C1 = out[int(len(output) * 0.05)]
            C2 = out[int(len(output) * 0.95)]
            for i in range(len(output[0])):
                pro.append(self.getfx(output[:, i], C1, C2))
            output_fx.append(pro)
        epsilon = []

        for i, j in list(itertools.combinations(range(k), 2)):
            output_in, output_out = output_fx[i], output_fx[j]
            epsilon.append(self.get_max(output_in, output_out))
            if D is None:
                break
        return max(epsilon)

#
# if __name__ == '__main__':
#     epsilons = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
#     # for epsilon in epsilons:
#     #     de = DEClient(epsilon, 100)
#     #     cde = countLDP_discrete(de)
#     #     print(cde.evaluation(100_000))
#     #
#     # print()
#     # for epsilon in epsilons:
#     #     sue = UEClient(epsilon, 100)
#     #     cde = countLDP_discrete(sue)
#     #     print(cde.evaluation(500_00))
#     #
#     # print()
#
#     print()
#     for epsilon in epsilons:
#         pm = PiecewiseMechanism(epsilon, [80, 120])
#         cde = countLDP_continue(pm)
#         print(cde.evaluation(500_00,5))
#     print()
#     # for epsilon in epsilons:
#     #     he = HEClient(epsilon, 100)
#     #     cde = countLDP_continue(he)
#     #     print(cde.evaluation(500_00))
#
#
#     # print()
#     # for epsilon in epsilons:
#     #     duchi = HE(epsilon, [80, 120])
#     #     cde = countLDP_discrete(duchi)
#     #     print(cde.evaluation(500_00,5))
#
#     # print()
#     # for epsilon in epsilons:
#     #     duchi = Duchi(epsilon, [80,120])
#     #     cde = countLDP_discrete(duchi)
#     #     print(cde.evaluation(500_00))
#     # oue = UEClient(epsilon, 100, use_oue=True)
#     # cde = countLDP_discrete(oue)
#     # print(cde.evaluation(100_0))
#     # print()
#     # oue
#     # run1(epsilons=epsilons, n=100, m=40000)
#     # # sue
#     # run2(epsilons=epsilons, n=100, m=40000)
#     # de
#     # run3(epsilons=epsilons, n=100, m=10000)
#     # # ds
#     # run4(epsilons=epsilons, n=100, m=40000)
#     # xxx = sorted(xxx)
#     # print(xxx[979], xxx[980])
#     # cdf(sorted(xxx))
#     # print(len(xxx))
#     # run5(epsilons=epsilons, n=100, m=10000)
#
# # def run(epsilons):
# #     from pure_ldp.frequency_oracles import HEClient
# #     for epsilon in epsilons:
# #         oue = HEClient(epsilon, 5)
# #         coue = countLDP_continue(oue)
# #         # run_n(coue.evaluation, n=100)
# #         run_n(coue.evaluation, n=100, label='$\epsilon=$' + str(epsilon), linestyle=linestyle[str(epsilon)])
# #     plt.legend()
# #     plt.title('直方图编码 (HE) ')
# #     plt.show()
# #
# #
# # if __name__ == '__main__':
# #     # run((0.3,0.5,0.8))
# #     run((0.9, 0.5, 0.8))
# #     print('++++++++')
# #     print(len(output_fx))
# #     print(len(output_fx[0]))
# #     print(len(output_fx[0][0]))
# #     plt.plot(list(range(0, len(err))), np.exp(np.array(err)), 'o')
# #     plt.title("Random Floats (0 to 0.02)")
# #     plt.xlabel("Index")
# #     plt.ylabel("Value")
# #     plt.show()
# #     print('++++++++')
