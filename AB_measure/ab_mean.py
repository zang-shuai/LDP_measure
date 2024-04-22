import math
import random

import numpy as np
from sklearn.cluster import KMeans

from LDP.mean_ldp.duchi import Duchi
from LDP.mean_ldp.piecewisemechanism import PiecewiseMechanism


class ab_DUCHI():
    def __init__(self, epsilon, inputData, n=40000):
        self.epsilon = epsilon
        self.inputData = inputData
        self.n = n
        self.client = Duchi(epsilon=self.epsilon, inputData=inputData)

    def getPerhurb(self, data):
        return self.client.privatise(data)

    def getEncode(self, data):
        return self.client.encode(data)

    def access(self):
        count = 0
        datas = [random.uniform(self.inputData[0], self.inputData[1]) for _ in range(10)]
        # print(datas)
        epsilons = []
        for data in datas:
            num = 0
            for i in range(self.n):
                if self.getPerhurb(data) > 0:
                    num += 1
            p = num / self.n
            try:
                # epsilons.append(math.log(1 / (1 / 2 - (p - 1 / 2) / self.getEncode(data)) - 1))
                # print(p)
                x = math.fabs(self.getEncode(data))
                epsilons.append(math.fabs(math.log((2*x)/(x-2*p+1) - 1)))
                # print('----')
                # print(epsilons)
                # print('----')
            except Exception:
                pass

        return np.array(epsilons).mean()


class ab_PiecewiseMechanism():
    def __init__(self, epsilon, inputData, n=40000):
        self.epsilon = epsilon
        self.inputData = inputData
        self.n = n
        self.client = PiecewiseMechanism(epsilon=self.epsilon, domain=inputData)

    def getPerhurb(self, data):
        return self.client.privatise(data)

    def getEncode(self, data):
        return self.client.encode(data)

    def access(self):
        datas = [random.uniform(self.inputData[0], self.inputData[1]) for _ in range(10)]
        epsilons = []
        for data in datas:
            perhurbs = [self.getPerhurb(data) for _ in range(self.n)]
            # 计算直方图
            bins = int(math.sqrt(self.n))  # 定义区间数
            counts, bin_edges = np.histogram(perhurbs, bins)
            kmeans = KMeans(n_clusters=2)
            kmeans_labels = kmeans.fit_predict(counts.reshape(-1, 1))
            p = 0
            p_count = 0
            q = 0
            q_count = 0
            for i in range(bins):
                if kmeans_labels[i] == 1:
                    p += counts[i]
                    p_count += 1
                else:
                    q += counts[i]
                    q_count += 1
            p = p / self.n / (p_count * (max(perhurbs) - min(perhurbs)) / bins)
            q = q / self.n / (q_count * (max(perhurbs) - min(perhurbs)) / bins)
            if p / q > 1:
                epsilons.append(math.log(p / q))
            else:
                epsilons.append(math.log(q / p))
        return np.array(epsilons).mean()
