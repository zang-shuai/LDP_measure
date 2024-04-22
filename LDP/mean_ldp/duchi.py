import math

import numpy as np
import random
from sklearn import preprocessing


def is_iterable(variable):
    try:
        iter(variable)
        return True
    except TypeError:
        return False


# class Duchi(object):
#     def __init__(self, epsilon, inputData):
#         self.epsilon = epsilon
#         self.a = inputData[0]
#         self.b = inputData[1]
#
#     def privatise(self, datas):
#         x = (np.exp(self.epsilon) + 1) / (np.exp(self.epsilon) - 1)
#         ans = []
#         if is_iterable(datas):
#             for data in datas:
#                 x_p = 2 * data / (self.b - self.a) + (self.a + self.b) / (self.a - self.b)
#                 p = x_p / (2 * x) + 1 / 2
#                 ans.append(random.choices([x, -x], weights=[p, 1 - p], k=1)[0])
#         else:
#             data = datas
#             x_p = 2 * data / (self.b - self.a) + (self.a + self.b) / (self.a - self.b)
#             p = x_p / (2 * x) + 1 / 2
#             ans.append(random.choices([x, -x], weights=[p, 1 - p], k=1)[0])
#             return ans[0]
#
#         return ans
#
#     def encode(self, datas):
#         if not is_iterable(datas):
#             return 2 * datas / (self.b - self.a) + (self.a + self.b) / (self.a - self.b)
#         try:
#             ans = [2 * data / (self.b - self.a) + (self.a + self.b) / (self.a - self.b) for data in datas]
#         except Exception:
#             ans = [2 * datas / (self.b - self.a) + (self.a + self.b) / (self.a - self.b), ]
#         return ans
#
#     def get_expectation(self, datas):
#         return ((sum(self.privatise(datas)) / len(datas)) * (self.b - self.a) + self.a + self.b) / 2


class Duchi(object):
    def __init__(self, epsilon, inputData):
        self.epsilon = epsilon
        self.a = inputData[0]
        self.b = inputData[1]
        self.x = (math.exp(self.epsilon) + 1) / (math.exp(self.epsilon) - 1)

    def privatise(self, data):
        # print(self.a)
        # print(self.b)
        # print(data )
        p = self.encode(data) / (2 * self.x) + 1 / 2
        if random.random() < p:
            return self.x
        else:
            return -self.x

    def encode(self, data):
        return 2 * data / (self.b - self.a) + (self.a + self.b) / (self.a - self.b)

    # def get_expectation(self, datas):
    #     return ((sum(self.privatise(datas)) / len(datas)) * (self.b - self.a) + self.a + self.b) / 2

# duchi = Duchi(2.3, [10, 100])
# datas = 10 + (100 - 10) * np.random.random(10000)
# privatise = duchi.privatise(50)
# print(privatise)
# print(datas)
# ans = []
# for i in range(100):
#     ans.append(duchi.get_expectation(datas))
# cdf(ans)


# datas = np.random.uniform(10, 100, 10000)
# print(datas)
# ans = []
# for i in range(100):
#     ans.append(pm.get_expectation(datas))
# cdf(ans)
