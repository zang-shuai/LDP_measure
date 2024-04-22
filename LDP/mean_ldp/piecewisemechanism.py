# import itertools

# import numpy as math
import random
import math

# import numpy as math
import random


# from sklearn import preprocessing
# from sklearn.neighbors import KernelDensity
#
# from LDP.mean_ldp.duchi import is_iterable


# class PiecewiseMechanism2(object):
#     def __init__(self, epsilon, domain):
#         self.epsilon = epsilon
#         self.domain = domain
#         self.a = domain[0]
#         self.b = domain[1]
#         # self.d = math.arange(self.a, self.b, 0.1)
#
#     def encode(self, datas):
#         # if not is_iterable(datas):
#         #     return 2 * datas / (self.b - self.a) + (self.a + self.b) / (self.a - self.b)
#         try:
#             ans = [2 * data / (self.b - self.a) + (self.a + self.b) / (self.a - self.b) for data in datas]
#         except Exception:
#             ans = [2 * datas / (self.b - self.a) + (self.a + self.b) / (self.a - self.b), ]
#         return ans
#
#     def privatise(self, datas):
#         C = (math.exp(self.epsilon / 2) + 1) / (math.exp(self.epsilon / 2) - 1)
#         p = (math.exp(self.epsilon) - math.exp(self.epsilon / 2)) / (2 * math.exp(self.epsilon / 2) + 2)
#         ans = []
#         datas = self.encode(datas)
#         for data in datas:
#             l = (C + 1) / 2 * data - (C - 1) / 2
#             r = l + C - 1
#             p1 = (C + l) * p / math.exp(self.epsilon)
#             p2 = (r - l) * p
#             p3 = (C - r) * p / math.exp(self.epsilon)
#             getOne = random.choices(
#                 [math.random.uniform(-C, l, 1)[0], math.random.uniform(l, r, 1)[0], math.random.uniform(r, C, 1)[0]],
#                 weights=[p1, p2, p3], k=1)[0]
#             ans.append(getOne)
#         if len(ans) == 1:
#             return ans[0]
#         return ans
#
#     def get_expectation(self, datas):
#         return ((sum(self.privatise(datas)) / len(datas)) * (self.b - self.a) + self.a + self.b) / 2

# def (self, data):
#     pass

class PiecewiseMechanism(object):
    def __init__(self, epsilon, domain):
        self.epsilon = epsilon
        self.domain = domain
        self.a = domain[0]
        self.b = domain[1]
        self.C = (math.exp(self.epsilon / 2) + 1) / (math.exp(self.epsilon / 2) - 1)
        self.p = (math.exp(self.epsilon) - math.exp(self.epsilon / 2)) / (2 * math.exp(self.epsilon / 2) + 2)

    def encode(self, data):
        return 2 * data / (self.b - self.a) + (self.a + self.b) / (self.a - self.b)

    def privatise(self, data):
        l = (self.C + 1) / 2 * self.encode(data) - (self.C - 1) / 2
        r = l + self.C - 1
        p1 = (self.C + l) * self.p / math.exp(self.epsilon)
        p2 = (r - l) * self.p
        p3 = (self.C - r) * self.p / math.exp(self.epsilon)
        return random.choices([random.uniform(-self.C, l), random.uniform(l, r), random.uniform(r, self.C)],
                              weights=[p1, p2, p3], k=1)[0]

        # datas = self.encode(datas)
        # for data in datas:
        #     l = (C + 1) / 2 * data - (C - 1) / 2
        #     r = l + C - 1
        #     p1 = (C + l) * p / math.exp(self.epsilon)
        #     p2 = (r - l) * p
        #     p3 = (C - r) * p / math.exp(self.epsilon)
        #     getOne = random.choices(
        #         [math.random.uniform(-C, l, 1)[0], math.random.uniform(l, r, 1)[0], math.random.uniform(r, C, 1)[0]],
        #         weights=[p1, p2, p3], k=1)[0]
        #     ans.append(getOne)
        # if len(ans) == 1:
        #     return ans[0]
        # return ans

    # def get_expectation(self, datas):
    #     return ((sum(self.privatise(datas)) / len(datas)) * (self.b - self.a) + self.a + self.b) / 2

# pm = PiecewiseMechanism(0.3, [10, 100])
#
# datas = math.random.uniform(0, 100, 10000)
#
# print(datas)
# ans = []
# for i in range(100):
#     ans.append(pm.get_expectation(datas))
# cdf(ans)

#
# pm = PiecewiseMechanism(0.8, [10, 100])
# datas = []
# for i in range(10, 101):
#     datas.append([i for _ in range(100000)])
#
# ans = []
# for data in datas:
#     d = pm.privatise(data)
#     # print(len(p))
#
#     # ans.append(count_probability(d))
#     C = 10
#     sequence = math.linspace(-C, C, 1000)
#     model = KernelDensity(bandwidth=0.125, kernel='gaussian')
#     model.fit(math.array(d)[:, math.newaxis])
#     f = model.score_samples(sequence[:, math.newaxis])
#     ans.append(f)
#
# print(len(ans))
# print(len(ans[0]))
#
#
#
#
# a = list(itertools.combinations(range(91), 2))
# print(a)
#
# # print(math.fabs(ans[0]-ans[-1]))
# res = []
# print(ans[0][:10])
# print(ans[-1][:10])
# for i in range(len(ans[0])):
#     res.append(math.fabs(ans[0][i]-ans[-1][i]))
#
# print(res)
#
# p = []
# for ii,j in itertools.combinations(range(91), 2):
#     res = []
#     print(len(p))
#     # print(ans[0][:10])
#     # print(ans[-1][:10])
#     for i in range(len(ans[0])):
#         res.append(math.fabs(ans[ii][i] - ans[j][i]))
#     p.append(max(res))
#     # print(res)
# print(max(p))
# # import math
# #
# # e = 1.3
# # print((math.exp(e/2)+1)/((math.exp(e/2)-1)))
