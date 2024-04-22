import math
import random

import numpy as np
import xxhash
from pure_ldp.frequency_oracles import DEClient,UEClient, HEClient, LHClient



class ab_DE():
    def __init__(self, epsilon, d, n):
        self.epsilon = epsilon
        self.d = d
        self.n = n
        self.client = DEClient(epsilon=self.epsilon, d=self.d)

    def getPerhurb(self, data):
        return self.client.privatise(data)

    def getEncode(self, data):
        return data - 1

    def access(self):
        count = 0
        for i in range(self.n):
            data = random.randint(0, self.d - 1)
            client_olh_perhurb = self.getPerhurb(data)
            client_olh_encode = self.getEncode(data)
            if client_olh_perhurb == client_olh_encode:
                count += 1
        p = count / self.n
        return math.log((self.d - 1) / (1 - p) - self.d + 1)


class ab_SUE():
    def __init__(self, epsilon=0.3, d=5, n=4000):
        self.epsilon = epsilon
        self.d = d
        self.n = n
        self.client = UEClient(epsilon=self.epsilon, d=self.d, use_oue=False)
        pass

    def getPerhurb(self, data):
        return self.client.privatise(data)

    def getEncode(self, data):
        index = self.client.index_mapper(data)
        en = np.zeros(self.d, dtype=int)
        en[index] = 1
        return en

    def access(self):
        count = 0
        for _ in range(self.n):
            data = random.randint(1, self.d)
            index = data
            client_perhurb = self.getPerhurb(index)
            client_encode = self.getEncode(index)
            for j in range(self.d):
                if client_perhurb[j] != client_encode[j]:
                    count += 1
        q = count / (self.n * self.d)

        return 2 * math.log(1 / q - 1)


class ab_OUE():
    def __init__(self, epsilon=0.3, d=5, n=4000):
        self.epsilon = epsilon
        self.d = d
        self.n = n
        self.client = UEClient(epsilon=self.epsilon, d=self.d, use_oue=True)
        pass

    def getPerhurb(self, data):
        return self.client.privatise(data)

    def getEncode(self, data):
        index = self.client.index_mapper(data)
        en = np.zeros(self.d, dtype=int)
        en[index] = 1
        return en

    def access(self):
        count = 0
        for _ in range(self.n):
            data = random.randint(1, self.d)
            index = data
            client_perhurb = self.getPerhurb(index)
            client_encode = self.getEncode(index)
            for j in range(self.d):
                # if client_encode[j] == 0 and client_perhurb[j] != 0:
                if client_encode[j] != client_perhurb[j]:
                    count += 1

        q = count / (self.n * (self.d - 1))

        return math.log(1 / q - 1)


class ab_HE():
    def __init__(self, epsilon=0.3, d=5, n=40000):
        self.epsilon = epsilon
        self.d = d
        self.n = n
        self.client = HEClient(epsilon=self.epsilon, d=self.d)
        pass

    def getPerhurb(self, data):
        return self.client.privatise(data)
        pass

    def getEncode(self, data):
        en = np.zeros(self.d, dtype=int)
        en[data] = 1
        return en

    def access(self):
        ans = []
        for _ in range(self.n):
            data = random.randint(1, self.d)
            index = self.client.index_mapper(data)
            client_perhurb = self.getPerhurb(index)
            client_encode = self.getEncode(index)
            for j in range(self.d):
                ans.append(client_encode[j] - client_perhurb[j])
        sigma = np.var(ans)
        return 2 / ((sigma / 2) ** (1 / 2))


class ab_BLH():
    def __init__(self, epsilon, d, n):
        self.epsilon = epsilon
        self.d = d
        self.n = n
        self.client = LHClient(epsilon=self.epsilon, d=self.d, g=2, use_olh=False)

    def getPerhurb(self, data):
        return self.client.privatise(data)

    def getEncode(self, data, seed):
        return (xxhash.xxh32(str(self.client.index_mapper(data)),
                             seed=seed).intdigest() % self.client.g)

    def access(self):
        count = 0
        for i in range(self.n):
            data = random.randint(0, self.d)
            client_olh_perhurb = self.getPerhurb(data)
            client_olh_perhurb_data = client_olh_perhurb[0]
            client_olh_perhurb_seed = client_olh_perhurb[1]
            client_olh_encode = self.getEncode(data, client_olh_perhurb_seed)
            if client_olh_perhurb_data != client_olh_encode:
                count += 1
        q = count / self.n
        return math.log(1 / q - self.client.g + 1)


class ab_OLH():
    def __init__(self, epsilon, d, n):
        self.epsilon = epsilon
        self.d = d
        self.n = n
        self.use_olh = True
        self.client = LHClient(epsilon=self.epsilon, d=self.d, g=d/2, use_olh=True)

    def getPerhurb(self, data):
        return self.client.privatise(data)

    def getEncode(self, data, seed):
        return (xxhash.xxh32(str(self.client.index_mapper(data)),
                             seed=seed).intdigest() % self.client.g)

    def access(self):
        count = 0
        for i in range(self.n):
            data = random.randint(0, self.d)
            client_olh_perhurb = self.getPerhurb(data)
            client_olh_perhurb_data = client_olh_perhurb[0]
            client_olh_perhurb_seed = client_olh_perhurb[1]
            client_olh_encode = self.getEncode(data, client_olh_perhurb_seed)
            if client_olh_perhurb_data != client_olh_encode:
                count += 1
        q = count / self.n / (self.client.g - 1)
        # print(q, self.client.g)
        return math.log(1 / q - self.client.g + 1)
