import struct

from LDP.frequency_ldp.frequency_oracles import UEClient, DEClient, HEClient
import numpy as np

from LDP.mean_ldp.duchi import Duchi
from LDP.mean_ldp.piecewisemechanism import PiecewiseMechanism
import subprocess
import time

from OB_measure_cpu.ob_cpu_continue import countLDP_continue
from OB_measure_cpu.ob_cpu_discrete import countLDP_discrete


def DE(epsilon, d):
    return DEClient(epsilon, d)


def OUE(epsilon, d):
    return UEClient(epsilon, d, use_oue=True)


def SUE(epsilon, d):
    return UEClient(epsilon, d)


def BLH(epsilon, d):
    return DEClient(epsilon, 2)


def OLH(epsilon, d):
    return DEClient(epsilon, d // 2)


def HE(epsilon, d):
    return HEClient(epsilon, d)


def DUCHI(epsilon, d):
    return Duchi(epsilon, d)


# def PiecewiseMechanism(epsilon, d):
#     return PiecewiseMechanism(epsilon, d)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


ldps = ['DE', 'OUE', 'SUE', 'BLH', 'OLH', 'DUCHI', 'PiecewiseMechanism', 'HE']
epsilons = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
ds = [64, 64, 64, 64, 64, [90, 110], [90, 110], 64]
Ds = [range(1, 64 + 1), range(1, 64 + 1), range(1, 64 + 1), range(1, 2 + 1), range(1, 32 + 1),
      range(90, 111), range(90, 111), range(1, 64 + 1)]
ks = [1, 64, 64, 1, 1, 1, 1, 64]
types = ['i', 'i', 'i', 'i', 'i', 'i', 'f', 'f']
end_times = []
for i in range(len(ldps)):
    hat_epsilon_all = []
    start_time = time.time()
    for epsilon in epsilons:
        hat_epsilons = []
        hat_epsilon = eval(ldps[i])(epsilon, ds[i])
        if types[i] == 'i':
            if isinstance(ds[i], int):
                hat_epsilons.append(countLDP_discrete(hat_epsilon).evaluation(10_0000))
            else:
                hat_epsilons.append(countLDP_discrete(hat_epsilon).evaluation(10_0000, 10))
        else:
            if isinstance(ds[i], int):
                hat_epsilons.append(countLDP_continue(hat_epsilon).evaluation(10_0000))
            else:
                hat_epsilons.append(countLDP_continue(hat_epsilon).evaluation(10_0000, 10))
        hat_epsilon_all.append(hat_epsilons)
    end_times.append((time.time() - start_time) / len(epsilons))
    print(end_times)
    np.savetxt('./data/ob_measure_cpu/' + ldps[i] + '.csv', np.array(hat_epsilon_all), delimiter=',')
    print()
np.savetxt('./data/ob_measure_cpu/time.csv', np.array(end_times), delimiter=',')
