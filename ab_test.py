import time

from AB_measure.ab_discrete import *
from AB_measure.ab_mean import *
import warnings

warnings.filterwarnings("ignore")

epsilons = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
ldps = ['ab_DE', 'ab_SUE', 'ab_OUE', 'ab_BLH', 'ab_OLH', 'ab_HE', 'ab_DUCHI', 'ab_PiecewiseMechanism']
ds = [64, 64, 64, 64, 64, 64, [90, 110], [90, 110]]
ks = [1, 64, 64, 1, 1, 64, 1, 1]
ns = []
for i in range(len(ks)):
    ns.append(100000 // ks[i])
end_times = []
for i in range(8):
    if i != 2:
        continue
    hat_epsilon_all = []
    start_time = time.time()
    for epsilon in epsilons:
        hat_epsilons = []
        for j in range(1):
            hat_epsilon = eval(ldps[i])(epsilon, ds[i], ns[i])
            hat_epsilons.append(hat_epsilon.access())
            print(j, end=' ')
        print(hat_epsilons)
        hat_epsilon_all.append(hat_epsilons)
    end_times.append((time.time() - start_time) / len(epsilons))
    np.savetxt('./data/ab_measure/' + ldps[i][3:] + '.csv', np.array(hat_epsilon_all), delimiter=',')
    print()
np.savetxt('./data/ab_measure/time.csv', np.array(end_times), delimiter=',')

# de = ab_SUE(0.1, 100, 1000)
# de = ab_OUE(0.1, 100, 1000)
# de = ab_BLH(0.1, 100, 100000)
# de = ab_OLH(0.1, 100, 100000)
# de = ab_HE(0.1, 100, 100000)
# de = ab_DUCHI(0.1, [50, 100], 100000)
# de = ab_PiecewiseMechanism(0.1, [50, 100], 100000)
# print(de.access())
