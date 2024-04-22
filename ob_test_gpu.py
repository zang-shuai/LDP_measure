import struct

from LDP.frequency_ldp.frequency_oracles import UEClient, DEClient, HEClient
import numpy as np

from LDP.mean_ldp.duchi import Duchi
from LDP.mean_ldp.piecewisemechanism import PiecewiseMechanism
import subprocess
import time


def write_bin(three_d_list, filepath):
    with open(filepath, 'wb') as file:
        # Writing dimensions first
        file.write(struct.pack('i', len(three_d_list)))  # Number of 2D lists
        file.write(struct.pack('i', len(three_d_list[0])))  # Number of rows
        file.write(struct.pack('i', len(three_d_list[0][0])))  # Number of columns

        # Writing data
        for two_d_list in three_d_list:
            for row in two_d_list:
                for item in row:
                    if isinstance(item, int) or isinstance(item, np.int64):
                        file.write(struct.pack('i', int(item)))  # Writing each integer
                    elif isinstance(item, float):
                        file.write(struct.pack('d', item))  # Writing each integer
                    else:
                        print(type(item))
                        file.write(struct.pack('f', item))  # Writing each integer
    # print(filepath, '写入成功')


epsilons = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
# epsilons = [0.3, ]
ds = [64, 64, 64, 64, 64, [90, 110], [90, 110], 64]
types = ['i', 'i', 'i', 'i', 'i', 'f', 'f', 'f']
Ds = [range(1, 64 + 1), range(1, 64 + 1), range(1, 64 + 1), range(1, 2 + 1), range(1, 32 + 1),
      range(90, 111), range(90, 111), range(1, 64 + 1)]

ldps = ['DE', 'OUE', 'SUE', 'BLH', 'OLH', 'DUCHI', 'PiecewiseMechanism', 'HE']


def DE(epsilon, d):
    return DEClient(epsilon, d)


def OUE(epsilon, d):
    return UEClient(epsilon, d, use_oue=True)


def SUE(epsilon, d):
    return UEClient(epsilon, d)


def BLH(epsilon, d):
    return DEClient(epsilon, 2)


def OLH(epsilon, d):
    return DEClient(epsilon, d / 2)


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


end_times = []
for x in range(1):
    print('第 ' + str(x) + ' 次运行')
    # for i in range(8):
    for i in range(8):
        if i != 2:
            continue
        start_time = time.time()

        hat_epsilon_max = []
        hat_epsilon_avg = []
        for epsilon in epsilons:
            ldp = eval(ldps[i])(epsilon, ds[i])
            if is_iterable(ldp.privatise(Ds[i][0])):
                n = 100_000
                res = [[list(ldp.privatise(inp)) for _ in range(n)] for inp in Ds[i]]
            else:
                n = 100_000
                out = [ldp.privatise(0) for _ in range(n)]
                res = [[[ldp.privatise(input), ] for _ in range(n)] for input in Ds[i]]
            write_bin(res, 'file1.bin')
            print('算法', ldps[i], epsilon, '数据生成完成', len(res), len(res[0]), len(res[0][0]))

            # 调用可执行文件并捕获输出
            try:
                result = subprocess.run(
                    ['/home/zangshuai/project/LDP/cmake-build-debug/measure', 'file1.bin', types[i]],
                    stdout=subprocess.PIPE,
                    text=True, check=True)
                # exit(0)
                output = result.stdout
                return_code = result.returncode
                hat_epsilon_max.append(eval(output)[0])
                hat_epsilon_avg.append(eval(output)[1])
            except subprocess.CalledProcessError as e:
                print("错误:", e)
            print(hat_epsilon_max, hat_epsilon_avg)
        np.savetxt('./data/ob_measure_gpu/max/' + ldps[i] + '.csv', np.array(hat_epsilon_max), delimiter=',')
        np.savetxt('./data/ob_measure_gpu/avg/' + ldps[i] + '.csv', np.array(hat_epsilon_avg), delimiter=',')

        # with open('./data/ob_measure_gpu/avg/' + ldps[i] + '/' + str(int(epsilon * 10)) + '.txt', 'a') as file:
        #     file.write(str(hat_epsilon_avg) + ',')
        # with open('./data/ob_measure_gpu/max/' + ldps[i] + '/' + str(int(epsilon * 10)) + '.txt', 'a') as file:
        #     file.write(str(hat_epsilon_max) + ',')
        # with open('./data/ob_measure_gpu/time/' + ldps[i] + '/' + str(int(epsilon * 10)) + '.txt', 'a') as file:
        #     file.write(str(time.time() - start_time) + ',')
        print(time.time() - start_time, '算法', ldps[i], '计算完成', epsilon, hat_epsilon_avg, hat_epsilon_max)
        end_times.append((time.time() - start_time) / 6)

    # np.savetxt('./data/ob_measure_gpu/time.csv', np.array(end_times), delimiter=',')
