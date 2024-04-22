import struct

from LDP.frequency_ldp.frequency_oracles import UEClient, DEClient, HEClient
import numpy as np

from LDP.mean_ldp.duchi import Duchi
from LDP.mean_ldp.piecewisemechanism import PiecewiseMechanism


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
                        file.write(struct.pack('f', item))  # Writing each integer
    print(filepath, '写入成功')


def write_csv(three_d_list, filepath):
    with open(filepath, 'w') as file:
        for two_d_list in three_d_list:
            for row in two_d_list:
                file.write(','.join(map(str, row)) + '\n')
            file.write('\n')  # Separate each 2D list by a newline
    print(filepath, '写入成功')


def write(res, epsilon, name):
    if epsilon == 0.3:
        write_bin(res, '../../LDP/cmake-build-debug/data/bin/03/' + name + '.bin')
        write_csv(res, '../../LDP/cmake-build-debug/data/csv/03/' + name + '.csv')
    elif epsilon == 0.5:
        write_bin(res, '../../LDP/cmake-build-debug/data/bin/05/' + name + '.bin')
        write_csv(res, '../../LDP/cmake-build-debug/data/csv/05/' + name + '.csv')
    elif epsilon == 0.8:
        write_bin(res, '../../LDP/cmake-build-debug/data/bin/08/' + name + '.bin')
        write_csv(res, '../../LDP/cmake-build-debug/data/csv/08/' + name + '.csv')
    elif epsilon == 1.0:
        write_bin(res, '../../LDP/cmake-build-debug/data/bin/10/' + name + '.bin')
        write_csv(res, '../../LDP/cmake-build-debug/data/csv/10/' + name + '.csv')
    elif epsilon == 1.5:
        write_bin(res, '../../LDP/cmake-build-debug/data/bin/15/' + name + '.bin')
        write_csv(res, '../../LDP/cmake-build-debug/data/csv/15/' + name + '.csv')
    elif epsilon == 2.0:
        write_bin(res, '../../LDP/cmake-build-debug/data/bin/20/' + name + '.bin')
        write_csv(res, '../../LDP/cmake-build-debug/data/csv/20/' + name + '.csv')
    elif epsilon == 3.0:
        write_bin(res, '../../LDP/cmake-build-debug/data/bin/30/' + name + '.bin')
        write_csv(res, '../../LDP/cmake-build-debug/data/csv/30/' + name + '.csv')


epsilons = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
d = 64
n = 10000
D = range(1, d + 1)
c = [80, 120]

for epsilon in epsilons:
    sue = UEClient(epsilon, d, use_oue=False)
    res = []

    for di in D:
        print(di, end=' ')
        ans = []
        for i in range(n):
            ans.append(list(sue.privatise(di)))
        res.append(list(ans))
    print('')
    write(res, epsilon, 'SUE')

    oue = UEClient(epsilon, d, use_oue=True)
    res = []
    for di in D:
        print(di, end=' ')
        ans = []
        for i in range(n):
            ans.append(list(oue.privatise(di)))
        res.append(list(ans))
    print('')
    write(res, epsilon, 'OUE')

    rr = DEClient(epsilon, d)
    res = []
    for di in D:
        print(di, end=' ')
        ans = []
        for i in range(n):
            ans.append(list([rr.privatise(di), ]))
        res.append(list(ans))
    print('')
    write(res, epsilon, 'RR')

    # ds = DSClient(epsilon, d, 3)
    # res = []
    # for di in D:
    #     print(di, end=' ')
    #     ans = []
    #     for i in range(n):
    #         ans.append(list(ds.privatise(di)))
    #     res.append(list(ans))
    # print('')
    # write(res, epsilon, 'ds')

    pm = PiecewiseMechanism(epsilon, [c[0], c[1]])
    res = []
    for di in np.arange(c[0], c[1], 1):
        print(di, end=' ')
        ans = []
        for i in range(n):
            ans.append(list([pm.privatise([di, ]), ]))
        res.append(list(ans))
    print('')
    write(res, epsilon, 'pm')

    duchi = Duchi(epsilon, [c[0], c[1]])
    res = []
    for di in np.arange(c[0], c[1], 1):
        print(di, end=' ')
        ans = []
        for i in range(n):
            ans.append(list([duchi.privatise(di), ]))
        res.append(list(ans))
    print('')
    write(res, epsilon, 'duchi')

    he = HEClient(epsilon, d)
    res = []
    for di in D:
        print(di, end=' ')
        ans = []
        for i in range(n):
            ans.append(list(he.privatise(di)))
        res.append(list(ans))
    print('')
    write(res, epsilon, 'HE')

import subprocess

# 设置参数
param1 = '/home/zangshuai/project/LDP/cmake-build-debug/data/bin/01/BLH.bin'
param2 = 'i'

# 构建命令
command = ['./measure', param1, param2]

# 调用可执行文件并捕获输出
try:
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
    output = result.stdout
    return_code = result.returncode
    eval1 = eval(output)
    print("输出:", eval1)
except subprocess.CalledProcessError as e:
    print("错误:", e)
# ./measure /home/zangshuai/project/LDP/cmake-build-debug/data/bin/01/BLH.bin i
