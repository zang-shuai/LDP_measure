import struct

import numpy as np


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


# np.random.laplace(0, 1, 1)
ans = [[[np.random.laplace(scale=(2 / 0.3), size=1)[0], ] for _ in range(50000)],
       [[np.random.laplace(loc=1, scale=(2 / 0.3), size=1)[0], ] for _ in range(50000)]]

write_bin(ans, 'm_file03.bin')
