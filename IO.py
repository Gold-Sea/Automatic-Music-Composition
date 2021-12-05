import numpy as np
import json
import re

from numpy.lib.function_base import append

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    ratio = load_dict['compress_ratio']
    bs = load_dict['batch_size']
    #data_file = load_dict['data_file']

def compress(arr):
    assert isinstance(arr, np.ndarray)
    return arr/ratio

def decompress(arr):
    assert isinstance(arr, np.ndarray)
    return np.array(arr*ratio, dtype=int)

def generate_bad(length):
    assert isinstance(length, int)
    tmp = np.random.randint(low=0, high=26, size=(length,))
    score = np.random.randint(low=0, high=10)
    return tmp, [int(score)]

def read_files(fname):
    return_list = []
    return2 = []
    with open(fname, 'r') as f:
        tmpstr = f.readline()
        while(tmpstr):
            tmplist = re.split(r'[\n\s]', tmpstr)
            t = []
            for i in tmplist[1:]:
                if i == '':
                    continue
                t.append(int(i))
            # print(t)
            return_list.append((np.array(t[:32])))
            return2.append([int(tmplist[0])])
            tmpstr = f.readline()
    return return_list, return2

def get_data(datafile = './data/data.txt'):
    x,y = read_files(datafile)
    for i in range(8):
        t1, t2 = generate_bad(32)
        x.append(t1)
        y.append(t2)
    per = np.random.permutation(range(len(x)))
    tmp_x = []
    tmp_y = []
    for i in per:
        tmp_x.append(x[i])
        tmp_y.append(y[i])
    x = tmp_x
    y = tmp_y
    for i in x:
        assert i.shape[0] == 32
    in_x = []
    ou_y = []
    num = len(y)
    batchs = num//bs
    if batchs*bs != num:
        batchs+=1
    for i in range(batchs):
        last = min((i+1)*bs, num)
        tmpx = x[i*bs:last]
        tmpy = y[i*bs:last]
        in_x.append(compress(np.array(tmpx)))
        ou_y.append(compress(np.array(tmpy)))
    # for i in in_x:
    #     print(i)
    # for j in ou_y:
    #     print(j)
    return in_x, ou_y

if __name__ == "__main__":
    # get_data()
    pass
    # x = []
    # for i in range(100):
    #     x.append(i)
    # x[49] = 0.
    # x = np.array(x)
    # x = np.reshape(x, (2,50))
    # print(x)
    # y = compress(x)
    # print(compress(x))
    # print(decompress(y))