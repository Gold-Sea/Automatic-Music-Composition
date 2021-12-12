import numpy as np
import IO
import json

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    num_tones = load_dict['num_tones']

course = {1:1, 2:1, 3:2, 4:2, 5:3, 6:4, 7:4, 8:5, 9:5, 10:6, 11:6, 0:7}

# These fitness func are all aimed to batch operate

def mean(arr):
    assert isinstance(arr, np.ndarray)
    return arr.mean(axis=1)

def var(arr):
    assert isinstance(arr, np.ndarray)
    return arr.var(axis=1)

def symmetry(arr):
    assert isinstance(arr, np.ndarray)
    ret = []
    length = arr.shape[1]
    # batch operate
    for i in arr:
        print(i)
        cnt = 0
        for j in range(length//2):
            if int(i[j]) == int(i[length - j - 1]):
                cnt += 1
        ret.append(cnt)
    return np.array(ret)

def l2_dis(arr):
    assert isinstance(arr, np.ndarray)
    x,_ = IO.read_files('./data/data.txt')
    x = np.array(x)
    res = []
    for i in arr:
        sco = 0
        for j in x:
            sco += 1/np.sqrt(np.sum(np.square(i - j)))
        res.append(sco)
    return np.array(res)

def knn(arr):
    assert isinstance(arr, np.ndarray)
    x,_ = IO.read_files('./data/data.txt')
    x = np.array(x)
    res = []
    for i in arr:
        sco = 0
        for j in x:
            sco += 1/np.sqrt(np.sum(np.square(i - j)))
        res.append(sco)
    return np.array(res)

def interval_eval(arr):
    assert isinstance(arr, np.ndarray)
    ret = []
    for i in arr:
        sco = 0
        ta = i.copy()
        index = []
        for k in range(len(ta)):
            if ta[k] == 0 or ta[k] == num_tones - 1:
                index.append(k)
        # delete 0 or 28
        ta = np.delete(ta, index)
        for j in range(len(i) - 1):
            a = i[j]
            b = i[j + 1]
            ca = course[a%12]
            cb = course[b%12]
            in_c = abs(ca -cb) + 1
            interval = abs(a - b)
            #高于八度
            if interval > 12:
                sco += 1
            #一度
            elif interval == 1:
                sco += 10
            #纯四度
            elif in_c == 4 and interval == 5:
                sco += 10
            #纯五度
            elif in_c == 5 and interval == 7:
                sco += 10
            #纯八度
            elif in_c == 0 and interval == 12:
                sco += 10
            #大六度
            elif in_c == 6 and interval == 9:
                sco += 6
            #小六度
            elif in_c == 6 and interval == 8:
                sco += 6
            #一个八度内其他的
            else:
                sco += 3
        ret.append(sco)
    return np.array(ret)




if __name__ == "__main__":
    
    # x = []
    # for i in range(100):
    #     x.append(i/100)
    # x[49] = 0.
    # x = np.array(x)
    # x = np.reshape(x, (2,50))
    # print(x)
    # print(symmetry(x))
    l2_dis(None)