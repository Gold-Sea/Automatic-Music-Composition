import numpy as np

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



if __name__ == "__main__":
    
    x = []
    for i in range(100):
        x.append(i/100)
    x[49] = 0.
    x = np.array(x)
    x = np.reshape(x, (2,50))
    print(x)
    print(symmetry(x))