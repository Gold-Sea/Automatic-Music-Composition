import numpy as np
import json

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    ratio = load_dict['compress_ratio']

def compress(arr):
    assert isinstance(arr, np.ndarray)
    return arr/ratio

def decompress(arr):
    assert isinstance(arr, np.ndarray)
    return np.array(arr*ratio, dtype=int)

if __name__ == "__main__":
    
    x = []
    for i in range(100):
        x.append(i)
    x[49] = 0.
    x = np.array(x)
    x = np.reshape(x, (2,50))
    print(x)
    y = compress(x)
    print(compress(x))
    print(decompress(y))