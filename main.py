import numpy as np
import time

import nn
from nn.npops import np_conv, np_pad


def main():
    tot_images = 50000
    bs = 200
    n_filters = 1024

    x = nn.Tensor((bs, 3, 32, 32))

    w_1 = nn.Constant(np.random.randn(n_filters, 3, 5, 5).astype(np.float32))
    y = nn.conv(x, w_1)
    # y = nn.activation(y, lambda x: "exp(-({x}))")

    mdl = nn.compile_model(x, y)

    # s = time.time()
    npx = np.random.randn(bs, 3, 32, 32).astype(np.float32)
    # npy = np_pad(npx, 2)
    # npy = np_conv(npy, w_1.value)
    # e = time.time()
    # print("Numpy time", e - s)

    s = time.time()
    res = mdl(npx)
    e = time.time()
    print("OpenCL time", e - s)
    print(f"Expected time for {tot_images} is {(tot_images / bs) * (e - s)} seconds ({(tot_images / (bs * 60)) * (e - s)} minutes)")

    # print(np.max(np.abs(npy - res)))


main()
