import numpy as np


def np_conv(x, w):
    y = np.zeros((x.shape[0], w.shape[0], x.shape[2] - w.shape[2] + 1, x.shape[3] - w.shape[3] + 1), dtype=np.float32)

    for b in range(y.shape[0]):
        for d in range(y.shape[1]):
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    y[b, d, i, j] = np.sum(x[b, :, i:i + w.shape[2], j:j + w.shape[3]] * w[d])

    return y


def np_pad(x, p):
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="edge")
