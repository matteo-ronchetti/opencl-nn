import numpy as np
import pyopencl as cl

from libs.tensor import Tensor, Constant
from libs.ops import BaseOp, FillPadOp, ConvOp
from libs.npops import np_conv, np_pad


def conv(x: [Tensor, BaseOp], w, pad_type=""):
    assert len(w.shape) == 4 and w.shape[1] == x.shape[1] and w.shape[2] == w.shape[3]

    radius = w.shape[2] // 2

    x.output.set_pad((0, 0, radius, radius))

    y = FillPadOp(x, radius, pad_type)
    return ConvOp(y, w)


class Model:
    def __init__(self, ctx, queue, ops, x: Tensor, y: Tensor):
        self.ctx = ctx
        self.queue = queue
        self.ops = ops
        self.x = x
        self.y = y

        self.np_y = np.empty(y.shape, dtype=np.float32)

    def __call__(self, xx):
        self.x.upload(self.queue, xx)
        for _, kernel, args in self.ops:
            kernel(self.queue, *args)

        self.y.download(self.queue, self.np_y)

        return self.np_y


def compile_model(x: Tensor, y):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    ops = y.compile(ctx)
    ops.sort(key=lambda v: v[0], reverse=True)
    print(ops)

    return Model(ctx, queue, ops, x, y.output)


def main():
    bs = 10
    x = Tensor((bs, 3, 32, 32))

    w = Constant(np.random.randn(5, 3, 5, 5).astype(np.float32))
    y = conv(x, w)

    mdl = compile_model(x, y)

    npx = np.random.randn(bs, 3, 32, 32).astype(np.float32)
    npy = np_pad(npx, 2)
    npy = np_conv(npy, w.value)

    res = mdl(npx)

    print(np.max(np.abs(npy - res)))


main()
