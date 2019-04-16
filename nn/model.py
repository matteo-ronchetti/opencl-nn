import pyopencl as cl
import numpy as np

from .tensor import Tensor


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
