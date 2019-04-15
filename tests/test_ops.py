from unittest import TestCase
import numpy as np
import pyopencl as cl

from libs.tensor import Constant, Tensor
from libs.kernel import SourceCode
from libs.npops import np_conv, np_pad


class TestOps(TestCase):
    def test_pad(self):
        x = np.random.randn(10, 5, 32, 32).astype(np.float32)
        padding = (0, 0, 2, 2)

        np_y = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
        x = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")
        cl_x = x.copy()

        tensors = {
            "X": Constant(x),
            "P": Tensor(padding)
        }

        source = SourceCode("kernels/padding.c").render(tensors)

        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        prg = cl.Program(ctx, source).build()

        x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

        prg.edge_padding(queue, (x.shape[3], x.shape[2], x.shape[0]), None, x_g)
        cl.enqueue_copy(queue, cl_x, x_g)

        self.assertLess(np.max(cl_x - np_y), 1e-4)

    def test_conv(self):
        x = np.random.randn(10, 3, 32, 32).astype(np.float32)
        w = np.random.randn(5, 3, 5, 5).astype(np.float32)

        np_y = np_conv(x, w)
        cl_y = np.empty(np_y.shape, dtype=np.float32)

        y = Tensor((10, 5, 28, 28))

        tensors = {
            "X": Constant(x),
            "W": Constant(w),
            "Y": y
        }

        operations = {
            "conv_op": lambda p: f"{p[0]} * {p[1]}",
            "final_op": lambda p: f"exp({p[0]})"
        }

        source = SourceCode("kernels/convolution.c").render(tensors, operations)

        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        prg = cl.Program(ctx, source).build()

        x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        w_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)

        y_g = cl.Buffer(ctx, mf.WRITE_ONLY, cl_y.nbytes)

        prg.convolution(queue, (y.shape[3], y.shape[2], y.shape[0]), None, x_g, w_g, y_g)
        cl.enqueue_copy(queue, cl_y, y_g)

        self.assertLess(np.max((cl_y - np.exp(np_y)) / np.exp(np_y)), 1e-4)
