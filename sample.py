import numpy as np
import pyopencl as cl
import time

from nn.tensor import Constant, Tensor
from nn.kernel import SourceCode

"ami-0e0a1f474ff96ec0a"

def main():
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

    print(source)

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    prg = cl.Program(ctx, source).build()

    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    prg.edge_padding(queue, (x.shape[3], x.shape[2], x.shape[0]), None, x_g)
    cl.enqueue_copy(queue, cl_x, x_g)

    print(np.max(cl_x - np_y))

    print(cl_x[0, 0, -5:, -5:])
    print(np_y[0, 0, -5:, -5:])


main()

#
# a_np = np.random.rand(50000000).astype(np.float32)
# b_np = np.random.rand(50000000).astype(np.float32)
#
# a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
# b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
#
# prg = cl.Program(ctx, """
# __kernel void sum(
#     __global const float *a_g, __global const float *b_g, __global float *res_g)
# {
#   int gid = get_global_id(0);
#   res_g[gid] = a_g[gid] + b_g[gid];
# }
# """).build()
#
# res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
#
# s = time.time()
# prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
# cl.enqueue_barrier(queue)
# e = time.time()
#
# print(e - s)
#
# res_np = np.empty_like(a_np)
# cl.enqueue_copy(queue, res_np, res_g)
#
# print(np.linalg.norm(res_np - (a_np + b_np)))
