import pyopencl as cl
import numpy as np


class Tensor:
    def __init__(self, shape, value=None):
        self.shape = shape
        self.pad = (0,) * len(self.shape)
        self.value = value

        self.axis_size = self.get_axis_size()

        self.output = self
        self.allocated_buffer = None

    def byte_size(self):
        s = 4
        for d in self.shape:
            s *= d

        return s

    def allocate(self, ctx: cl.Context, read_only=False) -> cl.Buffer:
        if self.allocated_buffer:
            return self.allocated_buffer
        mf = cl.mem_flags

        flags = mf.READ_ONLY if read_only else mf.READ_WRITE

        if self.value is not None:
            self.allocated_buffer = cl.Buffer(ctx, flags | mf.COPY_HOST_PTR, hostbuf=self.value)
        else:
            self.allocated_buffer = cl.Buffer(ctx, flags, size=self.byte_size())

        return self.allocated_buffer

    def compile(self, ctx: cl.Context, _):
        self.allocate(ctx)
        # return no operation
        return []

    def get_axis_size(self):
        return np.cumprod([1] + list(self.shape[::-1]))[::-1][1:]

    def set_pad(self, pad):
        assert len(pad) == len(self.shape)

        self.pad = pad

        self.shape = [s + 2*p for s, p in zip(self.shape, pad)]
        self.axis_size = self.get_axis_size()

    def unpadded_shape(self):
        return tuple([s - 2 * p for s, p in zip(self.shape, self.pad)])

    def padded_at(self, name, indices):
        at = " + ".join([f"{s}*(({i.strip(' ')}) + {p})" for s, i, p in zip(self.axis_size, indices, self.pad)])

        return f"{name}[{at}]"

    def at(self, name, indices):
        at = " + ".join([f"{s}*({i.strip(' ')})" for s, i in zip(self.axis_size, indices)])

        return f"{name}[{at}]"

    def upload(self, queue, x):
        assert self.allocated_buffer is not None

        if any(self.pad):
            pad = tuple([(int(p), int(p)) for p in self.pad])
            x = np.pad(x, pad, mode="constant")

        print(self.shape, x.shape)
        cl.enqueue_copy(queue, self.allocated_buffer, x)

    def download(self, queue, x):
        assert self.allocated_buffer is not None

        cl.enqueue_copy(queue, x, self.allocated_buffer)


class Constant(Tensor):
    def __init__(self, value: np.ndarray):
        super().__init__(value.shape, value)
