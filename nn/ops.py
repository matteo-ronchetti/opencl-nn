import numpy as np
import pyopencl as cl
from abc import ABC, abstractmethod

from nn.tensor import Tensor
from nn.kernel import read_compile


class BaseOp(ABC):
    def __init__(self):
        self.inputs = []
        self.output = None

    def compile(self, ctx: cl.Context, depth=0):
        res = []

        print(self, self.inputs)
        for inpt in self.inputs:
            res += inpt.compile(ctx, depth + 1)

        res.append([depth] + self._compile(ctx))

        return res

    @abstractmethod
    def _compile(self, ctx: cl.Context):
        pass


class FillPadOp(BaseOp):
    def __init__(self, x, r, pad_type):
        super().__init__()
        self.x = x
        self.r = r
        self.pad_type = pad_type

        self.inputs = [x]
        self.output = x.output

    def _compile(self, ctx):
        x = self.x.output.allocate(ctx)

        tensors = {
            "X": self.x.output,
            "P": Tensor((0, 0, self.r, self.r))
        }

        kernel = read_compile(ctx, "kernels/padding.c", tensors).edge_padding

        return [kernel, [(self.x.output.shape[3], self.x.output.shape[2], self.x.output.shape[0]), None, x]]


class ConvOp(BaseOp):
    def __init__(self, x, w):
        super().__init__()
        self.x = x
        self.w = w

        x_shape = x.output.shape
        w_shape = w.output.shape

        self.y = Tensor((x_shape[0], w_shape[0], x_shape[2] - w_shape[2] + 1, x_shape[3] - w_shape[3] + 1))

        self.inputs = [x, w]
        self.output = self.y

    def _compile(self, ctx):
        x = self.x.output.allocate(ctx, read_only=True)
        w = self.w.output.allocate(ctx, read_only=True)
        y = self.y.output.allocate(ctx)

        tensors = {
            "X": self.x.output,
            "W": self.w.output,
            "Y": self.y.output
        }

        operations = {
            "conv_op": lambda p: f"{p[0]} * {p[1]}",
            "final_op": lambda p: f"{p[0]}"
        }

        kernel = read_compile(ctx, "kernels/convolution.c", tensors, operations).convolution

        y_shape = self.y.output.unpadded_shape()

        return [kernel, [(y_shape[3], y_shape[2], y_shape[0]), (8, 8, 1), x, w, y]]


class ActivationOp(BaseOp):
    def __init__(self, x, f):
        super().__init__()
        self.x = x
        self.f = f

        self.inputs = [x]
        self.output = x.output

    def _compile(self, ctx):
        x = self.x.output.allocate(ctx)

        tensors = {
            "X": self.x.output
        }

        operations = {
            "f": self.f,
        }

        kernel = read_compile(ctx, "kernels/activation.c", tensors, operations).activation

        x_shape = self.x.output.unpadded_shape()

        return [kernel, [(x_shape[3], x_shape[2], x_shape[0]), None, x]]
