from .model import Model, compile_model
from .tensor import Tensor, Constant
from .ops import BaseOp, FillPadOp, ConvOp


def conv(x: [Tensor, BaseOp], w, pad_type=""):
    assert len(w.output.shape) == 4 and w.output.shape[1] == x.output.shape[1] and w.output.shape[2] == w.output.shape[
        3]

    radius = w.output.shape[2] // 2

    x.output.set_pad((0, 0, radius, radius))

    y = FillPadOp(x, radius, pad_type)
    return ConvOp(y, w)


def activation(x: [Tensor, BaseOp], f):
    return ActivationOp(x, f)
