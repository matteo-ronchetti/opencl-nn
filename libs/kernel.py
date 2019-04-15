import re
import pyopencl as cl

at_re = re.compile(r"([a-zA-Z]+)@\[([^\]]+)\]")
shape_re = re.compile(r"([a-zA-Z]+)@shape\[([0-9]+)\]")
ops_re = re.compile(r"\$([a-zA-Z_0-9]+){([^}]+)}")


class SourceCode:
    def __init__(self, path):
        with open(path, "r") as f:
            self.code = f.read()

    def render(self, tensors: dict, operations: dict = None):
        code = at_re.sub(lambda m: tensors[m.group(1)].at(m.group(1), m.group(2).split(",")), self.code)
        code = shape_re.sub(lambda m: str(tensors[m.group(1)].shape[int(m.group(2))]), code)

        if operations:
            code = ops_re.sub(lambda m: str(operations[m.group(1)](m.group(2).split(","))), code)

        return code


def read_compile(ctx, path: str, tensors: dict, operations: dict = None) -> cl.Kernel:
    source = SourceCode(path).render(tensors, operations)
    return cl.Program(ctx, source).build()
