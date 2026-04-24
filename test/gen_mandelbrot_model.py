"""Generate a zero-input ONNX model that computes the Mandelbrot set.

The graph produces a [H, W] float32 tensor whose value at each cell is the
number of iterations the sequence z_{n+1} = z_n^2 + c stayed inside the
disk |z| <= 2 (capped at MAX_ITER). Points in the set reach the cap;
points outside drop fast.

Complex arithmetic on reals: z = x + iy, so
    x' = x^2 - y^2 + cx
    y' = 2*x*y  + cy
    |z|^2 = x^2 + y^2
The iteration is unrolled MAX_ITER times (no Loop op needed); the grid of
c values is baked in as initializers.

Usage:  python3 test/gen_mandelbrot_model.py
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

H = 32
W = 32
MAX_ITER = 50
XMIN, XMAX = -2.0, 0.5
YMIN, YMAX = -1.25, 1.25


def build():
    x_axis = np.linspace(XMIN, XMAX, W, dtype=np.float32)
    y_axis = np.linspace(YMIN, YMAX, H, dtype=np.float32)
    cx_arr = np.broadcast_to(x_axis[None, :], (H, W)).astype(np.float32).copy()
    cy_arr = np.broadcast_to(y_axis[:, None], (H, W)).astype(np.float32).copy()

    cx_init = numpy_helper.from_array(cx_arr, name="cx")
    cy_init = numpy_helper.from_array(cy_arr, name="cy")
    zero_init = numpy_helper.from_array(
        np.zeros((H, W), dtype=np.float32), name="zero")
    four_init = numpy_helper.from_array(
        np.array(4.0, dtype=np.float32), name="four")
    two_init = numpy_helper.from_array(
        np.array(2.0, dtype=np.float32), name="two")

    nodes = []
    x_name = "zero"
    y_name = "zero"
    count_name = "zero"

    for i in range(MAX_ITER):
        x2 = f"x2_{i}"
        y2 = f"y2_{i}"
        mag2 = f"mag2_{i}"
        bnd_b = f"bndb_{i}"
        bnd = f"bnd_{i}"
        cnt = f"cnt_{i}"
        xyd = f"xyd_{i}"
        nx = f"nx_{i}"
        xy = f"xy_{i}"
        txy = f"txy_{i}"
        ny = f"ny_{i}"

        nodes += [
            helper.make_node("Mul", [x_name, x_name], [x2]),
            helper.make_node("Mul", [y_name, y_name], [y2]),
            helper.make_node("Add", [x2, y2], [mag2]),
            helper.make_node("LessOrEqual", [mag2, "four"], [bnd_b]),
            helper.make_node("Cast", [bnd_b], [bnd], to=TensorProto.FLOAT),
            helper.make_node("Add", [count_name, bnd], [cnt]),
            helper.make_node("Sub", [x2, y2], [xyd]),
            helper.make_node("Add", [xyd, "cx"], [nx]),
            helper.make_node("Mul", [x_name, y_name], [xy]),
            helper.make_node("Mul", [xy, "two"], [txy]),
            helper.make_node("Add", [txy, "cy"], [ny]),
        ]
        count_name = cnt
        x_name = nx
        y_name = ny

    nodes.append(helper.make_node("Identity", [count_name], ["count"]))

    out = helper.make_tensor_value_info(
        "count", TensorProto.FLOAT, [H, W])
    graph = helper.make_graph(
        nodes=nodes,
        name="mandelbrot",
        inputs=[],
        outputs=[out],
        initializer=[cx_init, cy_init, zero_init, four_init, two_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
        producer_name="ocaml-onnxruntime-tests",
    )
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "mandelbrot.onnx")
    onnx.save(build(), out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
