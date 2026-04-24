"""Generate a zero-input ONNX model that computes pi via Leibniz series.

    pi ~= 4 * sum_{n=0}^{N-1} (-1)^n / (2n + 1)

The graph has no inputs and a single scalar float32 output `pi`.
All N terms are computed inside the graph (no pre-baked term tensor), so
this exercises Range, Mod, Div, ReduceSum, and scalar arithmetic ops.

Usage:  python3 test/gen_pi_model.py
"""

import os
import onnx
from onnx import helper, TensorProto

N = 10_000  # Leibniz err ~ 1/N, so ~1e-4 precision — tested with tol=5e-4.


def build():
    # Scalar initializers (0-d float tensors).
    def scalar(name, value):
        return helper.make_tensor(name, TensorProto.FLOAT, [], [value])

    c_zero = scalar("c_zero", 0.0)
    c_limit = scalar("c_limit", float(N))
    c_one = scalar("c_one", 1.0)
    c_two = scalar("c_two", 2.0)
    c_four = scalar("c_four", 4.0)
    c_neg_two = scalar("c_neg_two", -2.0)
    # ReduceSum in opset 13 takes `axes` as an input tensor, not attribute.
    c_axes = helper.make_tensor("c_axes", TensorProto.INT64, [1], [0])

    nodes = [
        # n = [0, 1, 2, ..., N-1]
        helper.make_node("Range", ["c_zero", "c_limit", "c_one"], ["n"]),
        # denom_i = 2*i + 1
        helper.make_node("Mul", ["n", "c_two"], ["two_n"]),
        helper.make_node("Add", ["two_n", "c_one"], ["denom"]),
        # sign_i = 1 - 2*(i mod 2)  →  +1 for even i, -1 for odd i
        helper.make_node("Mod", ["n", "c_two"], ["n_mod"], fmod=1),
        helper.make_node("Mul", ["n_mod", "c_neg_two"], ["neg_two_mod"]),
        helper.make_node("Add", ["neg_two_mod", "c_one"], ["sign"]),
        # term_i = sign_i / denom_i
        helper.make_node("Div", ["sign", "denom"], ["term"]),
        # sum = Σ term_i  (scalar)
        helper.make_node("ReduceSum", ["term", "c_axes"], ["sum"],
                         keepdims=0),
        # pi ≈ 4 * sum
        helper.make_node("Mul", ["sum", "c_four"], ["pi"]),
    ]

    output = helper.make_tensor_value_info("pi", TensorProto.FLOAT, [])
    graph = helper.make_graph(
        nodes=nodes,
        name="leibniz_pi",
        inputs=[],
        outputs=[output],
        initializer=[c_zero, c_limit, c_one, c_two, c_four,
                     c_neg_two, c_axes],
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
    out_path = os.path.join(os.path.dirname(__file__), "pi.onnx")
    onnx.save(build(), out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
