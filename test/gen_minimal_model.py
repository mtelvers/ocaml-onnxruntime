"""Generate a minimal ONNX model for the test suite.

Produces `minimal.onnx` alongside this script: a single Add op with two
float32 inputs `a` and `b` of shape [3] and output `y = a + b`.

Usage:  python3 test/gen_minimal_model.py
"""

import os
import onnx
from onnx import helper, TensorProto


def build():
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])

    node = helper.make_node("Add", ["a", "b"], ["y"])
    graph = helper.make_graph([node], "minimal", [a, b], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
        producer_name="ocaml-onnxruntime-tests",
    )
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "minimal.onnx")
    onnx.save(build(), out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
