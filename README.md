# ocaml-onnxruntime

OCaml bindings to [ONNX Runtime](https://onnxruntime.ai/) via a thin C shim
and OCaml's native C FFI (`external` declarations). The C shim is compiled
into the library via dune `foreign_stubs` — no separate shared library to
deploy, and no runtime dependencies beyond ONNX Runtime itself.

Load an ONNX model, feed it float32 bigarrays, get float32 bigarrays back.

## Quick example

```ocaml
open Bigarray

let () =
  let env = Onnxruntime.Env.create ~log_level:2 "my_app" in
  let session = Onnxruntime.Session.create env ~threads:4 "model.onnx" in

  (* Prepare a float32 input tensor, flattened to 1-D *)
  let input = Array1.create float32 c_layout (1 * 40 * 11) in
  Array1.fill input 0.0;

  let outputs =
    Onnxruntime.Session.run_ba session
      [| ("input_name", input, [| 1L; 40L; 11L |]) |]
      [| "output_name" |]
      ~output_sizes:[| 128 |]
  in
  let output = outputs.(0) in
  Printf.printf "output[0] = %f\n" output.{0}
```

## API

```ocaml
exception Ort_error of string

module Env : sig
  type t
  val create : ?log_level:int -> string -> t
  (* log_level: 0=Verbose  1=Info  2=Warning  3=Error (default)  4=Fatal *)
end

module Session : sig
  type t
  val create :
    Env.t ->
    ?threads:int ->
    ?inter_op_threads:int ->
    ?parallel:bool ->
    ?cuda_device:int ->
    ?cuda_graph:bool ->
    string -> t
  val run_ba :
    t ->
    (string * (float, float32_elt, c_layout) Array1.t * int64 array) array ->
    string array ->
    output_sizes:int array ->
    (float, float32_elt, c_layout) Array1.t array
  val run_cached_ba :
    t ->
    ((float, float32_elt, c_layout) Array1.t * int64 array) array ->
    output_sizes:int array ->
    (float, float32_elt, c_layout) Array1.t array
end

module Io_binding : sig
  type t
  val create : Session.t -> t
  val bind_input  : t -> string -> (float, float32_elt, c_layout) Array1.t -> int64 array -> unit
  val bind_output : t -> string -> (float, float32_elt, c_layout) Array1.t -> int64 array -> unit
  val run : t -> unit
  val clear_inputs  : t -> unit
  val clear_outputs : t -> unit
end
```

`run_ba` takes an array of `(name, flat_data, shape)` input triples, an array
of output names, and the expected flat size of each output. It returns one
bigarray per output.

`run_cached_ba` is an optimised variant that uses input/output names cached
on the C side at session creation time, avoiding string marshalling overhead
in hot loops.

### Session options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~threads` | `0` (ORT default) | Intra-op parallelism threads |
| `~inter_op_threads` | `0` (ORT default) | Inter-op parallelism threads |
| `~parallel` | `false` | Set `ORT_PARALLEL` execution mode — runs independent graph nodes concurrently |
| `~cuda_device` | none | Enable CUDA execution provider on the given device id |
| `~cuda_graph` | `false` | When CUDA is enabled, capture the inference graph on GPU so subsequent runs skip CPU-side launch overhead |

### IO binding

`Io_binding` lets you bind pre-allocated bigarrays as inputs and outputs once,
then call `run` repeatedly. ORT reads input data from and writes results
directly into the bound bigarrays, avoiding per-call tensor creation and name
resolution. This is the fastest inference path, especially when combined with
CUDA graph capture.

```ocaml
let binding = Onnxruntime.Io_binding.create session in
Onnxruntime.Io_binding.bind_input  binding "input"  in_ba  [| 1L; 3L; 224L; 224L |];
Onnxruntime.Io_binding.bind_output binding "output" out_ba [| 1L; 1000L |];

(* Hot loop — update in_ba in-place, call run, read out_ba *)
for _ = 1 to 1000 do
  (* ... fill in_ba with new data ... *)
  Onnxruntime.Io_binding.run binding
  (* out_ba now contains the results *)
done
```

Use `clear_inputs` / `clear_outputs` to unbind and rebind to different
bigarrays if needed.

## Prerequisites

**ONNX Runtime shared library** (headers + `libonnxruntime.so`) must be
installed where the C compiler can find them. The build expects them in
`/usr/local/{include,lib}`.

On Ubuntu/Debian:

```bash
# Download and install ONNX Runtime 1.24.x (adjust version as needed)
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-1.24.1.tgz
tar xf onnxruntime-linux-x64-1.24.1.tgz
sudo cp onnxruntime-linux-x64-1.24.1/lib/* /usr/local/lib/
sudo cp onnxruntime-linux-x64-1.24.1/include/* /usr/local/include/
sudo ldconfig
```

**OCaml 5.1+** with opam:

```bash
opam install alcotest
```

## Build

```bash
dune build
```

At runtime the library needs to find `libonnxruntime.so`. If it is not on
the default linker path:

```bash
LD_LIBRARY_PATH=/usr/local/lib dune exec myapp/main.exe
```

## Tests

The test suite uses [Alcotest](https://github.com/mirage/alcotest) and
requires a Tessera ONNX model file (see next section for how to produce one).

```bash
TESSERA_MODEL=/path/to/tessera_model.onnx \
  LD_LIBRARY_PATH=/usr/local/lib \
  dune exec test/main.exe
```

The suite covers numerical correctness against known reference values, error
paths (bad model path, wrong input name, wrong tensor shape), determinism
across repeated runs, multiple concurrent sessions, and environment log
levels.

## Exporting a PyTorch model to ONNX

The bindings work with any ONNX model. The instructions below show the
concrete workflow for the Tessera geospatial embedding model; adapt as needed
for your own architecture.

### 1. Install Python dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install torch onnx onnxruntime numpy
```

### 2. Export

The `export_onnx.py` script rebuilds the model architecture in pure PyTorch,
loads a checkpoint, and calls `torch.onnx.export`:

```bash
python export_onnx.py \
  --checkpoint path/to/best_model.pt \
  --output tessera_model.onnx
```

This produces a `tessera_model.onnx` (and possibly a `.onnx.data` file for
the weights).

The key parts of the export call are:

```python
torch.onnx.export(
    model,
    (dummy_s2, dummy_s1),           # example inputs for tracing
    "tessera_model.onnx",
    input_names=["s2_input", "s1_input"],
    output_names=["output"],
    dynamic_axes={                   # allow variable batch size
        "s2_input":  {0: "batch"},
        "s1_input":  {0: "batch"},
        "output":    {0: "batch"},
    },
    opset_version=18,
)
```

The exported model expects:

| Input       | Shape            | Description                                    |
|-------------|------------------|------------------------------------------------|
| `s2_input`  | `[batch, 40, 11]`| 10 Sentinel-2 bands + day-of-year per timestep |
| `s1_input`  | `[batch, 40, 3]` | 2 Sentinel-1 bands + day-of-year per timestep  |

| Output      | Shape            | Description                  |
|-------------|------------------|------------------------------|
| `output`    | `[batch, 128]`   | 128-d embedding per sample   |

### 3. Validate (optional)

`validate_onnx.py` runs the same inputs through both PyTorch and ONNX
Runtime and checks that outputs agree within tolerance:

```bash
python validate_onnx.py \
  --checkpoint path/to/best_model.pt \
  --onnx tessera_model.onnx
```

### 4. Call from OCaml

```ocaml
open Bigarray

let () =
  let env = Onnxruntime.Env.create ~log_level:3 "tessera" in
  let session = Onnxruntime.Session.create env ~threads:4 "tessera_model.onnx" in

  (* Build input tensors — 1 sample, 40 timesteps *)
  let s2 = Array1.create float32 c_layout (1 * 40 * 11) in
  let s1 = Array1.create float32 c_layout (1 * 40 * 3) in
  (* ... fill s2 and s1 with normalised satellite data + day-of-year ... *)

  let outputs =
    Onnxruntime.Session.run_ba session
      [| ("s2_input", s2, [| 1L; 40L; 11L |]);
         ("s1_input", s1, [| 1L; 40L; 3L |]) |]
      [| "output" |]
      ~output_sizes:[| 128 |]
  in
  Printf.printf "embedding dim = %d\n" (Array1.dim outputs.(0))
```

### Adapting for your own model

The export pattern works for any PyTorch model:

1. Instantiate your model and call `model.eval()`.
2. Create dummy inputs with the right shapes.
3. Call `torch.onnx.export` with named inputs/outputs and `dynamic_axes` for
   any dimension that should be variable at inference time.
4. On the OCaml side, pass flat float32 bigarrays with the corresponding
   names and shapes to `Session.run_ba`.

## Project structure

```
src/
  lib/
    onnxruntime.ml[i]   OCaml API (Env, Session, Io_binding, Ort_error)
    ort_stubs.c          OCaml C FFI stubs
    ort_shim.c/.h        C shim over the ONNX Runtime C API
  c/
    ort_shim.c/.h        Reference copy of the C sources
test/
  main.ml                Alcotest test suite
```

## License

MIT
