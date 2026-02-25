(** OCaml bindings to ONNX Runtime via C shim. *)

exception Ort_error of string
(** Raised when an ONNX Runtime operation fails. *)

module Env : sig
  type t

  val create : ?log_level:int -> string -> t
  (** [create ?log_level logid] creates an ONNX Runtime environment.
      [log_level]: 0=Verbose, 1=Info, 2=Warning, 3=Error (default), 4=Fatal. *)
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
    string ->
    t
  (** [create env ?threads ?inter_op_threads ?parallel ?cuda_device ?cuda_graph model_path]
      loads an ONNX model.
      [threads]: number of intra-op threads (0 = default).
      [inter_op_threads]: number of inter-op threads (0 = default).
      [parallel]: if [true], sets execution mode to [ORT_PARALLEL] so independent
        graph nodes run concurrently (default [false] = sequential).
      [cuda_device]: if provided, uses CUDA execution provider on the given device.
      [cuda_graph]: if [true] and CUDA is enabled, captures the inference graph on
        GPU so subsequent runs skip CPU-side launch overhead (default [false]). *)

  val run_ba :
    t ->
    (string *
     (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t *
     int64 array) array ->
    string array ->
    output_sizes:int array ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t array
  (** [run_ba session inputs output_names ~output_sizes] runs inference.
      Each input is [(name, flat_data, shape)] where [flat_data] is a float32
      bigarray and [shape] is the tensor shape as int64 array.
      [output_sizes] gives the flat element count of each output.
      Returns output tensors as flat float32 bigarrays. *)

  val run_cached_ba :
    t ->
    ((float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t *
     int64 array) array ->
    output_sizes:int array ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t array
  (** [run_cached_ba session inputs ~output_sizes] runs inference using
      C-side cached input/output names. Input/output names are determined
      from the model at session creation time. Each input is [(flat_data, shape)].
      This avoids passing string names on every call. *)
end

module Io_binding : sig
  type t

  val create : Session.t -> t
  (** [create session] creates an IO binding for the given session.
      Bind pre-allocated bigarrays as inputs/outputs, then call [run]
      repeatedly — ORT reads/writes the bigarray memory directly,
      avoiding per-call tensor creation and name resolution. *)

  val bind_input :
    t -> string ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    int64 array -> unit
  (** [bind_input t name data shape] binds a CPU float32 bigarray as a named
      input. The bigarray memory is used directly — update contents in-place
      before each [run] to supply new input data. *)

  val bind_output :
    t -> string ->
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    int64 array -> unit
  (** [bind_output t name data shape] binds a CPU float32 bigarray as a named
      output. ORT writes inference results directly into this buffer. *)

  val run : t -> unit
  (** Run inference using the current bindings. After this returns, output
      bigarrays contain the results. *)

  val clear_inputs : t -> unit
  (** Clear all input bindings. Call [bind_input] again to rebind. *)

  val clear_outputs : t -> unit
  (** Clear all output bindings. Call [bind_output] again to rebind. *)
end
