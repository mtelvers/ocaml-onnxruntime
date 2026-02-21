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

  val create : Env.t -> ?threads:int -> ?cuda_device:int -> string -> t
  (** [create env ?threads ?cuda_device model_path] loads an ONNX model.
      [threads]: number of intra-op threads (0 = default).
      [cuda_device]: if provided, uses CUDA execution provider on the given device. *)

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
