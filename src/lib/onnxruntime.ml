exception Ort_error of string

let () = Callback.register_exception "Ort_error" (Ort_error "")

type ort_env
type ort_session

external ort_create_env : int -> string -> ort_env
  = "caml_ort_create_env"

external ort_create_session :
  ort_env -> int -> int option -> bool -> bool -> int -> string -> ort_session
  = "caml_ort_create_session_bytecode" "caml_ort_create_session"

external ort_session_input_names : ort_session -> string array
  = "caml_ort_session_input_names"

external ort_session_output_names : ort_session -> string array
  = "caml_ort_session_output_names"

external ort_run_ba :
  ort_session ->
  string array ->
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t array ->
  int64 array array ->
  string array ->
  int array ->
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t array
  = "caml_ort_run_ba_bytecode" "caml_ort_run_ba"

external ort_run_cached_ba :
  ort_session ->
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t array ->
  int64 array array ->
  int array ->
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t array
  = "caml_ort_run_cached_ba"

module Env = struct
  type t = ort_env

  let create ?(log_level = 3) logid =
    ort_create_env log_level logid
end

module Session = struct
  type t = {
    ptr : ort_session;
    env : Env.t;
    input_names : string array;
    output_names : string array;
  }

  let create (env : Env.t) ?(threads = 0) ?(inter_op_threads = 0)
      ?(parallel = false) ?cuda_device ?(cuda_graph = false) model_path =
    let ptr = ort_create_session env threads cuda_device cuda_graph
                parallel inter_op_threads model_path in
    let input_names = ort_session_input_names ptr in
    let output_names = ort_session_output_names ptr in
    { ptr; env; input_names; output_names }

  let run_ba session inputs output_names ~output_sizes =
    let n = Array.length inputs in
    let in_names  = Array.init n (fun i -> let (name, _, _) = inputs.(i) in name) in
    let in_bas    = Array.init n (fun i -> let (_, ba, _) = inputs.(i) in ba) in
    let in_shapes = Array.init n (fun i -> let (_, _, shape) = inputs.(i) in shape) in
    ort_run_ba session.ptr in_names in_bas in_shapes output_names output_sizes

  let run_cached_ba session inputs ~output_sizes =
    let n = Array.length inputs in
    let in_bas    = Array.init n (fun i -> fst inputs.(i)) in
    let in_shapes = Array.init n (fun i -> snd inputs.(i)) in
    ort_run_cached_ba session.ptr in_bas in_shapes output_sizes
end

type ort_io_binding

external ort_create_io_binding : ort_session -> ort_io_binding
  = "caml_ort_create_io_binding"

external ort_io_bind_input :
  ort_io_binding -> string ->
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int64 array -> unit
  = "caml_ort_io_bind_input"

external ort_io_bind_output :
  ort_io_binding -> string ->
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int64 array -> unit
  = "caml_ort_io_bind_output"

external ort_io_run : ort_session -> ort_io_binding -> unit
  = "caml_ort_io_run"

external ort_io_clear_inputs : ort_io_binding -> unit
  = "caml_ort_io_clear_inputs"

external ort_io_clear_outputs : ort_io_binding -> unit
  = "caml_ort_io_clear_outputs"

module Io_binding = struct
  type ba = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

  type t = {
    ptr : ort_io_binding;
    session : Session.t;
    mutable refs : ba list;
  }

  let create session =
    let ptr = ort_create_io_binding session.Session.ptr in
    { ptr; session; refs = [] }

  let bind_input t name ba shape =
    ort_io_bind_input t.ptr name ba shape;
    t.refs <- ba :: t.refs

  let bind_output t name ba shape =
    ort_io_bind_output t.ptr name ba shape;
    t.refs <- ba :: t.refs

  let run t =
    ort_io_run t.session.Session.ptr t.ptr

  let clear_inputs t =
    ort_io_clear_inputs t.ptr;
    t.refs <- []

  let clear_outputs t =
    ort_io_clear_outputs t.ptr;
    t.refs <- []
end
