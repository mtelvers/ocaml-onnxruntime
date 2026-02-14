open Ctypes
open Foreign

(* ------------------------------------------------------------------ *)
(* Low-level C shim bindings via ctypes Foreign (dlopen)              *)
(* ------------------------------------------------------------------ *)

(* All opaque ORT types are represented as [void] pointers. *)
type ort_env = unit ptr
type ort_session_options = unit ptr
type ort_session = unit ptr
type ort_memory_info = unit ptr
type ort_allocator = unit ptr
type ort_value = unit ptr
type ort_status = unit ptr

let ort_env : ort_env typ = ptr void
let ort_session_options : ort_session_options typ = ptr void
let ort_session : ort_session typ = ptr void
let ort_memory_info : ort_memory_info typ = ptr void
let ort_allocator : ort_allocator typ = ptr void
let ort_value : ort_value typ = ptr void
let ort_status : ort_status typ = ptr void

(* Load the C shim shared library. It links against libonnxruntime.so.
   Search order: ORT_SHIM_PATH env var, then libert_shim.so via LD_LIBRARY_PATH. *)
let () =
  let paths = [
    Sys.getenv_opt "ORT_SHIM_PATH";
    Some "libert_shim.so";
  ] in
  let rec try_load = function
    | [] -> failwith "Cannot load libert_shim.so -- set ORT_SHIM_PATH or LD_LIBRARY_PATH"
    | None :: rest -> try_load rest
    | Some p :: rest ->
      (try
         let _ = Dl.dlopen ~filename:p ~flags:[Dl.RTLD_NOW; Dl.RTLD_GLOBAL] in ()
       with Dl.DL_error _ -> try_load rest)
  in
  try_load paths

(* Bind C shim functions *)
let c_ort_init =
  foreign "ort_init" (void @-> returning int)

let c_ort_status_message =
  foreign "ort_status_message" (ort_status @-> returning string)

let c_ort_release_status =
  foreign "ort_release_status" (ort_status @-> returning void)

let c_ort_create_env =
  foreign "ort_create_env"
    (int @-> string @-> ptr ort_env @-> returning ort_status)

let c_ort_release_env =
  foreign "ort_release_env" (ort_env @-> returning void)

let c_ort_create_session_options =
  foreign "ort_create_session_options"
    (ptr ort_session_options @-> returning ort_status)

let c_ort_set_intra_op_threads =
  foreign "ort_set_intra_op_threads"
    (ort_session_options @-> int @-> returning ort_status)

let c_ort_set_graph_opt_level =
  foreign "ort_set_graph_opt_level"
    (ort_session_options @-> int @-> returning ort_status)

let c_ort_release_session_options =
  foreign "ort_release_session_options"
    (ort_session_options @-> returning void)

let c_ort_create_session =
  foreign "ort_create_session"
    (ort_env @-> string @-> ort_session_options @-> ptr ort_session
     @-> returning ort_status)

let c_ort_release_session =
  foreign "ort_release_session" (ort_session @-> returning void)

let c_ort_get_allocator =
  foreign "ort_get_allocator" (ptr ort_allocator @-> returning ort_status)

let c_ort_session_input_name =
  foreign "ort_session_input_name"
    (ort_session @-> size_t @-> ort_allocator @-> ptr (ptr char)
     @-> returning ort_status)

let c_ort_session_output_name =
  foreign "ort_session_output_name"
    (ort_session @-> size_t @-> ort_allocator @-> ptr (ptr char)
     @-> returning ort_status)

let c_ort_session_input_count =
  foreign "ort_session_input_count"
    (ort_session @-> ptr size_t @-> returning ort_status)

let c_ort_session_output_count =
  foreign "ort_session_output_count"
    (ort_session @-> ptr size_t @-> returning ort_status)

let c_ort_allocator_free =
  foreign "ort_allocator_free"
    (ort_allocator @-> ptr void @-> returning ort_status)

let c_ort_create_cpu_memory_info =
  foreign "ort_create_cpu_memory_info"
    (ptr ort_memory_info @-> returning ort_status)

let c_ort_release_memory_info =
  foreign "ort_release_memory_info" (ort_memory_info @-> returning void)

let c_ort_create_tensor_float =
  foreign "ort_create_tensor_float"
    (ort_memory_info @-> ptr float @-> size_t @-> ptr int64_t @-> size_t
     @-> ptr ort_value @-> returning ort_status)

let c_ort_release_value =
  foreign "ort_release_value" (ort_value @-> returning void)

let c_ort_get_tensor_float_data =
  foreign "ort_get_tensor_float_data"
    (ort_value @-> ptr (ptr float) @-> returning ort_status)

(* Use ptr (ptr char) instead of ptr string to avoid ctypes string type
   marshalling, which allocates C strings that can be GC'd while the
   pointer array still references them. *)
let c_ort_run =
  foreign "ort_run"
    (ort_session @-> ptr (ptr char) @-> ptr ort_value @-> size_t
     @-> ptr (ptr char) @-> size_t @-> ptr ort_value @-> returning ort_status)

(* Cached-names variants: names are cached on the C side so we never
   need to pass string pointers through ctypes during hot loops. *)
let c_ort_cache_session_names =
  foreign "ort_cache_session_names"
    (ort_session @-> returning ort_status)

let c_ort_run_cached =
  foreign "ort_run_cached"
    (ort_session @-> ptr ort_value @-> size_t
     @-> ptr ort_value @-> size_t @-> returning ort_status)

(* ------------------------------------------------------------------ *)
(* Error handling                                                     *)
(* ------------------------------------------------------------------ *)

exception Ort_error of string

let check_status (s : ort_status) =
  if not (is_null s) then begin
    let msg = c_ort_status_message s in
    c_ort_release_status s;
    raise (Ort_error msg)
  end

let () =
  let rc = c_ort_init () in
  if rc <> 0 then raise (Ort_error "Failed to initialise ONNX Runtime API")

(* ------------------------------------------------------------------ *)
(* C string helpers                                                    *)
(* ------------------------------------------------------------------ *)

(* Allocate a null-terminated C string from an OCaml string.
   Returns a CArray.t that keeps the memory alive as long as it is
   reachable from the OCaml heap. *)
let make_c_string s =
  let len = String.length s in
  let buf = CArray.make char (len + 1) in
  for i = 0 to len - 1 do
    CArray.set buf i s.[i]
  done;
  CArray.set buf len '\000';
  buf

(* ------------------------------------------------------------------ *)
(* High-level API                                                     *)
(* ------------------------------------------------------------------ *)

module Env = struct
  type t = { ptr : ort_env }

  let create ?(log_level = 3) logid =
    let out = allocate ort_env null in
    check_status (c_ort_create_env log_level logid out);
    let env = !@out in
    Gc.finalise (fun e -> c_ort_release_env e.ptr) { ptr = env };
    { ptr = env }
end

module Session = struct
  type t = {
    ptr : ort_session;
    input_names : string array;
    output_names : string array;
  }

  let get_names session_ptr count_fn name_fn =
    let alloc_out = allocate ort_allocator null in
    check_status (c_ort_get_allocator alloc_out);
    let alloc = !@alloc_out in
    let count_out = allocate size_t Unsigned.Size_t.zero in
    check_status (count_fn session_ptr count_out);
    let n = Unsigned.Size_t.to_int (!@count_out) in
    let names = Array.init n (fun i ->
      let name_out = allocate (ptr char) (from_voidp char null) in
      check_status (name_fn session_ptr (Unsigned.Size_t.of_int i) alloc name_out);
      let name_ptr = !@name_out in
      let name = coerce (ptr char) string name_ptr in
      (* Free the string allocated by ORT *)
      let _ = c_ort_allocator_free alloc (to_voidp name_ptr) in
      name
    ) in
    names

  let create (env : Env.t) ?(threads = 0) ?cuda_device:_ model_path =
    let opts_out = allocate ort_session_options null in
    check_status (c_ort_create_session_options opts_out);
    let opts = !@opts_out in
    if threads > 0 then
      check_status (c_ort_set_intra_op_threads opts threads);
    (* ORT_ENABLE_ALL = 99 *)
    check_status (c_ort_set_graph_opt_level opts 99);
    let sess_out = allocate ort_session null in
    check_status (c_ort_create_session env.ptr model_path opts sess_out);
    c_ort_release_session_options opts;
    let sess_ptr = !@sess_out in
    let input_names =
      get_names sess_ptr c_ort_session_input_count c_ort_session_input_name
    in
    let output_names =
      get_names sess_ptr c_ort_session_output_count c_ort_session_output_name
    in
    let t = { ptr = sess_ptr; input_names; output_names } in
    (* Cache names on the C side for ort_run_cached *)
    check_status (c_ort_cache_session_names sess_ptr);
    Gc.finalise (fun s -> c_ort_release_session s.ptr) t;
    t

  let run (session : t) inputs output_names =
    let n_inputs = List.length inputs in
    let n_outputs = List.length output_names in
    (* Create memory info for CPU tensors *)
    let mem_out = allocate ort_memory_info null in
    check_status (c_ort_create_cpu_memory_info mem_out);
    let mem_info = !@mem_out in
    (* Allocate C strings for input/output names manually.
       We keep explicit references to the CArray buffers so the GC
       cannot free the underlying C memory during the c_ort_run call. *)
    let input_c_strings = List.map (fun (name, _data, _shape) ->
      make_c_string name) inputs in
    let output_c_strings = List.map make_c_string output_names in
    (* Build pointer arrays: char*[] for names *)
    let input_name_ptrs = CArray.make (ptr char) n_inputs in
    List.iteri (fun i cs ->
      CArray.set input_name_ptrs i (CArray.start cs)
    ) input_c_strings;
    let output_name_ptrs = CArray.make (ptr char) n_outputs in
    List.iteri (fun i cs ->
      CArray.set output_name_ptrs i (CArray.start cs)
    ) output_c_strings;
    (* Build input tensors *)
    let input_val_arr = CArray.make ort_value n_inputs in
    let tensor_ptrs = ref [] in
    let shape_arrs = ref [] in
    List.iteri (fun i (_name, data, shape) ->
      let data_len = Bigarray.Array1.dim data in
      let data_ptr = bigarray_start array1 data in
      let ndims = Array.length shape in
      let shape_arr = CArray.make int64_t ndims in
      Array.iteri (fun j v -> CArray.set shape_arr j v) shape;
      shape_arrs := shape_arr :: !shape_arrs;
      let val_out = allocate ort_value null in
      check_status (c_ort_create_tensor_float
        mem_info (coerce (ptr float) (ptr float) data_ptr)
        (Unsigned.Size_t.of_int data_len)
        (CArray.start shape_arr) (Unsigned.Size_t.of_int ndims)
        val_out);
      CArray.set input_val_arr i (!@val_out);
      tensor_ptrs := !@val_out :: !tensor_ptrs
    ) inputs;
    (* Allocate output value array (NULLs — ORT will allocate) *)
    let output_val_arr = CArray.make ort_value n_outputs in
    for i = 0 to n_outputs - 1 do
      CArray.set output_val_arr i null
    done;
    (* Run *)
    check_status (c_ort_run session.ptr
      (CArray.start input_name_ptrs)
      (CArray.start input_val_arr)
      (Unsigned.Size_t.of_int n_inputs)
      (CArray.start output_name_ptrs)
      (Unsigned.Size_t.of_int n_outputs)
      (CArray.start output_val_arr));
    (* Prevent GC from collecting any of the above before c_ort_run returns *)
    ignore (Sys.opaque_identity input_c_strings);
    ignore (Sys.opaque_identity output_c_strings);
    ignore (Sys.opaque_identity input_name_ptrs);
    ignore (Sys.opaque_identity output_name_ptrs);
    ignore (Sys.opaque_identity input_val_arr);
    ignore (Sys.opaque_identity output_val_arr);
    ignore (Sys.opaque_identity !shape_arrs);
    ignore (Sys.opaque_identity inputs);
    (* Release input tensors *)
    List.iter c_ort_release_value !tensor_ptrs;
    c_ort_release_memory_info mem_info;
    (* Collect output OrtValues *)
    let results = List.init n_outputs (fun i ->
      CArray.get output_val_arr i
    ) in
    results

  (* run_cached: same as run but uses C-side cached names —
     avoids all ctypes string marshalling in the hot loop. *)
  let run_cached (session : t) inputs n_outputs =
    let n_inputs = List.length inputs in
    (* Create memory info for CPU tensors *)
    let mem_out = allocate ort_memory_info null in
    check_status (c_ort_create_cpu_memory_info mem_out);
    let mem_info = !@mem_out in
    (* Build input tensors *)
    let input_val_arr = CArray.make ort_value n_inputs in
    let tensor_ptrs = ref [] in
    let shape_arrs = ref [] in
    List.iteri (fun i (data, shape) ->
      let data_len = Bigarray.Array1.dim data in
      let data_ptr = bigarray_start array1 data in
      let ndims = Array.length shape in
      let shape_arr = CArray.make int64_t ndims in
      Array.iteri (fun j v -> CArray.set shape_arr j v) shape;
      shape_arrs := shape_arr :: !shape_arrs;
      let val_out = allocate ort_value null in
      check_status (c_ort_create_tensor_float
        mem_info (coerce (ptr float) (ptr float) data_ptr)
        (Unsigned.Size_t.of_int data_len)
        (CArray.start shape_arr) (Unsigned.Size_t.of_int ndims)
        val_out);
      CArray.set input_val_arr i (!@val_out);
      tensor_ptrs := !@val_out :: !tensor_ptrs
    ) inputs;
    (* Allocate output value array *)
    let output_val_arr = CArray.make ort_value n_outputs in
    for i = 0 to n_outputs - 1 do
      CArray.set output_val_arr i null
    done;
    (* Run using C-cached names *)
    check_status (c_ort_run_cached session.ptr
      (CArray.start input_val_arr)
      (Unsigned.Size_t.of_int n_inputs)
      (CArray.start output_val_arr)
      (Unsigned.Size_t.of_int n_outputs));
    (* Keep everything alive *)
    ignore (Sys.opaque_identity input_val_arr);
    ignore (Sys.opaque_identity output_val_arr);
    ignore (Sys.opaque_identity !shape_arrs);
    ignore (Sys.opaque_identity inputs);
    (* Release input tensors *)
    List.iter c_ort_release_value !tensor_ptrs;
    c_ort_release_memory_info mem_info;
    (* Collect output OrtValues *)
    List.init n_outputs (fun i -> CArray.get output_val_arr i)

  let run_ba (session : t)
      (inputs :
        (string *
         (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t *
         int64 array) list)
      (output_names : string list)
      ~(output_shapes : int array list)
    : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t list =
    let output_vals = run session inputs output_names in
    let results = List.map2 (fun oval shape ->
      let total = Array.fold_left ( * ) 1 shape in
      let data_out = allocate (ptr float) (from_voidp float null) in
      check_status (c_ort_get_tensor_float_data oval data_out);
      let src = !@data_out in
      (* Copy data out into a fresh bigarray *)
      let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout total in
      for j = 0 to total - 1 do
        Bigarray.Array1.set ba j !@(src +@ j)
      done;
      c_ort_release_value oval;
      ba
    ) output_vals output_shapes in
    results

  let run_cached_ba (session : t)
      (inputs :
        ((float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t *
         int64 array) list)
      ~(output_shapes : int array list)
    : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t list =
    let n_outputs = List.length output_shapes in
    let output_vals = run_cached session inputs n_outputs in
    let results = List.map2 (fun oval shape ->
      let total = Array.fold_left ( * ) 1 shape in
      let data_out = allocate (ptr float) (from_voidp float null) in
      check_status (c_ort_get_tensor_float_data oval data_out);
      let src = !@data_out in
      let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout total in
      for j = 0 to total - 1 do
        Bigarray.Array1.set ba j !@(src +@ j)
      done;
      c_ort_release_value oval;
      ba
    ) output_vals output_shapes in
    results
end
