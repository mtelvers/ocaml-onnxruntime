(* Alcotest suite for ocaml-onnxruntime bindings.

   Requires TESSERA_MODEL env var pointing to the .onnx model file. *)

open Bigarray

let seq_len = 40
let s2_dim = 11
let s1_dim = 3
let output_dim = 128

let model_path =
  lazy
    (try Sys.getenv "TESSERA_MODEL"
     with Not_found ->
       Alcotest.fail
         "TESSERA_MODEL environment variable not set — \
          point it at tessera_model.onnx")

let make_env () = Onnxruntime.Env.create ~log_level:3 "test"

let make_input ~modulus ~dim batch =
  let n = batch * seq_len * dim in
  let ba = Array1.create float32 c_layout n in
  for i = 0 to n - 1 do
    ba.{i} <- Float.of_int (i mod modulus) *. 0.1
  done;
  ba

let make_s2_input = make_input ~modulus:7 ~dim:s2_dim
let make_s1_input = make_input ~modulus:5 ~dim:s1_dim

(* First 10 known-good output values from validated ONNX Runtime run *)
let reference_output =
  [| 4.105261; -7.083618; 1.867566; 6.506818; -1.581951;
     -3.145885; 1.267286; 0.232035; 0.818696; 0.240831 |]

let assert_close ~tol expected actual =
  let diff = Float.abs (expected -. actual) in
  if diff > tol then
    Alcotest.failf "expected %.6f, got %.6f (diff %.6e > tol %.6e)"
      expected actual diff tol

let assert_matches_reference output =
  Alcotest.(check int) "output length" output_dim (Array1.dim output);
  Array.iteri
    (fun i expected -> assert_close ~tol:1e-4 expected output.{i})
    reference_output

let run_inference session batch =
  let s2 = make_s2_input batch in
  let s1 = make_s1_input batch in
  let s2_shape = [| Int64.of_int batch; 40L; 11L |] in
  let s1_shape = [| Int64.of_int batch; 40L; 3L |] in
  let outputs =
    Onnxruntime.Session.run_ba session
      [| ("s2_input", s2, s2_shape); ("s1_input", s1, s1_shape) |]
      [| "output" |]
      ~output_sizes:[| batch * output_dim |]
  in
  match outputs with
  | [| out |] -> out
  | _ ->
    Alcotest.failf "expected 1 output, got %d" (Array.length outputs)

let with_session f =
  let env = make_env () in
  let session = Onnxruntime.Session.create env (Lazy.force model_path) in
  f session

(* Alcotest.check_raises requires an exact exception match; we just need
   to verify that *some* Ort_error is raised, regardless of message. *)
let assert_raises_ort_error f =
  match f () with
  | () -> Alcotest.fail "expected Ort_error but no exception was raised"
  | exception Onnxruntime.Ort_error _ -> ()
  | exception exn ->
    Alcotest.failf "expected Ort_error, got %s" (Printexc.to_string exn)

(* ------------------------------------------------------------------ *)
(* Test cases                                                          *)
(* ------------------------------------------------------------------ *)

let test_correctness_batch1 () =
  with_session (fun session ->
    run_inference session 1 |> assert_matches_reference)

let test_sequential_runs () =
  (* The Tessera ONNX model was exported with batch=1 baked into a reshape,
     so true batch>1 is not supported.  Instead we run 4 sequential batch=1
     inferences through the same session and verify all produce correct
     output. *)
  with_session (fun session ->
    List.init 4 (fun _ -> run_inference session 1)
    |> List.iter assert_matches_reference)

let test_invalid_model_path () =
  let env = make_env () in
  assert_raises_ort_error (fun () ->
    ignore (Onnxruntime.Session.create env "/nonexistent/path/model.onnx"))

let test_wrong_input_name () =
  with_session (fun session ->
    let s2 = make_s2_input 1 in
    let s1 = make_s1_input 1 in
    assert_raises_ort_error (fun () ->
      ignore (
        Onnxruntime.Session.run_ba session
          [| ("s2_wrong", s2, [| 1L; 40L; 11L |]);
             ("s1_input", s1, [| 1L; 40L; 3L |]) |]
          [| "output" |]
          ~output_sizes:[| output_dim |])))

let test_wrong_shape () =
  with_session (fun session ->
    let bad_s2 = Array1.create float32 c_layout (1 * 10 * 11) in
    Array1.fill bad_s2 0.0;
    let s1 = make_s1_input 1 in
    assert_raises_ort_error (fun () ->
      ignore (
        Onnxruntime.Session.run_ba session
          [| ("s2_input", bad_s2, [| 1L; 10L; 11L |]);
             ("s1_input", s1, [| 1L; 40L; 3L |]) |]
          [| "output" |]
          ~output_sizes:[| output_dim |])))

let test_repeated_runs () =
  with_session (fun session ->
    let results = List.init 5 (fun _ -> run_inference session 1) in
    let first = List.hd results in
    List.iteri
      (fun run output ->
        for i = 0 to output_dim - 1 do
          if first.{i} <> output.{i} then
            Alcotest.failf
              "run %d differs from run 0 at index %d: %.8f vs %.8f"
              run i first.{i} output.{i}
        done)
      results)

let test_multiple_sessions () =
  let env = make_env () in
  let path = Lazy.force model_path in
  let session_a = Onnxruntime.Session.create env path in
  let session_b = Onnxruntime.Session.create env path in
  [ session_a; session_b ]
  |> List.map (fun s -> run_inference s 1)
  |> List.iter assert_matches_reference

let test_env_log_levels () =
  List.iter
    (fun level ->
      ignore (Onnxruntime.Env.create ~log_level:level
                (Printf.sprintf "level_%d" level)))
    [ 0; 1; 2; 3; 4 ]

(* ------------------------------------------------------------------ *)
(* Runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "onnxruntime"
    [ ( "correctness",
        [ Alcotest.test_case "batch=1 numerical" `Quick
            test_correctness_batch1;
          Alcotest.test_case "4 sequential runs" `Quick
            test_sequential_runs ] );
      ( "errors",
        [ Alcotest.test_case "invalid model path" `Quick
            test_invalid_model_path;
          Alcotest.test_case "wrong input name" `Quick
            test_wrong_input_name;
          Alcotest.test_case "wrong input shape" `Quick
            test_wrong_shape ] );
      ( "stability",
        [ Alcotest.test_case "repeated runs" `Quick test_repeated_runs;
          Alcotest.test_case "multiple sessions" `Quick
            test_multiple_sessions ] );
      ( "env",
        [ Alcotest.test_case "log levels 0-4" `Quick
            test_env_log_levels ] );
    ]
