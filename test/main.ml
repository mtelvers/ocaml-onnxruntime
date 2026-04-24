(* Alcotest suite for ocaml-onnxruntime bindings.

   The [smoke] group uses the tiny bundled [minimal.onnx] (a single Add op,
   ~100 bytes) to exercise every binding code path end-to-end in CI.

   The [tessera] group runs only when [TESSERA_MODEL] points at the real
   Tessera model. Its job is purely numerical: verify inference output
   matches a validated ONNX Runtime reference run on a production-scale
   model. Everything else (sessions, io_binding, error paths, determinism)
   is covered by [smoke] against the minimal model. *)

open Bigarray

(* ------------------------------------------------------------------ *)
(* Shared helpers                                                      *)
(* ------------------------------------------------------------------ *)

let make_env () = Onnxruntime.Env.create ~log_level:3 "test"

(* Alcotest.check_raises requires an exact exception match; we just need
   to verify that *some* Ort_error is raised, regardless of message. *)
let assert_raises_ort_error f =
  match f () with
  | () -> Alcotest.fail "expected Ort_error but no exception was raised"
  | exception Onnxruntime.Ort_error _ -> ()
  | exception exn ->
    Alcotest.failf "expected Ort_error, got %s" (Printexc.to_string exn)

let assert_close ~tol expected actual =
  let diff = Float.abs (expected -. actual) in
  if diff > tol then
    Alcotest.failf "expected %.6f, got %.6f (diff %.6e > tol %.6e)"
      expected actual diff tol

(* ------------------------------------------------------------------ *)
(* Smoke tests (bundled minimal.onnx — runs in CI unconditionally)     *)
(* ------------------------------------------------------------------ *)

(* minimal.onnx: y = a + b, where a,b,y are float32[3] *)
let minimal_path = "minimal.onnx"

let ba_of_list xs =
  let n = List.length xs in
  let ba = Array1.create float32 c_layout n in
  List.iteri (fun i x -> ba.{i} <- x) xs;
  ba

let assert_ba_equal ~tol expected actual =
  Alcotest.(check int) "output length"
    (Array.length expected) (Array1.dim actual);
  Array.iteri (fun i e -> assert_close ~tol e actual.{i}) expected

let with_minimal_session f =
  let env = make_env () in
  let session = Onnxruntime.Session.create env minimal_path in
  f session

let run_minimal session a b =
  let a_ba = ba_of_list a in
  let b_ba = ba_of_list b in
  let outputs =
    Onnxruntime.Session.run_ba session
      [| ("a", a_ba, [| 3L |]); ("b", b_ba, [| 3L |]) |]
      [| "y" |]
      ~output_sizes:[| 3 |]
  in
  match outputs with
  | [| y |] -> y
  | _ -> Alcotest.failf "expected 1 output, got %d" (Array.length outputs)

let test_smoke_add () =
  with_minimal_session (fun session ->
    let y = run_minimal session [ 1.0; 2.0; 3.0 ] [ 10.0; 20.0; 30.0 ] in
    assert_ba_equal ~tol:1e-6 [| 11.0; 22.0; 33.0 |] y)

let test_smoke_sequential_runs () =
  with_minimal_session (fun session ->
    for i = 0 to 3 do
      let k = Float.of_int i in
      let y = run_minimal session [ k; k; k ] [ 1.0; 2.0; 3.0 ] in
      assert_ba_equal ~tol:1e-6
        [| k +. 1.0; k +. 2.0; k +. 3.0 |] y
    done)

let test_smoke_repeated_runs_deterministic () =
  with_minimal_session (fun session ->
    let a = [ 0.1; 0.2; 0.3 ] in
    let b = [ 0.4; 0.5; 0.6 ] in
    let first = run_minimal session a b in
    for run = 1 to 4 do
      let y = run_minimal session a b in
      for i = 0 to 2 do
        if first.{i} <> y.{i} then
          Alcotest.failf
            "run %d differs from run 0 at index %d: %.8f vs %.8f"
            run i first.{i} y.{i}
      done
    done)

let test_smoke_multiple_sessions () =
  let env = make_env () in
  let s_a = Onnxruntime.Session.create env minimal_path in
  let s_b = Onnxruntime.Session.create env minimal_path in
  let a = [ 1.0; 2.0; 3.0 ] in
  let b = [ 0.5; 0.5; 0.5 ] in
  let expect = [| 1.5; 2.5; 3.5 |] in
  assert_ba_equal ~tol:1e-6 expect (run_minimal s_a a b);
  assert_ba_equal ~tol:1e-6 expect (run_minimal s_b a b)

let test_smoke_io_binding () =
  with_minimal_session (fun session ->
    let a = ba_of_list [ 4.0; 5.0; 6.0 ] in
    let b = ba_of_list [ 0.5; 0.5; 0.5 ] in
    let y = Array1.create float32 c_layout 3 in
    let binding = Onnxruntime.Io_binding.create session in
    Onnxruntime.Io_binding.bind_input binding "a" a [| 3L |];
    Onnxruntime.Io_binding.bind_input binding "b" b [| 3L |];
    Onnxruntime.Io_binding.bind_output binding "y" y [| 3L |];
    Onnxruntime.Io_binding.run binding;
    assert_ba_equal ~tol:1e-6 [| 4.5; 5.5; 6.5 |] y)

let test_smoke_io_binding_repeated () =
  with_minimal_session (fun session ->
    let a = ba_of_list [ 1.0; 2.0; 3.0 ] in
    let b = ba_of_list [ 10.0; 10.0; 10.0 ] in
    let y = Array1.create float32 c_layout 3 in
    let binding = Onnxruntime.Io_binding.create session in
    Onnxruntime.Io_binding.bind_input binding "a" a [| 3L |];
    Onnxruntime.Io_binding.bind_input binding "b" b [| 3L |];
    Onnxruntime.Io_binding.bind_output binding "y" y [| 3L |];
    for _ = 1 to 5 do
      Onnxruntime.Io_binding.run binding;
      assert_ba_equal ~tol:1e-6 [| 11.0; 12.0; 13.0 |] y
    done)

let test_smoke_io_binding_clear_rebind () =
  with_minimal_session (fun session ->
    let a = ba_of_list [ 1.0; 2.0; 3.0 ] in
    let b = ba_of_list [ 10.0; 20.0; 30.0 ] in
    let y = Array1.create float32 c_layout 3 in
    let binding = Onnxruntime.Io_binding.create session in
    Onnxruntime.Io_binding.bind_input binding "a" a [| 3L |];
    Onnxruntime.Io_binding.bind_input binding "b" b [| 3L |];
    Onnxruntime.Io_binding.bind_output binding "y" y [| 3L |];
    Onnxruntime.Io_binding.run binding;
    assert_ba_equal ~tol:1e-6 [| 11.0; 22.0; 33.0 |] y;
    Onnxruntime.Io_binding.clear_inputs binding;
    Onnxruntime.Io_binding.clear_outputs binding;
    Array1.fill y 0.0;
    let a' = ba_of_list [ 100.0; 200.0; 300.0 ] in
    let b' = ba_of_list [ 1.0; 2.0; 3.0 ] in
    Onnxruntime.Io_binding.bind_input binding "a" a' [| 3L |];
    Onnxruntime.Io_binding.bind_input binding "b" b' [| 3L |];
    Onnxruntime.Io_binding.bind_output binding "y" y [| 3L |];
    Onnxruntime.Io_binding.run binding;
    assert_ba_equal ~tol:1e-6 [| 101.0; 202.0; 303.0 |] y)

let test_smoke_wrong_input_name () =
  with_minimal_session (fun session ->
    let a = ba_of_list [ 1.0; 2.0; 3.0 ] in
    let b = ba_of_list [ 1.0; 2.0; 3.0 ] in
    assert_raises_ort_error (fun () ->
      ignore (
        Onnxruntime.Session.run_ba session
          [| ("wrong", a, [| 3L |]); ("b", b, [| 3L |]) |]
          [| "y" |]
          ~output_sizes:[| 3 |])))

let test_smoke_wrong_input_shape () =
  with_minimal_session (fun session ->
    let bad = Array1.create float32 c_layout 5 in
    Array1.fill bad 0.0;
    let b = ba_of_list [ 1.0; 2.0; 3.0 ] in
    assert_raises_ort_error (fun () ->
      ignore (
        Onnxruntime.Session.run_ba session
          [| ("a", bad, [| 5L |]); ("b", b, [| 3L |]) |]
          [| "y" |]
          ~output_sizes:[| 3 |])))

let test_smoke_invalid_model_path () =
  let env = make_env () in
  assert_raises_ort_error (fun () ->
    ignore (Onnxruntime.Session.create env "/nonexistent/model.onnx"))

let test_smoke_env_log_levels () =
  List.iter
    (fun level ->
      ignore (Onnxruntime.Env.create ~log_level:level
                (Printf.sprintf "level_%d" level)))
    [ 0; 1; 2; 3; 4 ]

(* pi.onnx: zero-input model that computes pi via 10000 terms of the
   Leibniz series. Exercises the zero-input code path and demonstrates
   that ONNX is a general numerical computation graph, not just ML. *)
let test_smoke_pi_leibniz () =
  let env = make_env () in
  let session = Onnxruntime.Session.create env "pi.onnx" in
  let outputs =
    Onnxruntime.Session.run_ba session [||] [| "pi" |] ~output_sizes:[| 1 |]
  in
  match outputs with
  | [| pi |] ->
    Alcotest.(check int) "scalar output" 1 (Array1.dim pi);
    (* Leibniz truncation error after 10k terms is ~1/N ≈ 1e-4. *)
    assert_close ~tol:5e-4 Float.pi pi.{0}
  | _ -> Alcotest.failf "expected 1 output, got %d" (Array.length outputs)

(* mandelbrot.onnx: zero-input model producing a 32x32 float32 grid where
   each cell holds the iteration count at which z_{n+1} = z_n^2 + c left
   the disk |z| <= 2 (capped at MAX_ITER=50). Iteration is unrolled in
   the graph — pure tensor arithmetic over the whole grid in parallel. *)
let test_smoke_mandelbrot () =
  let h = 32 and w = 32 and max_iter = 50 in
  let env = make_env () in
  let session = Onnxruntime.Session.create env "mandelbrot.onnx" in
  let outputs =
    Onnxruntime.Session.run_ba session [||] [| "count" |]
      ~output_sizes:[| h * w |]
  in
  match outputs with
  | [| grid |] ->
    Alcotest.(check int) "grid size" (h * w) (Array1.dim grid);
    let at r c = grid.{(r * w) + c} in
    (* All values lie in [0, max_iter]. *)
    for i = 0 to (h * w) - 1 do
      let v = grid.{i} in
      if v < 0.0 || v > Float.of_int max_iter then
        Alcotest.failf "out-of-range count %.1f at index %d" v i
    done;
    (* Corner (-2.0, -1.25) escapes at iter 1 → count = 1. *)
    assert_close ~tol:0.5 1.0 (at 0 0);
    (* Cell near (-0.71, 0.04): deep in the main cardioid, never escapes
       → count = max_iter. *)
    assert_close ~tol:0.5 (Float.of_int max_iter) (at 16 16);
    (* The grid should show real Mandelbrot structure — a wide spread of
       escape counts, not a flat field. *)
    let min_v = ref infinity and max_v = ref neg_infinity in
    for i = 0 to (h * w) - 1 do
      let v = grid.{i} in
      if v < !min_v then min_v := v;
      if v > !max_v then max_v := v
    done;
    if !max_v -. !min_v < 10.0 then
      Alcotest.failf "flat grid: min=%.1f max=%.1f" !min_v !max_v
  | _ -> Alcotest.failf "expected 1 output, got %d" (Array.length outputs)

let smoke_tests =
  [ ( "smoke/correctness",
      [ Alcotest.test_case "add a + b" `Quick test_smoke_add;
        Alcotest.test_case "4 sequential runs" `Quick
          test_smoke_sequential_runs ] );
    ( "smoke/stability",
      [ Alcotest.test_case "repeated runs deterministic" `Quick
          test_smoke_repeated_runs_deterministic;
        Alcotest.test_case "multiple sessions" `Quick
          test_smoke_multiple_sessions ] );
    ( "smoke/io_binding",
      [ Alcotest.test_case "basic" `Quick test_smoke_io_binding;
        Alcotest.test_case "repeated runs" `Quick
          test_smoke_io_binding_repeated;
        Alcotest.test_case "clear and rebind" `Quick
          test_smoke_io_binding_clear_rebind ] );
    ( "smoke/errors",
      [ Alcotest.test_case "wrong input name" `Quick
          test_smoke_wrong_input_name;
        Alcotest.test_case "wrong input shape" `Quick
          test_smoke_wrong_input_shape;
        Alcotest.test_case "invalid model path" `Quick
          test_smoke_invalid_model_path ] );
    ( "smoke/env",
      [ Alcotest.test_case "log levels 0-4" `Quick
          test_smoke_env_log_levels ] );
    ( "smoke/pi",
      [ Alcotest.test_case "leibniz ~ pi (10k terms)" `Quick
          test_smoke_pi_leibniz ] );
    ( "smoke/mandelbrot",
      [ Alcotest.test_case "32x32 grid, 50 iters" `Quick
          test_smoke_mandelbrot ] );
  ]

(* ------------------------------------------------------------------ *)
(* Tessera tests (opt-in via TESSERA_MODEL env var)                   *)
(*                                                                     *)
(* Purpose: verify that the bindings produce numerically correct       *)
(* results on a production-scale model by matching against a set of    *)
(* reference outputs from a validated ONNX Runtime run. Generic        *)
(* binding behaviours are exercised by the smoke group.                *)
(* ------------------------------------------------------------------ *)

let seq_len = 40
let s2_dim = 11
let s1_dim = 3
let output_dim = 128

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

let assert_matches_reference output =
  Alcotest.(check int) "output length" output_dim (Array1.dim output);
  Array.iteri
    (fun i expected -> assert_close ~tol:1e-4 expected output.{i})
    reference_output

let tessera_tests model_path =
  let run_inference session =
    let s2 = make_s2_input 1 in
    let s1 = make_s1_input 1 in
    let outputs =
      Onnxruntime.Session.run_ba session
        [| ("s2_input", s2, [| 1L; 40L; 11L |]);
           ("s1_input", s1, [| 1L; 40L; 3L |]) |]
        [| "output" |]
        ~output_sizes:[| output_dim |]
    in
    match outputs with
    | [| out |] -> out
    | _ ->
      Alcotest.failf "expected 1 output, got %d" (Array.length outputs)
  in
  let with_session f =
    let env = make_env () in
    let session = Onnxruntime.Session.create env model_path in
    f session
  in
  let test_reference_run_ba () =
    with_session (fun session ->
      run_inference session |> assert_matches_reference)
  in
  let test_reference_io_binding () =
    with_session (fun session ->
      let s2 = make_s2_input 1 in
      let s1 = make_s1_input 1 in
      let out = Array1.create float32 c_layout output_dim in
      Array1.fill out 0.0;
      let binding = Onnxruntime.Io_binding.create session in
      Onnxruntime.Io_binding.bind_input binding "s2_input" s2 [| 1L; 40L; 11L |];
      Onnxruntime.Io_binding.bind_input binding "s1_input" s1 [| 1L; 40L; 3L |];
      Onnxruntime.Io_binding.bind_output binding "output" out [| 1L; 128L |];
      Onnxruntime.Io_binding.run binding;
      assert_matches_reference out)
  in
  [ ( "tessera/reference",
      [ Alcotest.test_case "run_ba matches reference" `Quick
          test_reference_run_ba;
        Alcotest.test_case "io_binding matches reference" `Quick
          test_reference_io_binding ] ) ]

(* ------------------------------------------------------------------ *)
(* Runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  let tessera =
    match Sys.getenv_opt "TESSERA_MODEL" with
    | None ->
      prerr_endline
        "[info] TESSERA_MODEL not set — skipping tessera tests \
         (smoke tests still run).";
      []
    | Some path -> tessera_tests path
  in
  Alcotest.run "onnxruntime" (smoke_tests @ tessera)
