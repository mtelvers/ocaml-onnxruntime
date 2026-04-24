(lang dune 3.0)
(name onnxruntime)

(generate_opam_files true)

(package
 (name onnxruntime)
 (synopsis "OCaml bindings to ONNX Runtime via C shim")
 (depends
  (ocaml (>= 5.1.0))
  conf-onnxruntime
  (alcotest :with-test)))
