/* ort_shim.h — Thin C shim over ONNX Runtime's function-table API.
   Exposes the needed ORT functions as regular C functions so OCaml ctypes
   can bind them via dlopen without modelling the 500+ field OrtApi struct. */

#ifndef ORT_SHIM_H
#define ORT_SHIM_H

#include <stddef.h>
#include <stdint.h>

/* Opaque pointer types matching ONNX Runtime */
typedef struct OrtEnv OrtEnv;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtSession OrtSession;
typedef struct OrtMemoryInfo OrtMemoryInfo;
typedef struct OrtAllocator OrtAllocator;
typedef struct OrtValue OrtValue;
typedef struct OrtStatus OrtStatus;
typedef struct OrtRunOptions OrtRunOptions;

/* Initialise the global OrtApi pointer. Returns 0 on success, -1 on failure. */
int ort_init(void);

/* Status helpers */
const char *ort_status_message(OrtStatus *s);
void ort_release_status(OrtStatus *s);

/* Environment */
OrtStatus *ort_create_env(int log_level, const char *logid, OrtEnv **out);
void ort_release_env(OrtEnv *env);

/* Session options */
OrtStatus *ort_create_session_options(OrtSessionOptions **out);
OrtStatus *ort_set_intra_op_threads(OrtSessionOptions *opts, int n);
OrtStatus *ort_set_graph_opt_level(OrtSessionOptions *opts, int level);
void ort_release_session_options(OrtSessionOptions *opts);

/* Session */
OrtStatus *ort_create_session(OrtEnv *env, const char *path,
                              OrtSessionOptions *opts, OrtSession **out);
void ort_release_session(OrtSession *s);

/* Allocator & names */
OrtStatus *ort_get_allocator(OrtAllocator **out);
OrtStatus *ort_session_input_name(OrtSession *s, size_t i,
                                  OrtAllocator *a, char **out);
OrtStatus *ort_session_output_name(OrtSession *s, size_t i,
                                   OrtAllocator *a, char **out);
OrtStatus *ort_session_input_count(OrtSession *s, size_t *out);
OrtStatus *ort_session_output_count(OrtSession *s, size_t *out);
OrtStatus *ort_allocator_free(OrtAllocator *a, void *p);

/* Tensors */
OrtStatus *ort_create_cpu_memory_info(OrtMemoryInfo **out);
void ort_release_memory_info(OrtMemoryInfo *info);
OrtStatus *ort_create_tensor_float(OrtMemoryInfo *info, float *data,
                                   size_t data_len, const int64_t *shape,
                                   size_t ndims, OrtValue **out);
void ort_release_value(OrtValue *v);
OrtStatus *ort_get_tensor_float_data(OrtValue *v, float **out);

/* Run inference */
OrtStatus *ort_run(OrtSession *s,
                   const char *const *input_names,
                   const OrtValue *const *inputs, size_t n_inputs,
                   const char *const *output_names, size_t n_outputs,
                   OrtValue **outputs);

/* CUDA execution provider */
OrtStatus *ort_append_cuda_provider(OrtSessionOptions *opts, int device_id);

/* Cached-names run: caches input/output names in C to avoid passing
   through OCaml/ctypes on every call. */
OrtStatus *ort_cache_session_names(OrtSession *s);
OrtStatus *ort_run_cached(OrtSession *s,
                          const OrtValue *const *inputs, size_t n_inputs,
                          OrtValue **outputs, size_t n_outputs);

#endif /* ORT_SHIM_H */
