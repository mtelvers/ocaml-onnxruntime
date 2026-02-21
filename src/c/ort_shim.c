/* ort_shim.c — Implementation of the thin C shim over ONNX Runtime. */

#include "ort_shim.h"
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const OrtApi *g_ort = NULL;

int ort_init(void) {
    if (g_ort) return 0;
    const OrtApiBase *base = OrtGetApiBase();
    if (!base) return -1;
    g_ort = base->GetApi(ORT_API_VERSION);
    return g_ort ? 0 : -1;
}

const char *ort_status_message(OrtStatus *s) {
    if (!g_ort || !s) return "unknown error";
    return g_ort->GetErrorMessage(s);
}

void ort_release_status(OrtStatus *s) {
    if (g_ort && s) g_ort->ReleaseStatus(s);
}

/* Environment */
OrtStatus *ort_create_env(int log_level, const char *logid, OrtEnv **out) {
    return g_ort->CreateEnv((OrtLoggingLevel)log_level, logid, out);
}

void ort_release_env(OrtEnv *env) {
    if (g_ort && env) g_ort->ReleaseEnv(env);
}

/* Session options */
OrtStatus *ort_create_session_options(OrtSessionOptions **out) {
    return g_ort->CreateSessionOptions(out);
}

OrtStatus *ort_set_intra_op_threads(OrtSessionOptions *opts, int n) {
    return g_ort->SetIntraOpNumThreads(opts, n);
}

OrtStatus *ort_set_graph_opt_level(OrtSessionOptions *opts, int level) {
    return g_ort->SetSessionGraphOptimizationLevel(
        opts, (GraphOptimizationLevel)level);
}

void ort_release_session_options(OrtSessionOptions *opts) {
    if (g_ort && opts) g_ort->ReleaseSessionOptions(opts);
}

/* Session */
OrtStatus *ort_create_session(OrtEnv *env, const char *path,
                              OrtSessionOptions *opts, OrtSession **out) {
    return g_ort->CreateSession(env, path, opts, out);
}

void ort_release_session(OrtSession *s) {
    if (g_ort && s) g_ort->ReleaseSession(s);
}

/* Allocator & names */
OrtStatus *ort_get_allocator(OrtAllocator **out) {
    return g_ort->GetAllocatorWithDefaultOptions(out);
}

OrtStatus *ort_session_input_name(OrtSession *s, size_t i,
                                  OrtAllocator *a, char **out) {
    return g_ort->SessionGetInputName(s, i, a, out);
}

OrtStatus *ort_session_output_name(OrtSession *s, size_t i,
                                   OrtAllocator *a, char **out) {
    return g_ort->SessionGetOutputName(s, i, a, out);
}

OrtStatus *ort_session_input_count(OrtSession *s, size_t *out) {
    return g_ort->SessionGetInputCount(s, out);
}

OrtStatus *ort_session_output_count(OrtSession *s, size_t *out) {
    return g_ort->SessionGetOutputCount(s, out);
}

OrtStatus *ort_allocator_free(OrtAllocator *a, void *p) {
    return g_ort->AllocatorFree(a, p);
}

/* Tensors */
OrtStatus *ort_create_cpu_memory_info(OrtMemoryInfo **out) {
    /* OrtDeviceAllocator = 0, OrtMemTypeDefault = 0 */
    return g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, out);
}

void ort_release_memory_info(OrtMemoryInfo *info) {
    if (g_ort && info) g_ort->ReleaseMemoryInfo(info);
}

OrtStatus *ort_create_tensor_float(OrtMemoryInfo *info, float *data,
                                   size_t data_len, const int64_t *shape,
                                   size_t ndims, OrtValue **out) {
    return g_ort->CreateTensorWithDataAsOrtValue(
        info, data, data_len * sizeof(float), shape, ndims,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, out);
}

void ort_release_value(OrtValue *v) {
    if (g_ort && v) g_ort->ReleaseValue(v);
}

OrtStatus *ort_get_tensor_float_data(OrtValue *v, float **out) {
    return g_ort->GetTensorMutableData(v, (void **)out);
}

/* Run */
OrtStatus *ort_run(OrtSession *s,
                   const char *const *input_names,
                   const OrtValue *const *inputs, size_t n_inputs,
                   const char *const *output_names, size_t n_outputs,
                   OrtValue **outputs) {
    return g_ort->Run(s, NULL, input_names, inputs, n_inputs,
                      output_names, n_outputs, outputs);
}

/* ---- CUDA execution provider ---- */

OrtStatus *ort_append_cuda_provider(OrtSessionOptions *opts, int device_id) {
    OrtCUDAProviderOptionsV2 *cuda_opts = NULL;
    OrtStatus *status = g_ort->CreateCUDAProviderOptions(&cuda_opts);
    if (status) return status;

    const char *keys[] = {"device_id"};
    char device_id_str[16];
    snprintf(device_id_str, sizeof(device_id_str), "%d", device_id);
    const char *values[] = {device_id_str};

    status = g_ort->UpdateCUDAProviderOptions(cuda_opts, keys, values, 1);
    if (status) {
        g_ort->ReleaseCUDAProviderOptions(cuda_opts);
        return status;
    }

    status = g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cuda_opts);
    g_ort->ReleaseCUDAProviderOptions(cuda_opts);
    return status;
}

/* ---- Cached-names run: avoids passing names through OCaml/ctypes ---- */

static char **g_cached_input_names = NULL;
static char **g_cached_output_names = NULL;
static size_t g_cached_n_inputs = 0;
static size_t g_cached_n_outputs = 0;

OrtStatus *ort_cache_session_names(OrtSession *s) {
    OrtAllocator *alloc;
    OrtStatus *status = g_ort->GetAllocatorWithDefaultOptions(&alloc);
    if (status) return status;

    if (g_cached_input_names) {
        for (size_t i = 0; i < g_cached_n_inputs; i++)
            free(g_cached_input_names[i]);
        free(g_cached_input_names);
    }
    if (g_cached_output_names) {
        for (size_t i = 0; i < g_cached_n_outputs; i++)
            free(g_cached_output_names[i]);
        free(g_cached_output_names);
    }

    status = g_ort->SessionGetInputCount(s, &g_cached_n_inputs);
    if (status) return status;
    g_cached_input_names = (char **)malloc(g_cached_n_inputs * sizeof(char *));
    for (size_t i = 0; i < g_cached_n_inputs; i++) {
        char *name;
        status = g_ort->SessionGetInputName(s, i, alloc, &name);
        if (status) return status;
        g_cached_input_names[i] = strdup(name);
        g_ort->AllocatorFree(alloc, name);
    }

    status = g_ort->SessionGetOutputCount(s, &g_cached_n_outputs);
    if (status) return status;
    g_cached_output_names = (char **)malloc(g_cached_n_outputs * sizeof(char *));
    for (size_t i = 0; i < g_cached_n_outputs; i++) {
        char *name;
        status = g_ort->SessionGetOutputName(s, i, alloc, &name);
        if (status) return status;
        g_cached_output_names[i] = strdup(name);
        g_ort->AllocatorFree(alloc, name);
    }

    return NULL;
}

OrtStatus *ort_run_cached(OrtSession *s,
                          const OrtValue *const *inputs, size_t n_inputs,
                          OrtValue **outputs, size_t n_outputs) {
    if (!g_cached_input_names || !g_cached_output_names)
        return NULL;
    return g_ort->Run(s, NULL,
                      (const char *const *)g_cached_input_names, inputs, n_inputs,
                      (const char *const *)g_cached_output_names, n_outputs, outputs);
}
