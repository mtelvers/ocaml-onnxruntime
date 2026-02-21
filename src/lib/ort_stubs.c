/* ort_stubs.c — OCaml C API stubs for ONNX Runtime bindings.
   Marshals OCaml values to/from C types and calls into ort_shim.c. */

#include "ort_shim.h"

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/callback.h>
#include <caml/bigarray.h>

#include <string.h>

/* ------------------------------------------------------------------ */
/* Error handling                                                      */
/* ------------------------------------------------------------------ */

static const value *ort_error_exn = NULL;

static void check_ort_status(OrtStatus *status) {
    if (status != NULL) {
        const char *msg = ort_status_message(status);
        char *msg_copy = caml_stat_strdup(msg);
        ort_release_status(status);
        if (!ort_error_exn)
            ort_error_exn = caml_named_value("Ort_error");
        caml_raise_with_string(*ort_error_exn, msg_copy);
        /* msg_copy leaks (caml_raise_with_string is noreturn) —
           tiny, bounded, error-path only. */
    }
}

/* ------------------------------------------------------------------ */
/* Custom blocks: OrtEnv                                               */
/* ------------------------------------------------------------------ */

#define Env_val(v) (*((OrtEnv **) Data_custom_val(v)))

static void finalize_env(value v) {
    OrtEnv *env = Env_val(v);
    if (env) ort_release_env(env);
}

static struct custom_operations env_ops = {
    "onnxruntime.env",
    finalize_env,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default
};

/* ------------------------------------------------------------------ */
/* Custom blocks: OrtSession                                           */
/* ------------------------------------------------------------------ */

#define Session_val(v) (*((OrtSession **) Data_custom_val(v)))

static void finalize_session(value v) {
    OrtSession *s = Session_val(v);
    if (s) ort_release_session(s);
}

static struct custom_operations session_ops = {
    "onnxruntime.session",
    finalize_session,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default
};

/* ------------------------------------------------------------------ */
/* caml_ort_create_env : int -> string -> ort_env                      */
/* ------------------------------------------------------------------ */

CAMLprim value caml_ort_create_env(value v_log_level, value v_logid) {
    CAMLparam2(v_log_level, v_logid);
    CAMLlocal1(v_env);

    if (ort_init() != 0)
        caml_failwith("Failed to initialise ONNX Runtime API");

    char *logid = caml_stat_strdup(String_val(v_logid));

    OrtEnv *env = NULL;
    OrtStatus *status = ort_create_env(Int_val(v_log_level), logid, &env);
    caml_stat_free(logid);
    check_ort_status(status);

    v_env = caml_alloc_custom(&env_ops, sizeof(OrtEnv *), 0, 1);
    Env_val(v_env) = env;

    CAMLreturn(v_env);
}

/* ------------------------------------------------------------------ */
/* caml_ort_create_session :                                           */
/*   ort_env -> int -> int option -> string -> ort_session             */
/* ------------------------------------------------------------------ */

CAMLprim value caml_ort_create_session(value v_env, value v_threads,
                                        value v_cuda_device, value v_model_path) {
    CAMLparam4(v_env, v_threads, v_cuda_device, v_model_path);
    CAMLlocal1(v_session);

    OrtEnv *env = Env_val(v_env);
    int threads = Int_val(v_threads);
    char *model_path = caml_stat_strdup(String_val(v_model_path));

    OrtSessionOptions *opts = NULL;
    OrtStatus *status = ort_create_session_options(&opts);
    if (status) { caml_stat_free(model_path); check_ort_status(status); }

    if (threads > 0) {
        status = ort_set_intra_op_threads(opts, threads);
        if (status) {
            caml_stat_free(model_path);
            ort_release_session_options(opts);
            check_ort_status(status);
        }
    }

    /* ORT_ENABLE_ALL = 99 */
    status = ort_set_graph_opt_level(opts, 99);
    if (status) {
        caml_stat_free(model_path);
        ort_release_session_options(opts);
        check_ort_status(status);
    }

    /* CUDA provider (v_cuda_device is int option) */
    if (Is_block(v_cuda_device)) {
        int device_id = Int_val(Field(v_cuda_device, 0));
        status = ort_append_cuda_provider(opts, device_id);
        if (status) {
            caml_stat_free(model_path);
            ort_release_session_options(opts);
            check_ort_status(status);
        }
    }

    OrtSession *session = NULL;
    status = ort_create_session(env, model_path, opts, &session);
    caml_stat_free(model_path);
    ort_release_session_options(opts);
    check_ort_status(status);

    /* Cache names on the C side for ort_run_cached */
    status = ort_cache_session_names(session);
    if (status) {
        ort_release_session(session);
        check_ort_status(status);
    }

    v_session = caml_alloc_custom(&session_ops, sizeof(OrtSession *), 0, 1);
    Session_val(v_session) = session;

    CAMLreturn(v_session);
}

/* ------------------------------------------------------------------ */
/* Session name queries                                                */
/* ------------------------------------------------------------------ */

static value get_session_names(OrtSession *session,
        OrtStatus *(*count_fn)(OrtSession *, size_t *),
        OrtStatus *(*name_fn)(OrtSession *, size_t, OrtAllocator *, char **)) {
    CAMLparam0();
    CAMLlocal2(v_arr, v_str);

    OrtAllocator *alloc = NULL;
    check_ort_status(ort_get_allocator(&alloc));

    size_t count = 0;
    check_ort_status(count_fn(session, &count));

    /* Collect all names as C strings first (no OCaml allocation yet) */
    char **names = caml_stat_alloc(count * sizeof(char *));
    for (size_t i = 0; i < count; i++) {
        char *name = NULL;
        OrtStatus *status = name_fn(session, i, alloc, &name);
        if (status) {
            for (size_t j = 0; j < i; j++) caml_stat_free(names[j]);
            caml_stat_free(names);
            check_ort_status(status);
        }
        names[i] = caml_stat_strdup(name);
        ort_allocator_free(alloc, name);
    }

    /* Build OCaml string array */
    v_arr = caml_alloc(count, 0);
    for (size_t i = 0; i < count; i++) {
        v_str = caml_copy_string(names[i]);
        caml_stat_free(names[i]);
        Store_field(v_arr, i, v_str);
    }
    caml_stat_free(names);

    CAMLreturn(v_arr);
}

/* caml_ort_session_input_names : ort_session -> string array */
CAMLprim value caml_ort_session_input_names(value v_session) {
    CAMLparam1(v_session);
    CAMLreturn(get_session_names(Session_val(v_session),
                                 ort_session_input_count,
                                 ort_session_input_name));
}

/* caml_ort_session_output_names : ort_session -> string array */
CAMLprim value caml_ort_session_output_names(value v_session) {
    CAMLparam1(v_session);
    CAMLreturn(get_session_names(Session_val(v_session),
                                 ort_session_output_count,
                                 ort_session_output_name));
}

/* ------------------------------------------------------------------ */
/* caml_ort_run_ba (6 args — needs bytecode wrapper)                   */
/*   session * input_names * input_bas * input_shapes *                 */
/*   output_names * output_flat_sizes -> bigarray array                 */
/* ------------------------------------------------------------------ */

CAMLprim value caml_ort_run_ba(value v_session, value v_input_names,
                                value v_input_bas, value v_input_shapes,
                                value v_output_names, value v_output_sizes) {
    CAMLparam5(v_session, v_input_names, v_input_bas, v_input_shapes,
               v_output_names);
    CAMLxparam1(v_output_sizes);
    CAMLlocal2(v_result, v_ba);

    OrtSession *session = Session_val(v_session);
    int n_inputs  = Wosize_val(v_input_names);
    int n_outputs = Wosize_val(v_output_names);

    /* Copy input/output names to C heap */
    char **input_names  = caml_stat_alloc(n_inputs  * sizeof(char *));
    char **output_names = caml_stat_alloc(n_outputs * sizeof(char *));
    for (int i = 0; i < n_inputs; i++)
        input_names[i] = caml_stat_strdup(String_val(Field(v_input_names, i)));
    for (int i = 0; i < n_outputs; i++)
        output_names[i] = caml_stat_strdup(String_val(Field(v_output_names, i)));

    /* Copy output flat sizes */
    int *output_sizes = caml_stat_alloc(n_outputs * sizeof(int));
    for (int i = 0; i < n_outputs; i++)
        output_sizes[i] = Int_val(Field(v_output_sizes, i));

    /* Create CPU memory info */
    OrtMemoryInfo *mem_info = NULL;
    check_ort_status(ort_create_cpu_memory_info(&mem_info));

    /* Build input tensors */
    OrtValue **input_tensors = caml_stat_alloc(n_inputs * sizeof(OrtValue *));
    for (int i = 0; i < n_inputs; i++) {
        value v_ba_i    = Field(v_input_bas, i);
        value v_shape_i = Field(v_input_shapes, i);
        float *data     = (float *)Caml_ba_data_val(v_ba_i);
        int data_len    = Caml_ba_array_val(v_ba_i)->dim[0];
        int ndims       = Wosize_val(v_shape_i);
        int64_t *shape  = caml_stat_alloc(ndims * sizeof(int64_t));
        for (int j = 0; j < ndims; j++)
            shape[j] = Int64_val(Field(v_shape_i, j));

        OrtValue *tensor = NULL;
        OrtStatus *status = ort_create_tensor_float(
            mem_info, data, data_len, shape, ndims, &tensor);
        caml_stat_free(shape);
        check_ort_status(status);
        input_tensors[i] = tensor;
    }

    /* Allocate output array (NULLs — ORT allocates) */
    OrtValue **output_vals = caml_stat_alloc(n_outputs * sizeof(OrtValue *));
    for (int i = 0; i < n_outputs; i++) output_vals[i] = NULL;

    /* Run inference */
    OrtStatus *run_status = ort_run(session,
        (const char *const *)input_names,
        (const OrtValue *const *)input_tensors, n_inputs,
        (const char *const *)output_names, n_outputs,
        output_vals);

    /* Release input tensors and memory info */
    for (int i = 0; i < n_inputs; i++) ort_release_value(input_tensors[i]);
    caml_stat_free(input_tensors);
    ort_release_memory_info(mem_info);
    for (int i = 0; i < n_inputs; i++) caml_stat_free(input_names[i]);
    caml_stat_free(input_names);
    for (int i = 0; i < n_outputs; i++) caml_stat_free(output_names[i]);
    caml_stat_free(output_names);

    check_ort_status(run_status);

    /* Build result: array of bigarrays */
    v_result = caml_alloc(n_outputs, 0);
    for (int i = 0; i < n_outputs; i++) {
        float *out_data = NULL;
        check_ort_status(ort_get_tensor_float_data(output_vals[i], &out_data));

        int total = output_sizes[i];
        intnat dims[1] = { total };
        v_ba = caml_ba_alloc(CAML_BA_FLOAT32 | CAML_BA_C_LAYOUT, 1, NULL, dims);
        memcpy(Caml_ba_data_val(v_ba), out_data, total * sizeof(float));

        Store_field(v_result, i, v_ba);
        ort_release_value(output_vals[i]);
    }

    caml_stat_free(output_vals);
    caml_stat_free(output_sizes);

    CAMLreturn(v_result);
}

CAMLprim value caml_ort_run_ba_bytecode(value *argv, int argn) {
    (void)argn;
    return caml_ort_run_ba(argv[0], argv[1], argv[2],
                            argv[3], argv[4], argv[5]);
}

/* ------------------------------------------------------------------ */
/* caml_ort_run_cached_ba (4 args)                                     */
/*   session * input_bas * input_shapes * output_flat_sizes             */
/*   -> bigarray array                                                 */
/* ------------------------------------------------------------------ */

CAMLprim value caml_ort_run_cached_ba(value v_session, value v_input_bas,
                                       value v_input_shapes,
                                       value v_output_sizes) {
    CAMLparam4(v_session, v_input_bas, v_input_shapes, v_output_sizes);
    CAMLlocal2(v_result, v_ba);

    OrtSession *session = Session_val(v_session);
    int n_inputs  = Wosize_val(v_input_bas);
    int n_outputs = Wosize_val(v_output_sizes);

    /* Copy output flat sizes */
    int *output_sizes = caml_stat_alloc(n_outputs * sizeof(int));
    for (int i = 0; i < n_outputs; i++)
        output_sizes[i] = Int_val(Field(v_output_sizes, i));

    /* Create CPU memory info */
    OrtMemoryInfo *mem_info = NULL;
    check_ort_status(ort_create_cpu_memory_info(&mem_info));

    /* Build input tensors */
    OrtValue **input_tensors = caml_stat_alloc(n_inputs * sizeof(OrtValue *));
    for (int i = 0; i < n_inputs; i++) {
        value v_ba_i    = Field(v_input_bas, i);
        value v_shape_i = Field(v_input_shapes, i);
        float *data     = (float *)Caml_ba_data_val(v_ba_i);
        int data_len    = Caml_ba_array_val(v_ba_i)->dim[0];
        int ndims       = Wosize_val(v_shape_i);
        int64_t *shape  = caml_stat_alloc(ndims * sizeof(int64_t));
        for (int j = 0; j < ndims; j++)
            shape[j] = Int64_val(Field(v_shape_i, j));

        OrtValue *tensor = NULL;
        OrtStatus *status = ort_create_tensor_float(
            mem_info, data, data_len, shape, ndims, &tensor);
        caml_stat_free(shape);
        check_ort_status(status);
        input_tensors[i] = tensor;
    }

    /* Allocate output array */
    OrtValue **output_vals = caml_stat_alloc(n_outputs * sizeof(OrtValue *));
    for (int i = 0; i < n_outputs; i++) output_vals[i] = NULL;

    /* Run using C-cached names */
    OrtStatus *run_status = ort_run_cached(session,
        (const OrtValue *const *)input_tensors, n_inputs,
        output_vals, n_outputs);

    /* Release input tensors and memory info */
    for (int i = 0; i < n_inputs; i++) ort_release_value(input_tensors[i]);
    caml_stat_free(input_tensors);
    ort_release_memory_info(mem_info);

    check_ort_status(run_status);

    /* Build result: array of bigarrays */
    v_result = caml_alloc(n_outputs, 0);
    for (int i = 0; i < n_outputs; i++) {
        float *out_data = NULL;
        check_ort_status(ort_get_tensor_float_data(output_vals[i], &out_data));

        int total = output_sizes[i];
        intnat dims[1] = { total };
        v_ba = caml_ba_alloc(CAML_BA_FLOAT32 | CAML_BA_C_LAYOUT, 1, NULL, dims);
        memcpy(Caml_ba_data_val(v_ba), out_data, total * sizeof(float));

        Store_field(v_result, i, v_ba);
        ort_release_value(output_vals[i]);
    }

    caml_stat_free(output_vals);
    caml_stat_free(output_sizes);

    CAMLreturn(v_result);
}
