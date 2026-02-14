#include <stdio.h>
#include <node_api.h>

// x is n, M is (n,m)
static void vecmatmul(float *out, float *M, float *x, int32_t n, int32_t m) {
    for (int j = 0; j < m; j++) {
        float f = 0.0;
        for (int i = 0; i < n; i++) {
            f += M[n * j + i] * x[i];
        }
        out[j] = f;
    }
}

static napi_value vecmatmul_wrapper(
    napi_env env, napi_callback_info info
) {
    napi_value args[3];

    size_t num_args = 3; 
    napi_get_cb_info(
        env, info, &num_args, args, NULL, NULL);
    if (num_args != 3) {
        napi_throw_error(env, NULL, "Expected 3 args");
        return NULL;
    }

    napi_typedarray_type type_out;
    napi_typedarray_type type_M;
    napi_typedarray_type type_x;
    size_t     len_out, len_M, len_x;
    void       *data_out, *data_M, *data_x;
    napi_value ab_out;
    napi_value ab_M;
    napi_value ab_x;
    size_t     offset_out, offset_M, offset_x;
    // fprintf(stderr, "sizes: %zu, %xu, %zu\n", len_out, len_M, len_x);

    napi_get_typedarray_info(
        env, args[0],
        &type_out, &len_out, &data_out, &ab_out, &offset_out
    );
    napi_get_typedarray_info(
        env, args[1],
        &type_M, &len_M, &data_M, &ab_M, &offset_M
    );
    napi_get_typedarray_info(
        env, args[2],
        &type_x, &len_x, &data_x, &ab_x, &offset_x
    );

    if (
        type_out != napi_float32_array
        || type_M != napi_float32_array
        || type_x != napi_float32_array
    ) {
        napi_throw_type_error(env, NULL, "ERROR: all args must be Float32Array");
        return NULL;
    }

    vecmatmul(
        (float *)data_out,
        (float *)data_M,
        (float *)data_x,
        len_x,
        len_out);

    napi_value undefined;
    napi_get_undefined(env, &undefined);
    return undefined;
}

napi_value Init(napi_env env, napi_value exports) {
    fprintf(stderr, "INIT\n");
    napi_value fn;
    napi_create_function(env, NULL, 0, vecmatmul_wrapper, NULL, &fn);
    napi_set_named_property(env, exports, "vecmatmul", fn);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
