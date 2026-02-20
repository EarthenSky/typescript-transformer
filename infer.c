#include <stdio.h>
#include <string.h>
#include <node_api.h>

#include <emmintrin.h> // sse2

// vecmatmul -----------------------------------------------------------------

#pragma float_control(precise, off)
#pragma fp_contract(on)
#pragma fenv_access(off)

#pragma GCC push_options
#pragma GCC optimize ("-ffast-math")

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

#pragma GCC pop_options

#pragma float_control(precise, on)
#pragma fp_contract(off)
#pragma fenv_access(on)

// TODO: this might be faster if i run it in parallel across j instead of i, since i would never have to reduce
static void vecmatmul_sse2(float *out, float *M, float *x, int32_t n, int32_t m) {
    for (int j = 0; j < m; j++) {
        __m128 f = _mm_setzero_ps();
        for (int i = 0; i+3 < n; i+=4) {
            // TODO: how to force alignment? Will it improve memory lookup speed?
            __m128 M4f = _mm_loadu_ps(&M[n * j + i]);
            __m128 x4f = _mm_loadu_ps(&x[i]);
            f = _mm_add_ps(f, _mm_mul_ps(M4f, x4f));
        }

        for (int i = 0; i < n; i++) {
            __m128 M1f = _mm_load_ss(&M[n * j + i]);
            __m128 x1f = _mm_load_ss(&x[i]);
            f = _mm_add_ss(f, _mm_mul_ss(M1f, x1f));
        }

        __m128 x = _mm_movehl_ps(f, f);
        // y = [v0+v2,v1+v3,...]
        __m128 y = _mm_add_ps(f, x);
        __m128 z = _mm_shuffle_ps(y, y, _MM_SHUFFLE(1,0,2,3));
        out[j] = _mm_cvtss_f32(_mm_add_ss(y, z));

        //_mm_storeu_ps(&out[j], f);
    }
}


// ---------------------------------------------------------------------------

// TODO: profile just a single one of these runs and try to figure out whether the cpu or memory bandwidth is limiting.

// L2 cache is ~3 MB
// L1 cache is 288 KiB

// TODO: what are my memory bandwidths?

// NOTE: this version is slower, and I wonder if it's because its not being unrolled? Or maybe it's the
// extra zero assignments at the start

// Loading M will completely fill each cache line (64 bytes), allowing us
// to do column major traversal of our array without cache issues
#define VECMATMUL_TILE_SIZE (16)

static void vecmatmul_tiled(float *out, float *M, float *x, int32_t n, int32_t m) {
    for (int oi = 0; oi < m; oi++)
        out[oi] = 0.0;

    int xi = 0;
    while ((xi + VECMATMUL_TILE_SIZE - 1) < n) {
        for (int oi = 0; oi < m; oi++) {
            // TODO: gcc unroll this loop please
            for (int i = 0; i < VECMATMUL_TILE_SIZE; i++) {
                // x[xi .. xi+16] now hopefully lives in a single L1 cache line the entire loop,
                // so we avoid fetching excess memory
                out[oi] += M[n * oi + xi + i] * x[xi + i];
            }
        }

        xi += VECMATMUL_TILE_SIZE;
    }

    while (xi < n) {
        // curr_x can now hopefully live in register the entire loop,
        // so we avoid fetching excess memory
        float curr_x = x[xi];
        for (int oi = 0; oi < m; oi++) {
            out[oi] += M[n * oi + xi] * curr_x;
        }
        xi++;
    }
}

// f16 -----------------------------------------------------------------

#pragma float_control(precise, off)
#pragma fp_contract(on)
#pragma fenv_access(off)

#pragma GCC push_options
#pragma GCC optimize ("-ffast-math")

// n and m are float widths, not byte widths
static void vecmatmul_Mf16(float *out, uint8_t *M, float *x, int32_t n, int32_t m) {
    for (int j = 0; j < m; j++) {
        float acc = 0.0;
        _Float16 f;
        for (int i = 0; i < n; i++) {
            memcpy(&f, &M[2 * (n * j + i)], 2);
            acc += (float) f * x[i];
        }
        out[j] = acc;
    }
}

static void f32_to_f16(uint8_t *out, float *in, size_t in_size) {
    for (size_t i = 0; i < in_size; i++) {
        _Float16 f = in[i];
        memcpy(&out[2*i], &f, 2);
    }
}

#pragma GCC pop_options

#pragma float_control(precise, on)
#pragma fp_contract(off)
#pragma fenv_access(on)

// ---------------------------------------------------------------------------

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

static napi_value vecmatmul_Mf16_wrapper(
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
        || type_M != napi_uint8_array
        || type_x != napi_float32_array
    ) {
        napi_throw_type_error(env, NULL, "ERROR: all args must be Float32Array");
        return NULL;
    }

    vecmatmul_Mf16(
        (float *)data_out,
        (uint8_t *)data_M,
        (float *)data_x,
        len_x,
        len_out);

    napi_value undefined;
    napi_get_undefined(env, &undefined);
    return undefined;
}

static napi_value f32_to_f16_wrapper(
    napi_env env, napi_callback_info info
) {
    napi_value args[2];

    size_t num_args = 2; 
    napi_get_cb_info(
        env, info, &num_args, args, NULL, NULL);
    if (num_args != 2) {
        napi_throw_error(env, NULL, "Expected 2 args");
        return NULL;
    }

    napi_typedarray_type type_out;
    napi_typedarray_type type_x;
    size_t     len_out, len_x;
    void       *data_out, *data_x;
    napi_value ab_out;
    napi_value ab_x;
    size_t     offset_out, offset_x;

    napi_get_typedarray_info(
        env, args[0],
        &type_out, &len_out, &data_out, &ab_out, &offset_out
    );
    napi_get_typedarray_info(
        env, args[1],
        &type_x, &len_x, &data_x, &ab_x, &offset_x
    );

    if (
        type_out != napi_uint8_array
        || type_x != napi_float32_array
    ) {
        napi_throw_type_error(env, NULL, "ERROR: bad argument types");
        return NULL;
    }

    f32_to_f16(
        (uint8_t *)data_out,
        (float *)data_x,
        len_x);

    napi_value undefined;
    napi_get_undefined(env, &undefined);
    return undefined;
}

napi_value Init(napi_env env, napi_value exports) {
    fprintf(stderr, "NAPI INIT\n");

    napi_value fn;
    napi_create_function(env, NULL, 0, vecmatmul_wrapper, NULL, &fn);
    napi_set_named_property(env, exports, "vecmatmul", fn);

    napi_value fn1;
    napi_create_function(env, NULL, 0, vecmatmul_Mf16_wrapper, NULL, &fn1);
    napi_set_named_property(env, exports, "vecmatmul_Mf16", fn1);

    napi_value fn2;
    napi_create_function(env, NULL, 0, f32_to_f16_wrapper, NULL, &fn2);
    napi_set_named_property(env, exports, "f32_to_f16", fn2);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
