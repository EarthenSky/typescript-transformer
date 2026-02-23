#include <stdio.h>
#include <string.h>
#include <node_api.h>

#include <emmintrin.h> // sse2
#include <immintrin.h> // f32 to f16 from AVX

// vecmatmul -----------------------------------------------------------------

#pragma float_control(precise, off)
#pragma fp_contract(on)
#pragma fenv_access(off)

#pragma GCC push_options
#pragma GCC optimize ("-ffast-math")

// TODO: restrict?
// x is n, M is (n,m)
static void vecmatmul(float *out, const float *M, const float *x, int32_t n, int32_t m) {
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


// This impl is just as fast as the vecmatmul
// TODO: this might be faster if i run it in parallel across j instead of i, since i would never have to reduce
static void vecmatmul__sse(float *__restrict out, const float *__restrict M, const float *__restrict x, int32_t n, int32_t m) {
    for (int j = 0; j < m; j++) {
        __m128 f0 = _mm_setzero_ps();
        __m128 f1 = _mm_setzero_ps();
        
        int32_t i = 0;
        while (i+3 < n) {
            // TODO: how to force alignment? Will it improve memory lookup speed?
            __m128 M4f0 = _mm_loadu_ps(&M[n * j + i]);
            __m128 x4f0 = _mm_loadu_ps(&x[i]);

            __m128 M4f1 = _mm_loadu_ps(&M[n * j + i + 4]);
            __m128 x4f1 = _mm_loadu_ps(&x[i + 4]);

            // TODO: is the other version using FMA?
            f0 = _mm_add_ps(f0, _mm_mul_ps(M4f0, x4f0));
            f1 = _mm_add_ps(f1, _mm_mul_ps(M4f1, x4f1));
            i += 8;
        }

        // a = [v0+v2,v1+v3,...]
        __m128 a0 = _mm_add_ps(f0, _mm_movehl_ps(f0, f0));
        // all lanes of b are v1+v3
        __m128 b0 = _mm_shuffle_ps(a0, a0, 0b01010101);
        out[j] = _mm_cvtss_f32(_mm_add_ss(a0, b0));

        __m128 a1 = _mm_add_ps(f1, _mm_movehl_ps(f1, f1));
        __m128 b1 = _mm_shuffle_ps(a1, a1, 0b01010101);
        out[j] += _mm_cvtss_f32(_mm_add_ss(a1, b1));

        while (i < n) {
            out[j] += M[n * j + i] * x[i];
            i++;
        }
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
// NOTE: x86 only

// TODO: run f16 on cpu to ensure it works nicely.
// THEN, implement GPTQ in python + C and check that it works on cpu
// THEN, implement it on gpu

#if defined(__x86_64__) || defined(_M_X64)

// n and m are float widths, not byte widths
static void vecmatmul_Mf16__sse(float *out, uint8_t *M, float *x, int32_t n, int32_t m) {
    for (int j = 0; j < m; j++) {
        __m128 f = _mm_setzero_ps();
        int32_t i = 0;

        // TODO: unroll to improve perf
        while (i+3 < n) {
            __m128i f16_data = _mm_loadu_si128((__m128i const *) &M[2 * (n * j + i)]);
            __m128 M4f = _mm_cvtph_ps(f16_data);
            __m128 x4f = _mm_loadu_ps(&x[i]);
            f = _mm_add_ps(f, _mm_mul_ps(M4f, x4f));
            i += 4;
        }

        while (i < n) {
            uint16_t f16;
            memcpy(&f16, &M[2 * (n * j + i)], 2);
            // TODO: can we set just the lowest item?
            __m128 M1f = _mm_cvtph_ps(_mm_set1_epi16(f16));
            __m128 x1f = _mm_load_ss(&x[i]);
            f = _mm_add_ss(f, _mm_mul_ss(M1f, x1f));
            i++;
        }

        __m128 a = _mm_add_ps(f, _mm_movehl_ps(f, f));
        __m128 b = _mm_shuffle_ps(a, a, 0b01010101);
        out[j] = _mm_cvtss_f32(_mm_add_ss(a, b));
    }
}

static void f32_to_f16__sse(uint8_t *out, float *in, size_t in_size) {
    size_t i = 0; 
    while (i+3 < in_size) {
        __m128 f32_data = _mm_loadu_ps(&in[i]);
        __m128i f16_data = _mm_cvtps_ph(f32_data, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *) &out[2*i], f16_data);
        i += 4;
    }

    while (i < in_size) {
        __m128 f32_data = _mm_load_ss(&in[i]);
        __m128i f16_data = _mm_cvtps_ph(f32_data, _MM_FROUND_TO_NEAREST_INT);
        // little endian, so this moves the smallest byte
        memcpy(&out[2*i], &f16_data, 2);
        i++;
    }
}

#endif

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

    vecmatmul__sse(
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

    vecmatmul_Mf16__sse(
        (float *)data_out,
        (uint8_t *)data_M,
        (float *)data_x,
        len_x,
        len_out
    );

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

    f32_to_f16__sse(
        (uint8_t *)data_out,
        (float *)data_x,
        len_x
    );

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
