import math
import numpy as np

QUANT_SYM_4BIT=0

def quant_sym_4bit(x:float) -> int:
    rounded = int(round(x))
    return max(-8, min(7, rounded))
    
def transpose(out, m1, width:int, height:int):
    for i in range(height):
        for j in range(width):
            out[j * height + i] = m1[i * width + j]

# n is height, m is width, m is height2, o is width2
def matmul(out, m1, m2, n:int, m:int, o:int):
    height1 = n
    shared  = m
    width2  = o
    for i in range(0, height1):
        for j in range(0, width2):
            total = 0
            for k in range(0, shared):
                total += m1[i * m + k] * m2[j + k * o]
            out[i * width2 + j] = total

    return out

# returns L^T
def spd_cholesky_decomp(m1, n:int) -> np.ndarray:
    # it would be more efficient to store this sparsely
    R = np.zeros(n * n)

    # TODO: what happens to the output if m1 is not psd?
    for i in range(n):
        for j in range(i):
            r_sum = 0.0
            for k in range(j):
                r_sum += R[k * n + j] * R[k * n + i]

            inv_rii = 1 / R[j * n + j]
            R[j * n + i] = inv_rii * (m1[j * n + i] - r_sum)

        rki_total = 0
        for k in range(i+1):
            rki_total += R[k * n + i] ** 2

        # clamp to zero, in case aii - rki_total
        aii = m1[i * n + i]
        if (aii - rki_total) > 0:
            R[i * n + i] = math.sqrt(aii - rki_total)
        else:
            R[i * n + i] = 0

    return R

def spd_matinv(m1, n:int) -> np.ndarray:
    R = spd_cholesky_decomp(m1, n)

    # (R)(A^-1) = (L^-1)
    def spd_backsubstitution_inplace(
        out, R, vj, col:int, n:int
    ):
        s = np.zeros(n)
        s[col] = 1/vj

        for yi in range(0,col+1):
            row = col - yi
            weighted_sum = 0.0
            for k in range(n-1, row, -1):
                weighted_sum += out[(k * n) + col] * R[(row * n) + k]

            val = (s[row] - weighted_sum) / R[(row * n) + row]

            out[(row * n) + col] = val
            # symmetry immediately!
            out[(col * n) + row] = val

    out = np.zeros(n * n)
    for col in range(n-1, -1, -1):
        spd_backsubstitution_inplace(
            out,
            R,
            R[col * n + col],
            col, n
        )

    return out

def gptq(
    W:np.ndarray,
    X:np.ndarray,
    height:int,
    width:int,
    # method:int
) -> np.ndarray:
    B = 128
    # TODO: look into ndarray and determine if row major 
    Q = np.zeros(height * width)
    E = np.zeros(height * B) 
    # TODO: cholesky inverse H, then get cholesky upper tri of H_inv
    X_T = np.zeros(num_examples * width)
    transpose(X_T, X, width, num_examples)
    H = matmul(X, X_T, width, num_examples, width)
    for i in range(width * width):
        H[i] = 2 * H[i]

    H_inv = spd_matinv(H, width)
    H_inv = cholesky(H_inv)

    i = 0
    while i < width:
        batch_size = min(B, width - i)
        for bi in range(batch_size):
            for yi in range(height):
                Q[(yi * width) + i + bi] = quant_sym_4bit(W[(yi * width) + i + bi]) 
            for yi in range(height):
                E[yi + bi * width] = (
                    W[yi * width + i + bi]
                    - Q[yi * width + i + bi]
                ) / H_inv[(bi + i) * width + bi + i]

            # we multiply the row vector of H_inv
            # with a full column of E
            tmp = np.zeros(height * B)
            E_col = np.zeros(height)
            for yi in range(height):
                E_col[yi] = E[yi * width + bi]
            matmul(
                tmp,
                E_col,
                H_inv[(i+bi * width)+i+bi:(i+bi * width)+i+B],
                height, 1, B-bi
            )
            for yi in range(height):
                for xi in range(bi, B):
                    # TODO: are the indices of tmp correct here?
                    W[yi * width + xi + i] -= tmp[yi * B + xi - bi]
        i += B

        tmp = np.zeros(height * (width - i))
        H_inv_sub = np.zeros(B * (width-(i+B)))
        for yi in range(i, i+B):
            for xi in range(i+B, width):
                H_inv_sub[(yi-i)*B + xi-(i+B)] = H_inv[yi * B + xi]

        matmul(tmp, E, H_inv[i-B:i, i:], height, B, width - (i+B))
        for yi in range(height):
            for xi in range(i, width):
                W[yi * width + xi] -= tmp[yi * width + xi]

def random_upper(size:int) -> np.ndarray:
    m1 = np.random.rand(size * size)
    for i in range(size):
        for j in range(i):
             m1[i * size + j] = 0
    return m1

if __name__ == "__main__":
    size = 4
    R = random_upper(size)
    R_T = np.zeros(size * size)
    transpose(R_T, R, size, size)

    A = np.zeros(size * size)
    matmul(A, R_T, R, size, size, size)
    # print(f"A = {A} of len {len(A)}")

    R_prime = spd_cholesky_decomp(A, size)

    for i in range(size * size):
        if abs(R[i] - R_prime[i]) >= 0.01:
            raise Exception(f"error at i = {i}; diff = {R[i] - R_prime[i]}")

    A_inv = spd_matinv(A, size)

    maybe_I = np.zeros(size * size)
    matmul(maybe_I, A, A_inv, size, size, size)
    I = np.identity(size).reshape(-1)
    for i in range(size * size):
        if abs(I[i] - maybe_I[i]) > 0.001:
            raise Exception("Inversion: high error at i={i}")

    print(maybe_I)

    # TODO: write a simple test & ensure that
    # quantization error appears low
    N_EXAMPLES = 55
    WSIZE = 128
    W = np.random.rand(WSIZE * WSIZE)
    X = np.zeros(WSIZE * N_EXAMPLES)
    WQ = gptq(weights, X, WSIZE, WSIZE)

    X_prime = np.zeros(WSIZE * N_EXAMPLES)
    XQ_prime = np.zeros(WSIZE * N_EXAMPLES)
    matmul(X_prime, W, X, WSIZE, WSIZE, N_EXAMPLES)
    # TODO: I will need a special quantized matmul?
    matmul(XQ_prime, WQ, X, WSIZE, WSIZE, N_EXAMPLES)

    for i in range(WSIZE * N_EXAMPLES):
        err = abs(X_prime[i] - XQ_prime[i])
        if err >= 0.1:
            raise Exception("Quantization had high error of {err}")

