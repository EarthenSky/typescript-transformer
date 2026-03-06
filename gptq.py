import numpy as np

QUANT_SYM_4BIT=0

def quant_sym_4bit(x:float) -> int:
    rounded = int(round(x))
    return max(-8, min(7, rounded))
    
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
    # R = np.zeros(n * n)
    R = m1.copy()

    # TODO: what happens to the output if m1 is not psd?
    for i in range(n):
        # fix i, use all j up until but not including j
        for j in range(i):
            inv_rii = 1 / R[j * n + j]
            r_sum = 0.0
            for k in range(j):
                r_sum += R[k * n + j] * R[k * n + i]
            R[j * n + i] = inv_rii * (m1[j * n + i] - r_sum)

        rki_total = 0
        for k in range(i+1):
            rki_total += R[k * n + i] ** 2
        aii = m1[i * n + i]
        R[i * n + i] = sqrt(aii - (rki_total))

# solve for Rx = v, where v is a sparse vector
# containing vj at j
def early_exit_backsubstitution_inplace(
    out, R, vj, j, n:int, m:int
):
    # TODO: we can reuse tmp's memory between calls
    s = np.zeroes(n)
    s[j] = vj

    for xi in range(j+1):
        # solve for outi
        out[xi] = s[yi] / R[(j - xi) * m + (j - xi)]
        for yi in range(n):
            s[yi] -= R[yi * m + (j - xi)] * out[xi]

def spd_matinv(m1, n:int, m:int) -> np.ndarray:
    R = spd_cholesky_decomp(m1, n, m)

    out = np.zeroes((n, m))
    for j in range(m):
        early_exit_backsubstitution_inplace(out[j,:], R, R[j * m + j], j, n, m) 
    
    # symmetry
    for i in range(n):
        for j in range(m):
            if i > j:
                out[i * m + j] = out[j * m + i] 

    return out

def gptq(
    W:np.ndarray,
    X:np.ndarray,
    height:int,
    width:int,
    # method:int
):
    B = 128
    # TODO: look into ndarray and determine if row major 
    Q = np.zeros(height * width)
    E = np.zeros(height * B) 
    # TODO: cholesky inverse H, then get cholesky upper tri of H_inv
    X_T = np.zeros(num_examples * width)
    transpose(X_T, X)
    H = matmul(X, X_T, width, num_examples, width)
    for i in range(width * width):
        H[i] = 2 * H[i]

    # TODO: impl these
    H_inv = cholesky_inv(H)
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

def random_upper(size:int):
    m1 = np.random(size)
    for i in range(size):
        for j in range(i):
             m1[j * size + i] = 0
    return m1

if __name__ == "__main__":
    size = 4
    R = random_upper(size)
    R_T = np.zeros(size * size)
    transpose(R_T, R)

    print(R)
    print(R_T)
    A = np.zeros(size * size)
    matmul(A, R, R_T, size, size, size)
    print(A)

    # TODO: this part of the test
    R_prime = spd_cholesky_decomp(A, sixe)
    print(R_prime)

    for i in range(size * size):
        if abs(R[i] - R_prime[i]) >= 0.01:
            raise f"error at i = {i}; diff = {R[i] -R_prime[i]}"
    # Todo: write a simple test & ensure that
    # quantization error appears low
    #N_EXAMPLES = 55
    #weights = np.zeros(128 * 128)
    #X = np.zeros(128 * N_EXAMPLES)
    #gptq(weights, X, 128, 128)
