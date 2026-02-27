import numpy as no

QUANT_SYM_4BIT=0

def quant_sym_4bit(x:float) -> int:
    return max(-8, min(7, (int) round(x)))
    
# n is height, m is width, m is height2, o is width2
def matmul(m1, m2, n:int, m:int, o:int) -> np.ndarray:
    height1 = n
    shared  = m
    width2  = o
    out = np.zeros(height1 * width2, dtype.float32)
    for i in range(0, height1):
        for j in range(0, width2):
            total = 0
            for k in range(0, shared):
                total += m1[i * n + k] * m2[j + k * o]
            out[i * m + j] = total

    return out

# returns L^T
def spd_chomsky_decomp(m1: n:int) -> np.ndarray:
    # it would be more efficient to store this sparsely
    R = np.zeros((n, n))
    # TODO: what happens to the output if m1 is not psd?
    for i in range(n):
        pass
        rki_total = 0
        for k in range(i):
            rki_total += R[k * n + i] ** 2
        aii = m1[i * n + i]
        R[i * n + i] = sqrt(aii - (rki_total))

        for j in range(i+1):

            R[i * n + j] = rnn

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
    R = spd_chomsky_decomp(m1, n, m)

    out = np.zeroes((n, m))
    for j in range(m):
        early_exit_backsubstitution_inplace(out[j,:], R, R[j * m + j], j, n, m) 
    
    # symmetry
    for i in range(n):
        for j in range(m):
            if i > j:
                out[i * m + j] = out[j * m + i] 

    return out

# todo replace this
def gptq(weights:np.ndarray, method:int):
    B = 128
    width = weights.shape[0]
    height = weights.shape[1]
    
    H = matmul(x, y)

    i = 0
    while (i + B - 1) < width:
        for bi in range(B):
            
            pass
        # global update

