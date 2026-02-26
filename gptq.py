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

def vecmatmul_uppertri(out, mut, v, n:int, m:int):
    pass

# returns L^T
def spd_chomsky_decomp(m1: n:int, m:int) -> np.ndarray:
    # TODO: what happens to the output if m1 is not psd?
    pass

# solve for x in Rx = y <=> x = (R^-1)y
def uppertri_solve(m1, v, n:int, m:int)
    pass

def spd_matinv(m1, n:int, m:int) -> np.ndarray:
    R = spd_chomsky_decomp(m1, n, m)
    out = np.zeroes((n, m))

    for j in range(m):
        early_exit_backsubstitution_inplace(out, R[j * m + j], j, n, m) 
    
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

