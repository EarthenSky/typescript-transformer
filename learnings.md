# learnings

## 1 - SIMD attempt
- C vecmatmul is faster than typescript by ~20%, although it's not clear exactly why. Maybe because it has been unrolled?
- The C version is only using vector intrinsics for moves. Because floating optimizations were not enabled, it was not vectorizing. We're find with small differences in weights, so we can handle all sorts of float optimizations.
- For the smaller models that fit in memory, this is great! We don't need more than sse unless our data fits entirely in L3, L2, or L1. our 15M model fits in L3 (I think?) so it might be a candidate for some further speedup. At least the final vecmatmul does fit in memory.

## 2 - CUDA time
- We've learned that we are EXTREMELY memory bound. Our goal is to run the 7B param model, which takes up ~ 28GB at 32bit precision.
- I have a 3060, which have 12GB of VRAM. This means we only have to compress our model down to 10GB. Int8 quantization seems reasonable considering that models tend to normalize, so that's the first step; we have to quantize first!
- 