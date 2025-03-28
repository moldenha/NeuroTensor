# Matrix Multiplication Testing
This directory holds the matrix multiplication routes specific for each supported NeuroTensor numerical type. Currently, only CPU support has been added, GPU support soon to come. Performance was tested on a Macbook Pro 2.3 GHz 8-Core Intel Core i9 against the MKL cblas_sgemm_64 function, on MKL version 2023.2.2.

If you are looking for a tutorial that can offer logic behind building a fast matrix multiplication I reccomend looking at https://github.com/salykova/matmul.c. While there are many differences, such as their implementation works for column major matricies, and some of my logic differs, it is a good alternative if you are looking for a way to implement something similar on your own.

## Performance

Test enviroment:
- CPU: 8-Core Intel Core i9
- CPU locked clock speed: 2.3 GHz
- RAM: 16 GB 2667 MHz DDR4
- MKL 2023.2.2
- Compiler: Apple clang version 15.0.0 (clang-1500.3.9.4)
- OS: MacOS 14.4.1

The graph below compares the performance of matrix multiplications between NeuroTensor and the MKL library’s `cblas_sgemm` function for square matrices ranging in size from 1x1 to 2000x2000. On average, NeuroTensor demonstrated twice the speed of MKL for larger matrices.

**Note:** MKL generally outperformed NeuroTensor for matrices smaller than 500x500. However, NeuroTensor exhibited significant performance gains for larger matrices.

<p align="center">
  <img src="benchmarking/NeuroTensor_vs_MKL.png" alt="NeuroTensor performance" width="85%">
</p>

