# BMD (Binary Matrix Decomposition)
Efficient Computation of Quantized Neural Networks by {−1,+1} Encoding Decomposition

run: nvcc BMD-2-2bit.cu -std=c++11 -lcublas && ./a.out

This benchmark performs 8192x8192x8192 matrix multiplications with our method and baseline kernel.
We implement the multiplication of two matrixes via our {-1, +1} encoding scheme on GTX 1080 GPU. In our experiments, 
the matrix multiplication after encoded by 2-bit has obtained at most ∼15.89× speedup ratio than baseline kernel.
