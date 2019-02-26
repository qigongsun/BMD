# BMD (Binary Matrix Decomposition)
## Motivations

This subrepository demonstrates the binary matrix decomposition described in the article:  
[Efficient Computation of Quantized Neural Networks by {− 1,+ 1} Encoding Decomposition.](https://openreview.net/pdf?id=rylfIYoucQ)

##  Matrix multiplication
run: 
<p><code>nvcc BMD-2-2bit.cu -std=c++11 -lcublas && ./a.out </code></p>

This benchmark performs 8192x8192x8192 matrix multiplications with our method and baseline kernel.
We implement the multiplication of two matrixes via our {-1, +1} encoding scheme on GTX 1080 GPU. 

**The matrix multiplication after encoded by 2-bit has obtained at most ∼15.89× speedup ratio than baseline kernel.**

## Requirements 
* Nvidia GPU
* CUDA

## Reference

[BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)

Code: [BinaryNet](https://github.com/MatthieuCourbariaux/BinaryNet)
