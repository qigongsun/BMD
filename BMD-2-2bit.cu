#include "binary_kernels.cu"
#include <iostream>
#include <chrono>
#include <cublas_v2.h>
using namespace std;
double res;
double fres;
int main() {
	int M = 8192;
	int N = 8192;
	int K = 8192;

	// prepare data
	float *A1 = (float*)malloc(M * N * sizeof(float));
	float *B1 = (float*)malloc(N * K * sizeof(float));
	float *A2 = (float*)malloc(M * N * sizeof(float));
	float *B2 = (float*)malloc(N * K * sizeof(float));

	float *AA = (float*)malloc(M * N * sizeof(float));
	float *BB = (float*)malloc(N * K * sizeof(float));

	for (int i = 0; i < M * N; i ++) {
		double x = (double)rand() / RAND_MAX;
		A1[i] = (x > 0.5) ? 1 : -1;

	}
	for (int i = 0; i < K * N; i ++) {
		double x = (double)rand() / RAND_MAX;
		B1[i] = (x > 0.5) ? 1 : -1;
	}

	for (int i = 0; i < M * N; i ++) {
		double x = (double)rand() / RAND_MAX;
		A2[i] = (x > 0.5) ? 1 : -1;

	}
	for (int i = 0; i < K * N; i ++) {
		double x = (double)rand() / RAND_MAX;
		B2[i] = (x > 0.5) ? 1 : -1;
	}

	for (int i = 0; i < M * N; i ++) {
		AA[i] = 2*A1[i]+A2[i];

	}
	for (int i = 0; i < K * N; i ++) {
		double x = (double)rand() / RAND_MAX;
		BB[i] = 2*B1[i]+B2[i];
	}

	// copy to cuda
	float *fA1, *fB1, *fA2, *fB2, *fAA, *fBB, *fC;
	cudaMalloc(&fA1, M * N * sizeof(float));
	cudaMalloc(&fB1, N * K * sizeof(float));
	cudaMalloc(&fA2, M * N * sizeof(float));
	cudaMalloc(&fB2, N * K * sizeof(float));
	cudaMalloc(&fAA, M * N * sizeof(float));
	cudaMalloc(&fBB, N * K * sizeof(float));
	cudaMalloc(&fC, M * K * sizeof(float));
	cudaMemcpy(fA1, A1, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fB1, B1, N * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fA2, A2, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fB2, B2, N * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fAA, AA, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fBB, BB, N * K * sizeof(float), cudaMemcpyHostToDevice);

	auto test_xnor = [&]() {
		unsigned int *Aconc1, *Bconc1, *Aconc2, *Bconc2;
		cudaMalloc(&Aconc1, M * N);
		cudaMalloc(&Bconc1, N * K);
		cudaMalloc(&Aconc2, M * N);
		cudaMalloc(&Bconc2, N * K);
		cudaMemset(fC, 0, M * K * sizeof(int));
		auto start = chrono::high_resolution_clock::now();

		int block = 64, grid = M * N / (block * 32)  + 1;
		concatenate_rows_kernel<<<grid, block>>>(fA1, Aconc1, M * N / 32);
		concatenate_rows_kernel<<<grid, block>>>(fA2, Aconc2, M * N / 32);

		grid = K / block + 1;
		concatenate_cols_kernel<<<grid, block>>>(fB1, Bconc1, N, K);
		concatenate_cols_kernel<<<grid, block>>>(fB2, Bconc2, N, K);
		cudaDeviceSynchronize();

		dim3 blockDim(16, 16);
		dim3 gridDim(M / 16 + 1, K / 16 + 1);
		//xnor_gemm<<<gridDim, blockDim>>>(Aconc, Bconc, fC, M, N / 32, K);
		xnor_add_gemm<<<gridDim, blockDim>>>(Aconc1, Aconc2, Bconc1, Bconc2, fC, M, N / 32, K);
		cudaDeviceSynchronize();

		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		cout << "XNOR GEMM kernel time: " << diff.count() << " s\n";
		res=diff.count();
		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	};
	float* result_xnor = test_xnor();


	auto test_gemm = [&]() {
		cudaMemset(fC, 0, M * K * sizeof(int));
		dim3 blockDim(16, 16);
		dim3 gridDim(M / 16 + 1, K / 16 + 1);
		auto start = chrono::high_resolution_clock::now();
		gemm<<<gridDim, blockDim>>>(fAA, fBB, fC, M, N, K);
		cudaDeviceSynchronize();
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		cout << "GEMM kernel time: " << diff.count() << " s\n";
		fres=diff.count();
		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	};
	float* result_gemm = test_gemm();


	auto check_result = [&](float* p1, float* p2) {
		for (int i = 0; i < N * N; i ++) {
			float diff = p1[i] - p2[i];
			if (fabs(diff) > 1e-6) {
				printf("%f\n", diff);
				return false;
			}
		}
		return true;
	};
	double p=fres/res;
	cout << "times: " << p<< " \n";
	printf("success: %d\n", check_result(result_gemm, result_xnor));
}
