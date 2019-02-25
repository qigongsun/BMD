#include <stdio.h>
#define BLOCK_SIZE 16

// CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// A is shape (m,n), B is shape (n,k) and C is shape (m,k)







__global__ void prod_real_virt(float* A, float* B, float* C,int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub

    int row = threadIdx.y;
    int col = threadIdx.x;



    // Each thread block computes one sub-matrix Csub of A
    float* Asub = &A[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of B
    float* Bsub = &B[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of C

    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];



    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m){

     if(Asub[row*k+col]<1) Bsub[row*k+col]=-1;
     else Bsub[row*k+col]=1;

     if(Asub[row*k+col]==-3 || Asub[row*k+col]==1 ) Csub[row*k+col]=-1;
     else Csub[row*k+col]=1;
     }


}



__global__ void prod_real_virt_3bit(float* A, float* B, float* C, float* D,int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub

    int row = threadIdx.y;
    int col = threadIdx.x;



    // Each thread block computes one sub-matrix Csub of A
    float* Asub = &A[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of B
    float* Bsub = &B[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of C

    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];


    // Each thread block computes one sub-matrix Csub of C

    float* Dsub = &D[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m){

     if(Asub[row*k+col]<1) Bsub[row*k+col]=-1;
     else Bsub[row*k+col]=1;

     if(Asub[row*k+col]<-3 ||(Asub[row*k+col]<5 && Asub[row*k+col]>-1)) Csub[row*k+col]=-1;
     else Csub[row*k+col]=1;

     if(Asub[row*k+col]==-3 || Asub[row*k+col]==1 || Asub[row*k+col]==-7|| Asub[row*k+col]==5) Dsub[row*k+col]=-1;
     else Dsub[row*k+col]=1;
     }


}





__global__ void prod_real_virt_4bit(float* A, float* B, float* C, float* D,float* E,int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub

    int row = threadIdx.y;
    int col = threadIdx.x;



    // Each thread block computes one sub-matrix Csub of A
    float* Asub = &A[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of B
    float* Bsub = &B[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of C

    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];


    // Each thread block computes one sub-matrix Dsub of D

    float* Dsub = &D[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of C

    float* Esub = &E[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m){

     if(Asub[row*k+col]<1) Bsub[row*k+col]=-1;
     else Bsub[row*k+col]=1;

     if(Asub[row*k+col]<-7 ||(Asub[row*k+col]<9 && Asub[row*k+col]>-1)) Csub[row*k+col]=-1;
     else Csub[row*k+col]=1;

     if(Asub[row*k+col]==-15 || Asub[row*k+col]==-13 || Asub[row*k+col]==-7|| Asub[row*k+col]==-5
     ||Asub[row*k+col]==1 || Asub[row*k+col]==3 || Asub[row*k+col]==9|| Asub[row*k+col]==11) Dsub[row*k+col]=-1;
     else Dsub[row*k+col]=1;


     if(Asub[row*k+col]==-15 || Asub[row*k+col]==-11 || Asub[row*k+col]==-7|| Asub[row*k+col]==-3
     ||Asub[row*k+col]==1 || Asub[row*k+col]==5|| Asub[row*k+col]==9|| Asub[row*k+col]==13) Esub[row*k+col]=-1;
     else Esub[row*k+col]=1;

     }


}





















__global__ void gemm(float* A, float* B, float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    float Cvalue = 0.0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        float* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        float* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += As[row][j] * Bs[j][col]; 
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element


    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = Cvalue;
}







__global__ void gemm_addition(float* A, float* B, float* C, float* D, float* E,int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub

    int row = threadIdx.y;
    int col = threadIdx.x;



    // Each thread block computes one sub-matrix Csub of A
    float* Asub = &A[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of B
    float* Bsub = &B[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of C

    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of D
    float* Dsub = &D[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Each thread block computes one sub-matrix Csub of E
    float* Esub = &E[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Esub[row*k+col] =4*Asub[row*k+col]+Bsub[row*k+col]+2*(Csub[row*k+col]+Dsub[row*k+col]);


}




























// 32 single float array ->  32 bits unsigned int
__device__ unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = concatenate(&a[i*32]);
}

__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n)
{   

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(j<n){
        float * array = new float[32];
        for(int i=0; i<m; i+=32){
            for(int k=0; k<32;k++) array[k] = a[j + n*(i+k)];
            b[j+n*i/32]=concatenate(array); 
        } 
        delete[] array;
    }
}

// 32 bits unsigned int -> 32 single float array
// TODO: the array allocation should not be done here
__device__ float* deconcatenate(unsigned int x)
{
    float * array = new float[32];
    
    for (int i = 0; i < 32; i++)    
    {   
        array[i] = (x & ( 1 << i )) >> i;
    }
    
    return array;
}

__global__ void deconcatenate_rows_kernel(unsigned int *a, float *b, int size)
{ 
    float * array;
    
    for(int i=0; i<size; i+=32)
    {
        array = deconcatenate(a[i/32]);
        for (int k=0;k<32;k++) b[i+k] = array[k];
        delete[] array;
    }
}

// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int Cvalue = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A
        unsigned int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B
        unsigned int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += __popc(As[row][j]^Bs[j][col]);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = -(2*(float)Cvalue-32*n);
}


__global__ void xnor_2_1bit_gemm(unsigned int* A_a, unsigned int* A_b, unsigned int* B_a,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_as[BLOCK_SIZE][BLOCK_SIZE];


    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_avalue = 0;

    unsigned int C_bvalue = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B_a
        unsigned int* B_asub = &B_a[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];


        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        B_as[row][col] = B_asub[row*k+col];


        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_avalue += __popc(A_as[row][j]^B_as[j][col]);
          C_bvalue += __popc(A_bs[row][j]^B_as[j][col]);

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-2*(2*(float)C_avalue-32*n)-(2*(float)C_bvalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_add_gemm(unsigned int* A_a, unsigned int* A_b, unsigned int* B_a,unsigned int* B_b,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_avalue = 0;
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B_a
        unsigned int* B_asub = &B_a[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];


        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        B_as[row][col] = B_asub[row*k+col];
        B_bs[row][col] = B_bsub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
        
          //C_avalue += (4*(__popc(A_as[row][j]^B_as[j][col]))
          //              +(__popc(A_bs[row][j]^B_bs[j][col]))
          //              +2*(__popc(A_as[row][j]^B_bs[j][col]))
          //              +2*(__popc(A_bs[row][j]^B_as[j][col])));

        
          C_avalue += __popc(A_as[row][j]^B_as[j][col]);
          C_bvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_dvalue += __popc(A_bs[row][j]^B_as[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
	//Csub[row*k+col] = -(2*(float)C_avalue-32*n);
        Csub[row*k+col] = (-4*(2*(float)C_avalue-32*n)-(2*(float)C_bvalue-32*n)-2*(2*(float)C_cvalue-32*n)-2*(2*(float)C_dvalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_3_1bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* B_c,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element

    unsigned int C_cvalue = 0;

    unsigned int C_fvalue = 0;

    unsigned int C_ivalue = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Asub of A_c
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];



        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];

        B_cs[row][col] = B_csub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_fvalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_ivalue += __popc(A_cs[row][j]^B_cs[j][col]);


        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    	Csub[row*k+col] = (-4*((2*(float)C_cvalue-32*n))-2*(2*(float)C_fvalue-32*n)-(2*(float)C_ivalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_3_2bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* B_b,unsigned int* B_c,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element

    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;

    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;

    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Asub of A_c
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];

        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {

          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);

          C_evalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_fvalue += __popc(A_bs[row][j]^B_cs[j][col]);

          C_hvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_ivalue += __popc(A_cs[row][j]^B_cs[j][col]);


        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (
    -8*((2*(float)C_bvalue-32*n))
    -4*((2*(float)C_cvalue-32*n)+(2*(float)C_evalue-32*n))
    -2*((2*(float)C_fvalue-32*n)+(2*(float)C_hvalue-32*n))
    -(2*(float)C_ivalue-32*n));
}



// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_3_3bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* B_a,unsigned int* B_b,unsigned int* B_c,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_avalue = 0;
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;
    unsigned int C_gvalue = 0;
    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Asub of A_c
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B_a
        unsigned int* B_asub = &B_a[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        B_as[row][col] = B_asub[row*k+col];
        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_avalue += __popc(A_as[row][j]^B_as[j][col]);
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_bs[row][j]^B_as[j][col]);
          C_evalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_fvalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_gvalue += __popc(A_cs[row][j]^B_as[j][col]);
          C_hvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_ivalue += __popc(A_cs[row][j]^B_cs[j][col]);


        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-16*(2*(float)C_avalue-32*n)
    -8*((2*(float)C_bvalue-32*n)+(2*(float)C_dvalue-32*n))
    -4*((2*(float)C_cvalue-32*n)+(2*(float)C_evalue-32*n)+(2*(float)C_gvalue-32*n))
    -2*((2*(float)C_fvalue-32*n)+(2*(float)C_hvalue-32*n))
    -(2*(float)C_ivalue-32*n));
}




// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_4_1bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* B_d,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element

    unsigned int C_dvalue = 0;
    unsigned int C_hvalue = 0;
    unsigned int C_lvalue = 0;
    unsigned int C_pvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_a
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];

        B_ds[row][col] = B_dsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);

          C_hvalue += __popc(A_bs[row][j]^B_ds[j][col]);
;
          C_lvalue += __popc(A_cs[row][j]^B_ds[j][col]);

          C_pvalue += __popc(A_ds[row][j]^B_ds[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-8*(2*(float)C_dvalue-32*n)
                       -4*(2*(float)C_hvalue-32*n)
                       -2*(2*(float)C_lvalue-32*n)
                       -(2*(float)C_pvalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_4_2bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d,  unsigned int* B_a,unsigned int* B_b,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_avalue = 0;
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;
    unsigned int C_gvalue = 0;
    unsigned int C_hvalue = 0;



    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_a
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_a
        unsigned int* B_asub = &B_a[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];


        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];

        B_as[row][col] = B_asub[row*k+col];
        B_bs[row][col] = B_bsub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_avalue += (__popc(A_as[row][j]^B_as[j][col]));
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_bs[row][j]^B_as[j][col]);
          C_dvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_evalue += __popc(A_cs[row][j]^B_as[j][col]);
          C_fvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_gvalue += __popc(A_ds[row][j]^B_as[j][col]);
          C_hvalue += __popc(A_ds[row][j]^B_bs[j][col]);

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-16*(2*(float)C_avalue-32*n)-8*((2*(float)C_bvalue-32*n)+(2*(float)C_cvalue-32*n))
    -4*((2*(float)C_dvalue-32*n)+(2*(float)C_evalue-32*n))-2*((2*(float)C_fvalue-32*n)+(2*(float)C_gvalue-32*n))
    -(2*(float)C_hvalue-32*n));
}

// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_4_3bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* B_b,unsigned int* B_c,unsigned int* B_d,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_fvalue = 0;
    unsigned int C_gvalue = 0;
    unsigned int C_hvalue = 0;
    unsigned int C_jvalue = 0;
    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;
    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;
    unsigned int C_pvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_a
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];

        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);

          C_fvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_gvalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_hvalue += __popc(A_bs[row][j]^B_ds[j][col]);

          C_jvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_kvalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_lvalue += __popc(A_cs[row][j]^B_ds[j][col]);

          C_nvalue += __popc(A_ds[row][j]^B_bs[j][col]);
          C_ovalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_pvalue += __popc(A_ds[row][j]^B_ds[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-32*(2*(float)C_bvalue-32*n)
                       -16*(2*(float)C_cvalue-32*n)
                       -8*(2*(float)C_dvalue-32*n)
                       -16*(2*(float)C_fvalue-32*n)
                       -8*(2*(float)C_gvalue-32*n)
                       -4*(2*(float)C_hvalue-32*n)
                       -8*(2*(float)C_jvalue-32*n)
                       -4*(2*(float)C_kvalue-32*n)
                       -2*(2*(float)C_lvalue-32*n)
                       -4*(2*(float)C_nvalue-32*n)
                       -2*(2*(float)C_ovalue-32*n)
                       -(2*(float)C_pvalue-32*n));
}



// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_4_4bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d,  unsigned int* B_a,unsigned int* B_b,unsigned int* B_c,unsigned int* B_d,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_avalue = 0;
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;
    unsigned int C_gvalue = 0;
    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;
    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;
    unsigned int C_mvalue = 0;
    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;
    unsigned int C_pvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_a
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_a
        unsigned int* B_asub = &B_a[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];

        B_as[row][col] = B_asub[row*k+col];
        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_avalue += __popc(A_as[row][j]^B_as[j][col]);
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);

          C_evalue += __popc(A_bs[row][j]^B_as[j][col]);
          C_fvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_gvalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_hvalue += __popc(A_bs[row][j]^B_ds[j][col]);

          C_ivalue += __popc(A_cs[row][j]^B_as[j][col]);
          C_jvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_kvalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_lvalue += __popc(A_cs[row][j]^B_ds[j][col]);

          C_mvalue += __popc(A_ds[row][j]^B_as[j][col]);
          C_nvalue += __popc(A_ds[row][j]^B_bs[j][col]);
          C_ovalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_pvalue += __popc(A_ds[row][j]^B_ds[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-64*(2*(float)C_avalue-32*n)
                       -32*(2*(float)C_bvalue-32*n)
                       -16*(2*(float)C_cvalue-32*n)
                       -8*(2*(float)C_dvalue-32*n)
                       -32*(2*(float)C_evalue-32*n)
                       -16*(2*(float)C_fvalue-32*n)
                       -8*(2*(float)C_gvalue-32*n)
                       -4*(2*(float)C_hvalue-32*n)
                       -16*(2*(float)C_ivalue-32*n)
                       -8*(2*(float)C_jvalue-32*n)
                       -4*(2*(float)C_kvalue-32*n)
                       -2*(2*(float)C_lvalue-32*n)
                       -8*(2*(float)C_mvalue-32*n)
                       -4*(2*(float)C_nvalue-32*n)
                       -2*(2*(float)C_ovalue-32*n)
                       -(2*(float)C_pvalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_5_1bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* B_e,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_evalue = 0;

    unsigned int C_jvalue = 0;

    unsigned int C_ovalue = 0;

    unsigned int C_tvalue = 0;

    unsigned int C_yvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];


        B_es[row][col] = B_esub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);


          C_jvalue += __popc(A_bs[row][j]^B_es[j][col]);

          C_ovalue += __popc(A_cs[row][j]^B_es[j][col]);

          C_tvalue += __popc(A_ds[row][j]^B_es[j][col]);

          C_yvalue += __popc(A_es[row][j]^B_es[j][col]);

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (
                       -16*(2*(float)C_evalue-32*n)
                       -8*(2*(float)C_jvalue-32*n)
                       -4*(2*(float)C_ovalue-32*n)
                       -2*(2*(float)C_tvalue-32*n)
                       -(2*(float)C_yvalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_5_2bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* B_d,unsigned int* B_e,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;
    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;
    unsigned int C_svalue = 0;
    unsigned int C_tvalue = 0;
    unsigned int C_xvalue = 0;
    unsigned int C_yvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];


        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);
          C_ivalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_jvalue += __popc(A_bs[row][j]^B_es[j][col]);
          C_nvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_ovalue += __popc(A_cs[row][j]^B_es[j][col]);
          C_svalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_tvalue += __popc(A_ds[row][j]^B_es[j][col]);
          C_xvalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_yvalue += __popc(A_es[row][j]^B_es[j][col]);

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = ( -32*(2*(float)C_dvalue-32*n)
                       -16*(2*(float)C_evalue-32*n)
                       -16*(2*(float)C_ivalue-32*n)
                       -8*(2*(float)C_jvalue-32*n)
                       -8*(2*(float)C_nvalue-32*n)
                       -4*(2*(float)C_ovalue-32*n)
                       -4*(2*(float)C_svalue-32*n)
                       -2*(2*(float)C_tvalue-32*n)
                       -2*(2*(float)C_xvalue-32*n)
                       -(2*(float)C_yvalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_5_3bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e, unsigned int* B_c,unsigned int* B_d,unsigned int* B_e,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;

    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;

    unsigned int C_mvalue = 0;
    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;

    unsigned int C_rvalue = 0;
    unsigned int C_svalue = 0;
    unsigned int C_tvalue = 0;

    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;
    unsigned int C_yvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];





        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];


        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);


          C_hvalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_ivalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_jvalue += __popc(A_bs[row][j]^B_es[j][col]);

          C_mvalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_nvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_ovalue += __popc(A_cs[row][j]^B_es[j][col]);


          C_rvalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_svalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_tvalue += __popc(A_ds[row][j]^B_es[j][col]);

          C_wvalue += __popc(A_es[row][j]^B_cs[j][col]);
          C_xvalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_yvalue += __popc(A_es[row][j]^B_es[j][col]);

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (
                       -64*(2*(float)C_cvalue-32*n)
                       -32*(2*(float)C_dvalue-32*n)
                       -16*(2*(float)C_evalue-32*n)

                       -32*(2*(float)C_hvalue-32*n)
                       -16*(2*(float)C_ivalue-32*n)
                       -8*(2*(float)C_jvalue-32*n)

                       -16*(2*(float)C_mvalue-32*n)
                       -8*(2*(float)C_nvalue-32*n)
                       -4*(2*(float)C_ovalue-32*n)

                       -8*(2*(float)C_rvalue-32*n)
                       -4*(2*(float)C_svalue-32*n)
                       -2*(2*(float)C_tvalue-32*n)

                       -4*(2*(float)C_wvalue-32*n)
                       -2*(2*(float)C_xvalue-32*n)
                       -(2*(float)C_yvalue-32*n));
}



// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_5_4bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e, unsigned int* B_b,unsigned int* B_c,unsigned int* B_d,unsigned int* B_e,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;

    unsigned int C_gvalue = 0;
    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;

    unsigned int C_lvalue = 0;
    unsigned int C_mvalue = 0;
    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;

    unsigned int C_qvalue = 0;
    unsigned int C_rvalue = 0;
    unsigned int C_svalue = 0;
    unsigned int C_tvalue = 0;

    unsigned int C_vvalue = 0;
    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;
    unsigned int C_yvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];




        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];


        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);


          C_gvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_hvalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_ivalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_jvalue += __popc(A_bs[row][j]^B_es[j][col]);

          C_lvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_mvalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_nvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_ovalue += __popc(A_cs[row][j]^B_es[j][col]);

          C_qvalue += __popc(A_ds[row][j]^B_bs[j][col]);
          C_rvalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_svalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_tvalue += __popc(A_ds[row][j]^B_es[j][col]);

          C_vvalue += __popc(A_es[row][j]^B_bs[j][col]);
          C_wvalue += __popc(A_es[row][j]^B_cs[j][col]);
          C_xvalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_yvalue += __popc(A_es[row][j]^B_es[j][col]);

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-128*(2*(float)C_bvalue-32*n)
                       -64*(2*(float)C_cvalue-32*n)
                       -32*(2*(float)C_dvalue-32*n)
                       -16*(2*(float)C_evalue-32*n)

                       -64*(2*(float)C_gvalue-32*n)
                       -32*(2*(float)C_hvalue-32*n)
                       -16*(2*(float)C_ivalue-32*n)
                       -8*(2*(float)C_jvalue-32*n)

                       -32*(2*(float)C_lvalue-32*n)
                       -16*(2*(float)C_mvalue-32*n)
                       -8*(2*(float)C_nvalue-32*n)
                       -4*(2*(float)C_ovalue-32*n)

                       -16*(2*(float)C_qvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)
                       -4*(2*(float)C_svalue-32*n)
                       -2*(2*(float)C_tvalue-32*n)

                       -8*(2*(float)C_vvalue-32*n)
                       -4*(2*(float)C_wvalue-32*n)
                       -2*(2*(float)C_xvalue-32*n)
                       -(2*(float)C_yvalue-32*n));
}




// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_5_5bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,  unsigned int* B_a,unsigned int* B_b,unsigned int* B_c,unsigned int* B_d,unsigned int* B_e,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_avalue = 0;
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;
    unsigned int C_gvalue = 0;
    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;
    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;
    unsigned int C_mvalue = 0;
    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;
    unsigned int C_pvalue = 0;
    unsigned int C_qvalue = 0;
    unsigned int C_rvalue = 0;
    unsigned int C_svalue = 0;
    unsigned int C_tvalue = 0;
    unsigned int C_uvalue = 0;
    unsigned int C_vvalue = 0;
    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;
    unsigned int C_yvalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_a
        unsigned int* B_asub = &B_a[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];

        B_as[row][col] = B_asub[row*k+col];
        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_avalue += __popc(A_as[row][j]^B_as[j][col]);
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);

          C_fvalue += __popc(A_bs[row][j]^B_as[j][col]);
          C_gvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_hvalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_ivalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_jvalue += __popc(A_bs[row][j]^B_es[j][col]);

          C_kvalue += __popc(A_cs[row][j]^B_as[j][col]);
          C_lvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_mvalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_nvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_ovalue += __popc(A_cs[row][j]^B_es[j][col]);

          C_pvalue += __popc(A_ds[row][j]^B_as[j][col]);
          C_qvalue += __popc(A_ds[row][j]^B_bs[j][col]);
          C_rvalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_svalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_tvalue += __popc(A_ds[row][j]^B_es[j][col]);

          C_uvalue += __popc(A_es[row][j]^B_as[j][col]);
          C_vvalue += __popc(A_es[row][j]^B_bs[j][col]);
          C_wvalue += __popc(A_es[row][j]^B_cs[j][col]);
          C_xvalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_yvalue += __popc(A_es[row][j]^B_es[j][col]);

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-256*(2*(float)C_avalue-32*n)
                       -128*(2*(float)C_bvalue-32*n)
                       -64*(2*(float)C_cvalue-32*n)
                       -32*(2*(float)C_dvalue-32*n)
                       -16*(2*(float)C_evalue-32*n)
                       -128*(2*(float)C_fvalue-32*n)
                       -64*(2*(float)C_gvalue-32*n)
                       -32*(2*(float)C_hvalue-32*n)
                       -16*(2*(float)C_ivalue-32*n)
                       -8*(2*(float)C_jvalue-32*n)
                       -64*(2*(float)C_kvalue-32*n)
                       -32*(2*(float)C_lvalue-32*n)
                       -16*(2*(float)C_mvalue-32*n)
                       -8*(2*(float)C_nvalue-32*n)
                       -4*(2*(float)C_ovalue-32*n)
                       -32*(2*(float)C_pvalue-32*n)
                       -16*(2*(float)C_qvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)
                       -4*(2*(float)C_svalue-32*n)
                       -2*(2*(float)C_tvalue-32*n)
                       -16*(2*(float)C_uvalue-32*n)
                       -8*(2*(float)C_vvalue-32*n)
                       -4*(2*(float)C_wvalue-32*n)
                       -2*(2*(float)C_xvalue-32*n)
                       -(2*(float)C_yvalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_6_1bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* A_f,unsigned int* B_f,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_fs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_fs[BLOCK_SIZE][BLOCK_SIZE];


    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_fvalue = 0;
    unsigned int C_lvalue = 0;
    unsigned int C_rvalue = 0;
    unsigned int C_xvalue = 0;
    unsigned int C_davalue = 0;
    unsigned int C_javalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_fsub = &A_f[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* B_fsub = &B_f[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];
        A_fs[row][col] = A_fsub[row*n+col];

        B_fs[row][col] = B_fsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_fvalue += __popc(A_as[row][j]^B_fs[j][col]);

          C_lvalue += __popc(A_bs[row][j]^B_fs[j][col]);

          C_rvalue += __popc(A_cs[row][j]^B_fs[j][col]);

          C_xvalue += __popc(A_ds[row][j]^B_fs[j][col]);

          C_davalue += __popc(A_es[row][j]^B_fs[j][col]);

          C_javalue += __popc(A_fs[row][j]^B_fs[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-32*(2*(float)C_fvalue-32*n)
                       -16*(2*(float)C_lvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)
                       -4*(2*(float)C_xvalue-32*n)
                       -2*(2*(float)C_davalue-32*n)
                       -(2*(float)C_javalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_6_2bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* A_f,unsigned int* B_e,unsigned int* B_f,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_fs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_fs[BLOCK_SIZE][BLOCK_SIZE];


    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;

    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;

    unsigned int C_qvalue = 0;
    unsigned int C_rvalue = 0;

    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;

    unsigned int C_cavalue = 0;
    unsigned int C_davalue = 0;

    unsigned int C_iavalue = 0;
    unsigned int C_javalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_fsub = &A_f[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_fsub = &B_f[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];
        A_fs[row][col] = A_fsub[row*n+col];

        B_es[row][col] = B_esub[row*k+col];
        B_fs[row][col] = B_fsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);
          C_fvalue += __popc(A_as[row][j]^B_fs[j][col]);

          C_kvalue += __popc(A_bs[row][j]^B_es[j][col]);
          C_lvalue += __popc(A_bs[row][j]^B_fs[j][col]);

          C_qvalue += __popc(A_cs[row][j]^B_es[j][col]);
          C_rvalue += __popc(A_cs[row][j]^B_fs[j][col]);

          C_wvalue += __popc(A_ds[row][j]^B_es[j][col]);
          C_xvalue += __popc(A_ds[row][j]^B_fs[j][col]);

          C_cavalue += __popc(A_es[row][j]^B_es[j][col]);
          C_davalue += __popc(A_es[row][j]^B_fs[j][col]);

          C_iavalue += __popc(A_fs[row][j]^B_es[j][col]);
          C_javalue += __popc(A_fs[row][j]^B_fs[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-64*(2*(float)C_evalue-32*n)
                       -32*(2*(float)C_fvalue-32*n)

                       -32*(2*(float)C_kvalue-32*n)
                       -16*(2*(float)C_lvalue-32*n)

                       -16*(2*(float)C_qvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)

                       -8*(2*(float)C_wvalue-32*n)
                       -4*(2*(float)C_xvalue-32*n)

                       -4*(2*(float)C_cavalue-32*n)
                       -2*(2*(float)C_davalue-32*n)

                       -2*(2*(float)C_iavalue-32*n)
                       -(2*(float)C_javalue-32*n));
}



// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_6_3bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* A_f,unsigned int* B_d,unsigned int* B_e,unsigned int* B_f,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_fs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_fs[BLOCK_SIZE][BLOCK_SIZE];


    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;

    unsigned int C_jvalue = 0;
    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;

    unsigned int C_pvalue = 0;
    unsigned int C_qvalue = 0;
    unsigned int C_rvalue = 0;

    unsigned int C_vvalue = 0;
    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;

    unsigned int C_bavalue = 0;
    unsigned int C_cavalue = 0;
    unsigned int C_davalue = 0;

    unsigned int C_havalue = 0;
    unsigned int C_iavalue = 0;
    unsigned int C_javalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_fsub = &A_f[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_fsub = &B_f[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];
        A_fs[row][col] = A_fsub[row*n+col];

        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        B_fs[row][col] = B_fsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);
          C_fvalue += __popc(A_as[row][j]^B_fs[j][col]);

          C_jvalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_kvalue += __popc(A_bs[row][j]^B_es[j][col]);
          C_lvalue += __popc(A_bs[row][j]^B_fs[j][col]);

          C_pvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_qvalue += __popc(A_cs[row][j]^B_es[j][col]);
          C_rvalue += __popc(A_cs[row][j]^B_fs[j][col]);

          C_vvalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_wvalue += __popc(A_ds[row][j]^B_es[j][col]);
          C_xvalue += __popc(A_ds[row][j]^B_fs[j][col]);

          C_bavalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_cavalue += __popc(A_es[row][j]^B_es[j][col]);
          C_davalue += __popc(A_es[row][j]^B_fs[j][col]);

          C_havalue += __popc(A_fs[row][j]^B_ds[j][col]);
          C_iavalue += __popc(A_fs[row][j]^B_es[j][col]);
          C_javalue += __popc(A_fs[row][j]^B_fs[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-128*(2*(float)C_dvalue-32*n)
                       -64*(2*(float)C_evalue-32*n)
                       -32*(2*(float)C_fvalue-32*n)

                       -64*(2*(float)C_jvalue-32*n)
                       -32*(2*(float)C_kvalue-32*n)
                       -16*(2*(float)C_lvalue-32*n)

                       -32*(2*(float)C_pvalue-32*n)
                       -16*(2*(float)C_qvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)

                       -16*(2*(float)C_vvalue-32*n)
                       -8*(2*(float)C_wvalue-32*n)
                       -4*(2*(float)C_xvalue-32*n)

                       -8*(2*(float)C_bavalue-32*n)
                       -4*(2*(float)C_cavalue-32*n)
                       -2*(2*(float)C_davalue-32*n)

                       -4*(2*(float)C_havalue-32*n)
                       -2*(2*(float)C_iavalue-32*n)
                       -(2*(float)C_javalue-32*n));
}



// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_6_4bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* A_f,unsigned int* B_c,unsigned int* B_d,unsigned int* B_e,unsigned int* B_f,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_fs[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_fs[BLOCK_SIZE][BLOCK_SIZE];


    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;

    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;
    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;

    unsigned int C_ovalue = 0;
    unsigned int C_pvalue = 0;
    unsigned int C_qvalue = 0;
    unsigned int C_rvalue = 0;

    unsigned int C_uvalue = 0;
    unsigned int C_vvalue = 0;
    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;

    unsigned int C_aavalue = 0;
    unsigned int C_bavalue = 0;
    unsigned int C_cavalue = 0;
    unsigned int C_davalue = 0;

    unsigned int C_gavalue = 0;
    unsigned int C_havalue = 0;
    unsigned int C_iavalue = 0;
    unsigned int C_javalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_fsub = &A_f[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_fsub = &B_f[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];
        A_fs[row][col] = A_fsub[row*n+col];

        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        B_fs[row][col] = B_fsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);
          C_fvalue += __popc(A_as[row][j]^B_fs[j][col]);

          C_ivalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_jvalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_kvalue += __popc(A_bs[row][j]^B_es[j][col]);
          C_lvalue += __popc(A_bs[row][j]^B_fs[j][col]);

          C_ovalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_pvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_qvalue += __popc(A_cs[row][j]^B_es[j][col]);
          C_rvalue += __popc(A_cs[row][j]^B_fs[j][col]);

          C_uvalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_vvalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_wvalue += __popc(A_ds[row][j]^B_es[j][col]);
          C_xvalue += __popc(A_ds[row][j]^B_fs[j][col]);

          C_aavalue += __popc(A_es[row][j]^B_cs[j][col]);
          C_bavalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_cavalue += __popc(A_es[row][j]^B_es[j][col]);
          C_davalue += __popc(A_es[row][j]^B_fs[j][col]);

          C_gavalue += __popc(A_fs[row][j]^B_cs[j][col]);
          C_havalue += __popc(A_fs[row][j]^B_ds[j][col]);
          C_iavalue += __popc(A_fs[row][j]^B_es[j][col]);
          C_javalue += __popc(A_fs[row][j]^B_fs[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-256*(2*(float)C_cvalue-32*n)
                       -128*(2*(float)C_dvalue-32*n)
                       -64*(2*(float)C_evalue-32*n)
                       -32*(2*(float)C_fvalue-32*n)

                       -128*(2*(float)C_ivalue-32*n)
                       -64*(2*(float)C_jvalue-32*n)
                       -32*(2*(float)C_kvalue-32*n)
                       -16*(2*(float)C_lvalue-32*n)

                       -64*(2*(float)C_ovalue-32*n)
                       -32*(2*(float)C_pvalue-32*n)
                       -16*(2*(float)C_qvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)

                       -32*(2*(float)C_uvalue-32*n)
                       -16*(2*(float)C_vvalue-32*n)
                       -8*(2*(float)C_wvalue-32*n)
                       -4*(2*(float)C_xvalue-32*n)

                       -16*(2*(float)C_aavalue-32*n)
                       -8*(2*(float)C_bavalue-32*n)
                       -4*(2*(float)C_cavalue-32*n)
                       -2*(2*(float)C_davalue-32*n)

                       -8*(2*(float)C_gavalue-32*n)
                       -4*(2*(float)C_havalue-32*n)
                       -2*(2*(float)C_iavalue-32*n)
                       -(2*(float)C_javalue-32*n));
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_6_5bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* A_f,unsigned int* B_b,unsigned int* B_c,unsigned int* B_d,unsigned int* B_e,unsigned int* B_f,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_fs[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_fs[BLOCK_SIZE][BLOCK_SIZE];


    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;

    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;
    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;

    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;
    unsigned int C_pvalue = 0;
    unsigned int C_qvalue = 0;
    unsigned int C_rvalue = 0;

    unsigned int C_tvalue = 0;
    unsigned int C_uvalue = 0;
    unsigned int C_vvalue = 0;
    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;

    unsigned int C_zvalue = 0;
    unsigned int C_aavalue = 0;
    unsigned int C_bavalue = 0;
    unsigned int C_cavalue = 0;
    unsigned int C_davalue = 0;

    unsigned int C_favalue = 0;
    unsigned int C_gavalue = 0;
    unsigned int C_havalue = 0;
    unsigned int C_iavalue = 0;
    unsigned int C_javalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_fsub = &A_f[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_fsub = &B_f[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];
        A_fs[row][col] = A_fsub[row*n+col];

        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        B_fs[row][col] = B_fsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);
          C_fvalue += __popc(A_as[row][j]^B_fs[j][col]);

          C_hvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_ivalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_jvalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_kvalue += __popc(A_bs[row][j]^B_es[j][col]);
          C_lvalue += __popc(A_bs[row][j]^B_fs[j][col]);

          C_nvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_ovalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_pvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_qvalue += __popc(A_cs[row][j]^B_es[j][col]);
          C_rvalue += __popc(A_cs[row][j]^B_fs[j][col]);

          C_tvalue += __popc(A_ds[row][j]^B_bs[j][col]);
          C_uvalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_vvalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_wvalue += __popc(A_ds[row][j]^B_es[j][col]);
          C_xvalue += __popc(A_ds[row][j]^B_fs[j][col]);

          C_zvalue += __popc(A_es[row][j]^B_bs[j][col]);
          C_aavalue += __popc(A_es[row][j]^B_cs[j][col]);
          C_bavalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_cavalue += __popc(A_es[row][j]^B_es[j][col]);
          C_davalue += __popc(A_es[row][j]^B_fs[j][col]);

          C_favalue += __popc(A_fs[row][j]^B_bs[j][col]);
          C_gavalue += __popc(A_fs[row][j]^B_cs[j][col]);
          C_havalue += __popc(A_fs[row][j]^B_ds[j][col]);
          C_iavalue += __popc(A_fs[row][j]^B_es[j][col]);
          C_javalue += __popc(A_fs[row][j]^B_fs[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-512*(2*(float)C_bvalue-32*n)
                       -256*(2*(float)C_cvalue-32*n)
                       -128*(2*(float)C_dvalue-32*n)
                       -64*(2*(float)C_evalue-32*n)
                       -32*(2*(float)C_fvalue-32*n)

                       -256*(2*(float)C_hvalue-32*n)
                       -128*(2*(float)C_ivalue-32*n)
                       -64*(2*(float)C_jvalue-32*n)
                       -32*(2*(float)C_kvalue-32*n)
                       -16*(2*(float)C_lvalue-32*n)

                       -128*(2*(float)C_nvalue-32*n)
                       -64*(2*(float)C_ovalue-32*n)
                       -32*(2*(float)C_pvalue-32*n)
                       -16*(2*(float)C_qvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)

                       -64*(2*(float)C_tvalue-32*n)
                       -32*(2*(float)C_uvalue-32*n)
                       -16*(2*(float)C_vvalue-32*n)
                       -8*(2*(float)C_wvalue-32*n)
                       -4*(2*(float)C_xvalue-32*n)

                       -32*(2*(float)C_zvalue-32*n)
                       -16*(2*(float)C_aavalue-32*n)
                       -8*(2*(float)C_bavalue-32*n)
                       -4*(2*(float)C_cavalue-32*n)
                       -2*(2*(float)C_davalue-32*n)

                       -16*(2*(float)C_favalue-32*n)
                       -8*(2*(float)C_gavalue-32*n)
                       -4*(2*(float)C_havalue-32*n)
                       -2*(2*(float)C_iavalue-32*n)
                       -(2*(float)C_javalue-32*n));
}




// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_6_6bit_gemm(unsigned int* A_a, unsigned int* A_b,unsigned int* A_c, unsigned int* A_d, unsigned int* A_e,unsigned int* A_f,  unsigned int* B_a,unsigned int* B_b,unsigned int* B_c,unsigned int* B_d,unsigned int* B_e,unsigned int* B_f,float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int A_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int A_fs[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ unsigned int B_as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_ds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_es[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int B_fs[BLOCK_SIZE][BLOCK_SIZE];


    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int C_avalue = 0;
    unsigned int C_bvalue = 0;
    unsigned int C_cvalue = 0;
    unsigned int C_dvalue = 0;
    unsigned int C_evalue = 0;
    unsigned int C_fvalue = 0;
    unsigned int C_gvalue = 0;
    unsigned int C_hvalue = 0;
    unsigned int C_ivalue = 0;
    unsigned int C_jvalue = 0;
    unsigned int C_kvalue = 0;
    unsigned int C_lvalue = 0;
    unsigned int C_mvalue = 0;
    unsigned int C_nvalue = 0;
    unsigned int C_ovalue = 0;
    unsigned int C_pvalue = 0;
    unsigned int C_qvalue = 0;
    unsigned int C_rvalue = 0;
    unsigned int C_svalue = 0;
    unsigned int C_tvalue = 0;
    unsigned int C_uvalue = 0;
    unsigned int C_vvalue = 0;
    unsigned int C_wvalue = 0;
    unsigned int C_xvalue = 0;
    unsigned int C_yvalue = 0;
    unsigned int C_zvalue = 0;
    unsigned int C_aavalue = 0;
    unsigned int C_bavalue = 0;
    unsigned int C_cavalue = 0;
    unsigned int C_davalue = 0;
    unsigned int C_eavalue = 0;
    unsigned int C_favalue = 0;
    unsigned int C_gavalue = 0;
    unsigned int C_havalue = 0;
    unsigned int C_iavalue = 0;
    unsigned int C_javalue = 0;


    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A_a
        unsigned int* A_asub = &A_a[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Asub of A_b
        unsigned int* A_bsub = &A_b[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_csub = &A_c[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_dsub = &A_d[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_esub = &A_e[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        unsigned int* A_fsub = &A_f[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];


        // Get sub-matrix Bsub of B_a
        unsigned int* B_asub = &B_a[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_b
        unsigned int* B_bsub = &B_b[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        // Get sub-matrix Bsub of B_c
        unsigned int* B_csub = &B_c[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B_d
        unsigned int* B_dsub = &B_d[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        unsigned int* B_esub = &B_e[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        unsigned int* B_fsub = &B_f[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_as[row][col] = A_asub[row*n+col];
        A_bs[row][col] = A_bsub[row*n+col];
        A_cs[row][col] = A_csub[row*n+col];
        A_ds[row][col] = A_dsub[row*n+col];
        A_es[row][col] = A_esub[row*n+col];
        A_fs[row][col] = A_fsub[row*n+col];

        B_as[row][col] = B_asub[row*k+col];
        B_bs[row][col] = B_bsub[row*k+col];
        B_cs[row][col] = B_csub[row*k+col];
        B_ds[row][col] = B_dsub[row*k+col];
        B_es[row][col] = B_esub[row*k+col];
        B_fs[row][col] = B_fsub[row*k+col];
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          C_avalue += __popc(A_as[row][j]^B_as[j][col]);
          C_bvalue += __popc(A_as[row][j]^B_bs[j][col]);
          C_cvalue += __popc(A_as[row][j]^B_cs[j][col]);
          C_dvalue += __popc(A_as[row][j]^B_ds[j][col]);
          C_evalue += __popc(A_as[row][j]^B_es[j][col]);
          C_fvalue += __popc(A_as[row][j]^B_fs[j][col]);

          C_gvalue += __popc(A_bs[row][j]^B_as[j][col]);
          C_hvalue += __popc(A_bs[row][j]^B_bs[j][col]);
          C_ivalue += __popc(A_bs[row][j]^B_cs[j][col]);
          C_jvalue += __popc(A_bs[row][j]^B_ds[j][col]);
          C_kvalue += __popc(A_bs[row][j]^B_es[j][col]);
          C_lvalue += __popc(A_bs[row][j]^B_fs[j][col]);

          C_mvalue += __popc(A_cs[row][j]^B_as[j][col]);
          C_nvalue += __popc(A_cs[row][j]^B_bs[j][col]);
          C_ovalue += __popc(A_cs[row][j]^B_cs[j][col]);
          C_pvalue += __popc(A_cs[row][j]^B_ds[j][col]);
          C_qvalue += __popc(A_cs[row][j]^B_es[j][col]);
          C_rvalue += __popc(A_cs[row][j]^B_fs[j][col]);

          C_svalue += __popc(A_ds[row][j]^B_as[j][col]);
          C_tvalue += __popc(A_ds[row][j]^B_bs[j][col]);
          C_uvalue += __popc(A_ds[row][j]^B_cs[j][col]);
          C_vvalue += __popc(A_ds[row][j]^B_ds[j][col]);
          C_wvalue += __popc(A_ds[row][j]^B_es[j][col]);
          C_xvalue += __popc(A_ds[row][j]^B_fs[j][col]);

          C_yvalue += __popc(A_es[row][j]^B_as[j][col]);
          C_zvalue += __popc(A_es[row][j]^B_bs[j][col]);
          C_aavalue += __popc(A_es[row][j]^B_cs[j][col]);
          C_bavalue += __popc(A_es[row][j]^B_ds[j][col]);
          C_cavalue += __popc(A_es[row][j]^B_es[j][col]);
          C_davalue += __popc(A_es[row][j]^B_fs[j][col]);

          C_eavalue += __popc(A_fs[row][j]^B_as[j][col]);
          C_favalue += __popc(A_fs[row][j]^B_bs[j][col]);
          C_gavalue += __popc(A_fs[row][j]^B_cs[j][col]);
          C_havalue += __popc(A_fs[row][j]^B_ds[j][col]);
          C_iavalue += __popc(A_fs[row][j]^B_es[j][col]);
          C_javalue += __popc(A_fs[row][j]^B_fs[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = (-1024*(2*(float)C_avalue-32*n)
                       -512*(2*(float)C_bvalue-32*n)
                       -256*(2*(float)C_cvalue-32*n)
                       -128*(2*(float)C_dvalue-32*n)
                       -64*(2*(float)C_evalue-32*n)
                       -32*(2*(float)C_fvalue-32*n)
                       
                       -512*(2*(float)C_gvalue-32*n)
                       -256*(2*(float)C_hvalue-32*n)
                       -128*(2*(float)C_ivalue-32*n)
                       -64*(2*(float)C_jvalue-32*n)
                       -32*(2*(float)C_kvalue-32*n)
                       -16*(2*(float)C_lvalue-32*n)
                       
                       -256*(2*(float)C_mvalue-32*n)
                       -128*(2*(float)C_nvalue-32*n)
                       -64*(2*(float)C_ovalue-32*n)
                       -32*(2*(float)C_pvalue-32*n)
                       -16*(2*(float)C_qvalue-32*n)
                       -8*(2*(float)C_rvalue-32*n)
                       
                       -128*(2*(float)C_svalue-32*n)
                       -64*(2*(float)C_tvalue-32*n)
                       -32*(2*(float)C_uvalue-32*n)
                       -16*(2*(float)C_vvalue-32*n)
                       -8*(2*(float)C_wvalue-32*n)
                       -4*(2*(float)C_xvalue-32*n)
                       
                       
                       -64*(2*(float)C_yvalue-32*n)
                       -32*(2*(float)C_zvalue-32*n)                       
                       -16*(2*(float)C_aavalue-32*n)
                       -8*(2*(float)C_bavalue-32*n)
                       -4*(2*(float)C_cavalue-32*n)
                       -2*(2*(float)C_davalue-32*n)
                       
                       -32*(2*(float)C_eavalue-32*n)                       
                       -16*(2*(float)C_favalue-32*n)                       
                       -8*(2*(float)C_gavalue-32*n)
                       -4*(2*(float)C_havalue-32*n)
                       -2*(2*(float)C_iavalue-32*n)
                       -(2*(float)C_javalue-32*n));
}


