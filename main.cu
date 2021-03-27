#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

#define CUDA_CHECK(status) (assert(status == cudaSuccess))
#define threads_per_block 1024

__global__ void column_reduce(float * matrix, float * result, int m /* lines */, int n /* rows*/, int num_blocks_per_line) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int curr_block_limit = (blockIdx.x == (num_blocks_per_line - 1)) ? n % threads_per_block : blockDim.x;
    unsigned int line_block_idx = blockIdx.x % num_blocks_per_line;
    //                        matrix line                  +        previous blocks in line     + id in block
    unsigned int i = blockIdx.x / num_blocks_per_line  * n + line_block_idx * threads_per_block + threadIdx.x;
    
    if (threadIdx.x < curr_block_limit) {
        // if (blockIdx.x != 0) printf("Thread %d Block %d has i=%d %d\n", threadIdx.x, blockIdx.x, i, curr_block_limit);
        if (i < m * n) sdata[tid] = matrix[i];
        __syncthreads();

        for (unsigned int s = 1; s < blockDim.x; s *= 2) {
            if (tid % (2*s) == 0) {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads();
        }

        if (tid == 0) result[blockIdx.x] = sdata[0];
    }

}

int main(int argc, char * argv[])  {
    if (argc < 3) {
        printf("Usage: %s <m> <n>\n", argv[0]);
        return 0;
    }

    int m = atoi(argv[1]), n = atoi(argv[2]);
    
    int num_blocks_per_line = n / threads_per_block + (n % threads_per_block ? 1 : 0);
    int tot_num_blocks = m * num_blocks_per_line;

    printf("Running with %d threads per block, %d num blocks per line, %d tot number blocks\n", threads_per_block, num_blocks_per_line, tot_num_blocks);

    srand(time(NULL)); // seed 

    // create row-major matrix m x n
    float * matrix = (float *) malloc(sizeof(float) * m * n); // m x n
    // create array to store result
    float * result_gpu = (float *) malloc(sizeof(float) * tot_num_blocks); // tot_num_blocks x 1
    float * result_cpu = (float *) malloc(sizeof(float) * m); // validation

    printf("Populating array \n");
    // populate the array 
    for (int i = 0; i < m * n; i++)
        matrix[i] = 1.0 / ((rand() % 977) + 1);
    

    printf("Calculating final result\n");
    // calculate cpu result
    for (int i = 0; i < m; i++) {
        int row = i * n;
        result_cpu[i] = 0;
        for (int j = 0; j < n; j++) {
            result_cpu[i] += matrix[j + row];
        }
    }

    // printf("--- Result CPU ---\n");
    // for (int i = 0; i < m; i++) {
    //     printf("%0.4f = ", result_cpu[i]);
    //     for (int j = 0; j < n; j++) {
    //         printf("%0.4f ", matrix[j + i * n]);
    //     }
    //     printf("\n");
    // }

    printf("Allocating GPU memory\n");
    // allocate gpu memory
    float * matrix_gpu, * device_result;
    CUDA_CHECK(cudaMalloc(&matrix_gpu, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(float) * tot_num_blocks));
    
    // move matrix into gpu
    CUDA_CHECK(cudaMemcpy(matrix_gpu, matrix, m * n * sizeof(float), cudaMemcpyHostToDevice));

    printf("Calling kernel\n");
    // call kernel
    column_reduce<<<tot_num_blocks, threads_per_block, sizeof(float)*threads_per_block>>>(matrix_gpu, device_result, m, n, num_blocks_per_line);

    printf("Kernel launched. Waiting...\n");
    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel finished. Copying back results.\n");
    // copy back results
    CUDA_CHECK(cudaMemcpy(result_gpu, device_result, m * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free gpu memory
    CUDA_CHECK(cudaFree(matrix_gpu));
    CUDA_CHECK(cudaFree(device_result));
    
    printf("Released GPU memory. Validating results...\n");
    // compare results
    for (int i = 0; i < m; i++) {
        if (result_cpu[i] - result_gpu[i] > 1e-4) 
            printf("INCORRECT RESULT: %.10f %.10f @ %d\n", result_cpu[i], result_gpu[i], i);
    }
    
    free(result_gpu);
    free(result_cpu);
    free(matrix);
    return 0;
}