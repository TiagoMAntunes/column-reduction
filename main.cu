#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

#define CUDA_CHECK(status) (assert(status == cudaSuccess))
#define threads_per_block 1024

__global__ void column_reduce(float * matrix, float * result, int m /* lines */, int n /* columns*/, int num_blocks_per_line) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    //                                  if last block of the line                   size of last block  else max blockdim
    unsigned int curr_block_limit = (blockIdx.x % num_blocks_per_line == (num_blocks_per_line - 1)) ? n - (num_blocks_per_line - 1) * threads_per_block : blockDim.x;
    unsigned int line_block_idx = blockIdx.x % num_blocks_per_line;
    //                        matrix line                  +        previous blocks in line     + id in block
    unsigned int i = blockIdx.x / num_blocks_per_line  * n + line_block_idx * threads_per_block + threadIdx.x;
    
    if (i < m * n) sdata[tid] = matrix[i]; 
    __syncthreads();

    if (threadIdx.x < curr_block_limit) {
        for (unsigned int s = 1; tid + s < curr_block_limit; s *= 2) {
            if (tid % (2*s) == 0) {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads();
        }

        if (tid == 0) {result[blockIdx.x] = sdata[0];}
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

    
    unsigned long seed = time(NULL);
    srand(seed); // seed 
    printf("Running with seed %ld\n", seed);

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
    //     printf("%0.10f = ", result_cpu[i]);
    //     for (int j = 0; j < n; j++) {
    //         printf("%0.10f ", matrix[j + i * n]);
    //     }
    //     printf("\n");
    // }

    printf("Allocating GPU memory\n");
    // allocate gpu memory
    float * matrix_gpu, * device_result, * helper_result = NULL;

    CUDA_CHECK(cudaMalloc(&matrix_gpu, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(float) * tot_num_blocks));
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(float) * tot_num_blocks));

    // move matrix into gpu
    CUDA_CHECK(cudaMemcpy(matrix_gpu, matrix, m * n * sizeof(float), cudaMemcpyHostToDevice));

    printf("Running with %d threads per block, %d num blocks per line, %d tot number blocks, total of %d elements per line\n", threads_per_block, num_blocks_per_line, tot_num_blocks,n);

    // call kernel
    column_reduce<<<tot_num_blocks, threads_per_block, sizeof(float)*threads_per_block>>>(matrix_gpu, device_result, m, n, num_blocks_per_line);
    
    if (num_blocks_per_line != 1) {
        // need to keep calling the kernel until the new values are calculated
        printf("Allocating memory for extra array\n");
        
        CUDA_CHECK(cudaMalloc(&helper_result, sizeof(float) * tot_num_blocks));
        CUDA_CHECK(cudaMemset(helper_result, 0, sizeof(float) * tot_num_blocks));


        while (num_blocks_per_line != 1) {            
            // swap the values
            float * tmp;
            tmp = device_result;
            device_result = helper_result;
            helper_result = tmp;

            n = num_blocks_per_line;
            num_blocks_per_line = n / threads_per_block + (n % threads_per_block ? 1 : 0);
            tot_num_blocks = num_blocks_per_line * m;

            // wait for last kernel to finish
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("Running with %d threads per block, %d num blocks per line, %d tot number blocks, total of %d elements per line\n", threads_per_block, num_blocks_per_line, tot_num_blocks,n);
            
            // launch kernel again to reduce further
            column_reduce<<<tot_num_blocks, threads_per_block, sizeof(float)*threads_per_block>>>(helper_result, device_result, m, n, num_blocks_per_line);
        }

    }

    // Wait for final kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel finished. Copying back results.\n");
    // copy back results
    CUDA_CHECK(cudaMemcpy(result_gpu, device_result, m * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free gpu memory
    CUDA_CHECK(cudaFree(matrix_gpu));
    CUDA_CHECK(cudaFree(device_result));
    if (helper_result) CUDA_CHECK(cudaFree(helper_result));
    
    printf("Released GPU memory. Validating results...\n");
    // compare results
    for (int i = 0; i < m; i++) {
        if (abs(result_cpu[i] - result_gpu[i]) > 1e-4) 
            printf("INCORRECT RESULT: %.10f %.10f @ %d, diff=%.10f\n", result_cpu[i], result_gpu[i], i, result_cpu[i] - result_gpu[i]);
        // else printf("Correct result! cpu=%.10f, gpu=%.10f, diff=%.10f\n", result_cpu[i], result_gpu[i], result_cpu[i] - result_gpu[i]);
    }
    
    free(result_gpu);
    free(result_cpu);
    free(matrix);
    return 0;
}