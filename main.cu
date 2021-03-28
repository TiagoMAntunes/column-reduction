#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

#define CUDA_CHECK(status) (assert(status == cudaSuccess))
#define threads_per_block 1024

__global__ void column_reduce(float * matrix, float * result, int m /* lines */, int n /* columns*/) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * n + threadIdx.x; // get to idx th line
    unsigned int offset = 0;

    // sum all the values from that line to fit in one single block
    sdata[tid] = 0;
    while (tid + offset < n) {
        sdata[tid] += matrix[i + offset];
        offset += blockDim.x;
    }
    __syncthreads();

    for (unsigned int s = 1; tid + s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {result[blockIdx.x] = sdata[0];}

}

int main(int argc, char * argv[])  {
    if (argc < 3) {
        printf("Usage: %s <m> <n>\n", argv[0]);
        return 0;
    }

    int m = atoi(argv[1]), n = atoi(argv[2]);
    
    unsigned long seed = time(NULL);
    srand(seed); // seed 
    printf("Running with seed %ld\n", seed);

    // create row-major matrix m x n
    float * matrix = (float *) malloc(sizeof(float) * m * n); // m x n
    // create array to store result
    float * result_gpu = (float *) malloc(sizeof(float) * m); // tot_num_blocks x 1
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
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(float) * m));
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(float) * m));

    // move matrix into gpu
    CUDA_CHECK(cudaMemcpy(matrix_gpu, matrix, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // call kernel
    column_reduce<<<m, threads_per_block, sizeof(float)*threads_per_block>>>(matrix_gpu, device_result, m, n);

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