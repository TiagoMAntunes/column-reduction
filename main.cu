#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <chrono>

#define CUDA_CHECK(status) (assert(status == cudaSuccess))
#define threads_per_block 1024

__global__ void column_reduce(float * matrix, float * result, int m /* lines */, int n /* columns*/) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x; // line
    unsigned int i = threadIdx.x * n + threadIdx.y + blockIdx.y * blockDim.y; // get to idx th line
    unsigned int offset = 0;
    unsigned int it = n * blockDim.x; // advance blockDim.x threads vertically
    unsigned int real_y = blockIdx.y * blockDim.y + threadIdx.y;

    // sum all the values from that column to fit in one single block
    sdata[tid] = 0;
    if (real_y < n && threadIdx.x < m) // remember we only have one x block
        while (i + offset < n*m) {
            sdata[tid] += matrix[i + offset];
            offset += it; 
            
        }
    __syncthreads();

    unsigned int lowest = blockDim.x > m ? m : blockDim.x;
    if (real_y < n && threadIdx.x < m)
        for (unsigned int s = 1; threadIdx.x + s < lowest; s *= 2) {
            if (threadIdx.x % (2*s) == 0) {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads();
        }

    if (threadIdx.x == 0 && real_y < n) {
        result[real_y] = sdata[tid];
    }

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
    float * result_gpu = (float *) malloc(sizeof(float) * n); // tot_num_blocks x 1
    float * result_cpu = (float *) malloc(sizeof(float) * n); // validation
    memset(result_cpu, 0, sizeof(float) * n);

    printf("Populating array \n");
    // populate the array 
    for (int i = 0; i < m * n; i++)
        matrix[i] = 1.0 / ((rand() % 977) + 1);
    

    printf("Calculating final result\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    // calculate cpu result
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) 
            result_cpu[j] += matrix[i * n + j];
    auto cpu_end = std::chrono::high_resolution_clock::now();

    printf("CPU took %f ms.\n", std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0);
    

    printf("Allocating GPU memory, m=%d, n=%d\n", m, n);
    // allocate gpu memory
    float * matrix_gpu, * device_result, * helper_result = NULL;
    
    CUDA_CHECK(cudaMalloc(&matrix_gpu, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(float) * n));
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(float) * n));
    printf("Finished allocating. Copying matrix...\n");
    // move matrix into gpu
    CUDA_CHECK(cudaMemcpy(matrix_gpu, matrix, m * n * sizeof(float), cudaMemcpyHostToDevice));

    printf("Calling kernel with m=%d n=%d\n", m, n);
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // call kernel
    dim3 block_threads(32, 32);
    dim3 grid_threads(1, n / 32 + (n % 32 ? 1 : 0));

    CUDA_CHECK(cudaEventRecord(start));
    column_reduce<<<grid_threads, block_threads, sizeof(float)*threads_per_block>>>(matrix_gpu, device_result, m, n);
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for final kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    // end = std::chrono::high_resolution_clock::now();
    
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Kernel finished. Took %f ms. Copying back results.\n", elapsed_time);
    // copy back results
    CUDA_CHECK(cudaMemcpy(result_gpu, device_result, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free gpu memory
    CUDA_CHECK(cudaFree(matrix_gpu));
    CUDA_CHECK(cudaFree(device_result));
    if (helper_result) CUDA_CHECK(cudaFree(helper_result));
    
    printf("Released GPU memory. Validating results...\n");
    // compare results
    for (int i = 0; i < n; i++) {
        if (abs(result_cpu[i] - result_gpu[i]) > 1e-3) 
            printf("INCORRECT RESULT: %.10f %.10f @ %d, diff=%.10f\n", result_cpu[i], result_gpu[i], i, result_cpu[i] - result_gpu[i]);
        // else printf("Correct result! cpu=%.10f, gpu=%.10f, diff=%.10f\n", result_cpu[i], result_gpu[i], result_cpu[i] - result_gpu[i]);
    }
    
    free(result_gpu);
    free(result_cpu);
    free(matrix);
    return 0;
}