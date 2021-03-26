#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

#define CUDA_CHECK(status) (assert(status == cudaSuccess))


int main(int argc, char * argv[])  {
    if (argc < 3) {
        printf("Usage: %s <m> <n>\n", argv[0]);
        return 0;
    }

    int m = atoi(argv[1]), n = atoi(argv[2]);
    
    srand(time(NULL)); // seed 

    // create row-major matrix m x n
    float * matrix = (float *) malloc(sizeof(float) * m * n); // m x n
    // create array to store result
    float * result_gpu = (float *) malloc(sizeof(float) * m); // m x 1
    float * result_cpu = (float *) malloc(sizeof(float) * m); // validation

    // populate the array 
    for (int i = 0; i < m * n; i++)
        matrix[i] = 1.0 / (rand() % 977);
    
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

    // allocate gpu memory
    float * matrix_gpu, * device_result;
    CUDA_CHECK(cudaMalloc(&matrix_gpu, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(float) * m));
    
    // move matrix into gpu
    CUDA_CHECK(cudaMemcpy(matrix_gpu, matrix, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // call kernel


    // copy back results
    CUDA_CHECK(cudaMemcpy(result_gpu, device_result, m * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free gpu memory
    CUDA_CHECK(cudaFree(matrix_gpu));
    CUDA_CHECK(cudaFree(device_result));

    // compare results
    for (int i = 0; i < m; i++) {
        if (result_cpu - result_gpu > 1e-6)
            printf("INCORRECT RESULT\n");
            break;
    }
    
    free(result_gpu);
    free(result_cpu);
    free(matrix);
    return 0;
}