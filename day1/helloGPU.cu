#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    if(threadIdx.x == 0) {
        printf("Hello from the GPU!\n");
    }
}

int main() {
    helloFromGPU<<<1, 1>>>();

    cudaDeviceSynchronize();

    printf("Hello From CPU\n");

    return 0;
}