#include <cuda_runtime.h>
#include <stdio.h>

int N = 128;  // 총 배열 크기
#define WARP_SIZE 32  // 워프 크기

__global__ void warpSyncKernel(float *A, float *B, float *C, int N) {
    __shared__ float sharedA[WARP_SIZE];
    __shared__ float sharedB[WARP_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x % WARP_SIZE;

    if (idx < N) {
        sharedA[warpIdx] = A[idx];
        sharedB[warpIdx] = B[idx];
        __syncthreads(); // 블록 내 스레드 동기화

        C[idx] = sharedA[warpIdx] + sharedB[warpIdx];
    }
}

int main() {
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // 호스트 배열 초기화
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 디바이스 메모리 할당
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 스레드와 블록 설정 (1D 배열이므로 1차원 그리드 설정)
    int threadsPerBlock = WARP_SIZE;
    int blocksPerGrid = (N + WARP_SIZE - 1) / WARP_SIZE;

    // 커널 실행
    warpSyncKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 결과를 호스트로 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {  // 각 요소가 1 + 2 = 3이어야 함
            printf("Error at index %d: %f != 3.0\n", i, h_C[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Kernel execution successful!\n");
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
