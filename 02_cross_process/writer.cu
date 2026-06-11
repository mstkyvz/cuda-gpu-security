#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

/*
 * Cross-Process GPU Memory Leak — Writer (Process A)
 *
 * Allocates GPU memory, writes a recognizable 32-bit magic pattern,
 * then exits WITHOUT calling cudaFree. The CUDA context is destroyed
 * on process exit, releasing the memory back to the driver.
 *
 * Pattern: word[i] = 0xDEAD0000 + i
 * This value is never zero, eliminating false positives in the reader.
 */

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

__global__ void write_magic(uint32_t *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = 0xDEAD0000u + (uint32_t)i;
}

int main() {
    const int N = 64 * 1024 / 4;

    uint32_t *d_buf;
    CHECK(cudaMalloc(&d_buf, N * sizeof(uint32_t)));
    write_magic<<<(N+255)/256, 256>>>(d_buf, N);
    CHECK(cudaDeviceSynchronize());

    printf("[writer] Wrote 0xDEAD0000+i pattern to GPU ptr: %p\n",
           (void*)d_buf);
    printf("[writer] %d words (%d bytes)\n", N, N * 4);
    printf("[writer] Exiting WITHOUT cudaFree\n");
    return 0;
}
