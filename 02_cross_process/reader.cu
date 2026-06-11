#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/*
 * Cross-Process GPU Memory Leak — Reader (Process B)
 *
 * Reads GPU memory immediately after the writer process exits.
 * Uses a non-wrapping 32-bit magic pattern to avoid false positives
 * from zeroed memory accidentally matching a wrap-around byte pattern.
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

__global__ void read_buf(uint32_t *buf, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = buf[i];
}

int main() {
    /* N 32-bit words = N*4 bytes */
    const int N = 64 * 1024 / 4;
    uint32_t *d_buf, *d_out;
    uint32_t  h_result[N];

    printf("[reader] Allocating %d bytes without initialization...\n", N * 4);
    CHECK(cudaMalloc(&d_buf, N * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_out, N * sizeof(uint32_t)));
    printf("[reader] Got GPU ptr: %p\n\n", (void*)d_buf);

    read_buf<<<(N+255)/256, 256>>>(d_buf, d_out, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_result, d_out, N * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));

    /* Magic value written by writer: 0xDEAD0000 + index
     * This never equals 0 so there are no false positives from zeroed memory */
    int matches   = 0;
    int non_zero  = 0;
    for (int i = 0; i < N; i++) {
        uint32_t expected = 0xDEAD0000u + (uint32_t)i;
        if (h_result[i] == expected) matches++;
        if (h_result[i] != 0)        non_zero++;
    }

    double match_pct    = 100.0 * matches  / N;
    double nonzero_pct  = 100.0 * non_zero / N;

    printf("[reader] Results:\n");
    printf("  Total 32-bit words:      %d\n", N);
    printf("  Non-zero words:          %d / %d (%.2f%%)\n",
           non_zero, N, nonzero_pct);
    printf("  Match writer pattern:    %d / %d (%.4f%%)\n",
           matches, N, match_pct);

    if (matches == N) {
        printf("\n[!!!] CROSS-PROCESS LEAK CONFIRMED\n");
        printf("  Full secret recovered from previous process.\n");
    } else if (matches > 10) {
        printf("\n[!] PARTIAL CROSS-PROCESS LEAK — %d words match\n", matches);
    } else if (non_zero == 0) {
        printf("\n[SAFE] Driver zeroed GPU memory between processes.\n");
        printf("  This is the expected secure behavior on modern NVIDIA drivers.\n");
    } else {
        printf("\n[INFO] Memory is not zeroed but secret pattern not found.\n");
        printf("  Non-zero bytes present (%.2f%%) — likely driver internal state.\n",
               nonzero_pct);
        printf("  First 8 words: ");
        for (int i = 0; i < 8 && i < N; i++) printf("0x%08x ", h_result[i]);
        printf("\n");
    }

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_out));
    return 0;
}
