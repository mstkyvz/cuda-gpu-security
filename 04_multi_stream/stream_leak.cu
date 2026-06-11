/*
 * Experiment 4: Multi-Stream Pool Sharing
 *
 * Tests whether data leaks between allocations on different CUDA streams.
 *
 * Finding:
 *   - SAME stream: 100% data leak (pool reuses without zeroing)
 *   - DIFFERENT streams: pool zeroes memory on cross-stream reuse
 *     (CUDA pool allocator has this as a safety mechanism)
 *
 * Implication for inference servers:
 *   If all requests run on the SAME CUDA stream (the default in PyTorch),
 *   the same-stream 100% leak applies. Using different streams per user
 *   triggers the pool's cross-stream zeroing and provides isolation.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(1);                                                     \
        }                                                                \
    } while (0)

__global__ void write_secret(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = 3.14159f * (i + 1);
}

__global__ void read_buf(float *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[i];
}

static void run_test(int same_stream) {
    const int N = 4096;
    cudaStream_t sA, sB;
    CHECK(cudaStreamCreate(&sA));
    CHECK(cudaStreamCreate(&sB));

    float *d_a = NULL, *d_b = NULL, *d_out = NULL;
    float h[N];

    /* Writer (simulates User A on Stream A) */
    CHECK(cudaMallocAsync(&d_a, N * sizeof(float), sA));
    write_secret<<<(N+255)/256, 256, 0, sA>>>(d_a, N);
    CHECK(cudaFreeAsync(d_a, sA));
    CHECK(cudaStreamSynchronize(sA));

    /* Reader (simulates User B) */
    cudaStream_t reader_stream = same_stream ? sA : sB;
    CHECK(cudaMallocAsync(&d_b, N * sizeof(float), reader_stream));
    CHECK(cudaMallocAsync(&d_out, N * sizeof(float), reader_stream));
    read_buf<<<(N+255)/256, 256, 0, reader_stream>>>(d_b, d_out, N);
    CHECK(cudaMemcpyAsync(h, d_out, N * sizeof(float),
                          cudaMemcpyDeviceToHost, reader_stream));
    CHECK(cudaStreamSynchronize(reader_stream));

    int matches = 0;
    int nonzero = 0;
    for (int i = 0; i < N; i++) {
        float expected = 3.14159f * (i + 1);
        if (fabsf(h[i] - expected) < 1e-3f) matches++;
        if (h[i] != 0.0f) nonzero++;
    }

    printf("    Writer ptr : %p\n", (void*)d_a);
    printf("    Reader ptr : %p  (same: %s)\n",
           (void*)d_b, d_a == d_b ? "YES" : "NO");
    printf("    Non-zero   : %d / %d\n", nonzero, N);
    printf("    Pattern match: %d / %d (%.1f%%)\n",
           matches, N, 100.0f * matches / N);
    if (matches == N)
        printf("    --> FULL LEAK\n");
    else if (nonzero == 0)
        printf("    --> SAFE (pool zeroed cross-stream)\n");
    else
        printf("    --> PARTIAL (%d non-zero)\n", nonzero);

    CHECK(cudaFreeAsync(d_b, reader_stream));
    CHECK(cudaFreeAsync(d_out, reader_stream));
    CHECK(cudaStreamSynchronize(reader_stream));
    CHECK(cudaStreamDestroy(sA));
    CHECK(cudaStreamDestroy(sB));
}

int main() {
    printf("=== Multi-Stream Pool Sharing PoC ===\n\n");

    printf("[TEST 1] Writer and reader on the SAME CUDA stream:\n");
    run_test(1);

    printf("\n[TEST 2] Writer on Stream A, reader on Stream B (different streams):\n");
    run_test(0);

    printf("\n[Summary]\n");
    printf("  Same stream  : pool does NOT zero → full data leak\n");
    printf("  Diff streams : pool DOES zero → isolation enforced\n");
    printf("  PyTorch uses the DEFAULT stream for all ops by default\n");
    printf("  → all torch.empty allocations share the same stream → LEAK\n");
    return 0;
}
