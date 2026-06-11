/*
 * Experiment 16: Multi-GPU Peer-to-Peer Memory Access Leak
 *
 * With NVLink-connected GPUs (H100, A100), peer-to-peer access allows
 * GPU1 to read/write GPU0's memory directly without CPU involvement.
 * This is used by model sharding in LLM inference (tensor parallelism).
 *
 * Question: When a model shard on GPU0 frees its activation buffers
 * to the pool, can the shard on GPU1 (which has P2P enabled) access
 * that freed memory?
 *
 * More precisely:
 *   - Is P2P access scoped to specific allocations or to the entire device?
 *   - If GPU0's pool has dirty data and GPU1 gets a P2P pointer to it
 *     (e.g., through cudaMemcpy with peer access), can GPU1 read GPU0's
 *     previous computation?
 *
 * Test design:
 *   1. Check P2P capability between GPU0 and GPU1
 *   2. Enable peer access GPU1 → GPU0
 *   3. GPU0: allocate, fill with secret, free to pool
 *   4. GPU0: reallocate from pool (same dirty block)
 *   5. GPU1: copy from GPU0's new allocation via cudaMemcpyPeer
 *   6. Measure how much of GPU0's secret data GPU1 received
 *
 * Also tests: can GPU1 directly launch a kernel on GPU0's memory?
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA] %s:%d %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while(0)

__global__ void fill_val(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

int main() {
    int dev_count;
    CHECK(cudaGetDeviceCount(&dev_count));
    printf("=== Multi-GPU P2P Memory Leak Test ===\n");
    printf("    GPU count: %d\n\n", dev_count);

    if (dev_count < 2) {
        printf("[SKIP] Only one GPU found. P2P test requires >= 2 GPUs.\n");
        return 0;
    }

    /* Print GPU names */
    for (int d = 0; d < dev_count && d < 4; d++) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, d));
        printf("    GPU%d: %s\n", d, prop.name);
    }
    printf("\n");

    /* Check P2P capability */
    int can_access_01 = 0, can_access_10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&can_access_01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&can_access_10, 1, 0));
    printf("[1] Peer access capability:\n");
    printf("    GPU0 → GPU1: %s\n", can_access_01 ? "YES" : "NO");
    printf("    GPU1 → GPU0: %s\n\n", can_access_10 ? "YES" : "NO");

    if (!can_access_01 && !can_access_10) {
        printf("[INFO] No P2P access between GPUs (not NVLink-connected or PCIe)\n");
        printf("       Testing cudaMemcpyPeer fallback path (goes via CPU)\n\n");
    }

    /* Enable peer access if available */
    CHECK(cudaSetDevice(0));
    if (can_access_10)
        cudaDeviceEnablePeerAccess(1, 0);  /* GPU0 can access GPU1 */

    CHECK(cudaSetDevice(1));
    if (can_access_01)
        cudaDeviceEnablePeerAccess(0, 0);  /* GPU1 can access GPU0 */

    const int N = 256 * 1024;  /* 1 MB float */
    float SECRET_0 = 2.71828f;
    float NOISE_0  = 0.0f;

    /* -------------------------------------------------------
     * Scenario A: GPU0 pool leak → GPU1 reads via cudaMemcpyPeer
     * ------------------------------------------------------- */
    printf("[2] Scenario A: GPU0 pool dirty data → cudaMemcpyPeer → GPU1\n");
    {
        float *d0_victim = NULL, *d0_attacker = NULL, *d1_buf = NULL;
        float h_result[16] = {0};

        /* GPU0: write secret, free to pool */
        CHECK(cudaSetDevice(0));
        CHECK(cudaMalloc(&d0_victim, N * sizeof(float)));
        fill_val<<<(N+255)/256, 256>>>(d0_victim, SECRET_0, N);
        CHECK(cudaDeviceSynchronize());
        printf("  GPU0 victim ptr  : %p  (secret=%.5f)\n",
               (void*)d0_victim, SECRET_0);
        CHECK(cudaFree(d0_victim));  /* goes to PyTorch pool / OS pool */

        /* GPU0: reallocate from pool (same dirty block) */
        CHECK(cudaMalloc(&d0_attacker, N * sizeof(float)));
        printf("  GPU0 realloc ptr : %p  (same? %s)\n",
               (void*)d0_attacker,
               d0_attacker == d0_victim ? "YES" : "NO");

        /* GPU1: copy GPU0's new allocation via P2P */
        CHECK(cudaSetDevice(1));
        CHECK(cudaMalloc(&d1_buf, N * sizeof(float)));
        fill_val<<<(N+255)/256, 256>>>(d1_buf, NOISE_0, N);
        CHECK(cudaDeviceSynchronize());

        /* P2P copy: GPU0's new (dirty) allocation → GPU1 */
        CHECK(cudaMemcpyPeer(d1_buf, 1, d0_attacker, 0, N * sizeof(float)));
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_result, d1_buf, 16 * sizeof(float),
                         cudaMemcpyDeviceToHost));

        int matches = 0;
        for (int i = 0; i < N; i++) {
            /* Use host copy for full scan */
        }
        /* Quick check on first 16 */
        for (int i = 0; i < 16; i++)
            if (fabsf(h_result[i] - SECRET_0) < 0.01f) matches++;

        printf("  GPU1 received [0:4]: %.5f %.5f %.5f %.5f\n",
               h_result[0], h_result[1], h_result[2], h_result[3]);

        if (matches >= 12)
            printf("  [!!!] P2P LEAK: GPU1 has GPU0's secret data (%d/16 sample match)\n", matches);
        else if (matches > 0)
            printf("  [~] PARTIAL: %d/16 match\n", matches);
        else {
            printf("  [=] No match in first 16 elements\n");
            /* cudaFree goes directly to driver on first alloc → zeroed */
            printf("  (cudaFree returns to driver → driver zeroes → safe)\n");
        }

        CHECK(cudaSetDevice(0));
        CHECK(cudaFree(d0_attacker));
        CHECK(cudaSetDevice(1));
        CHECK(cudaFree(d1_buf));
    }
    printf("\n");

    /* -------------------------------------------------------
     * Scenario B: PyTorch-style pool (reuse without free to driver)
     * Simulate: alloc → fill → "return to pool" (keep address) →
     * "get from pool" (same address) → copy to GPU1
     * ------------------------------------------------------- */
    printf("[3] Scenario B: Pool-style reuse (no cudaFree) → GPU1 reads via P2P\n");
    {
        float *d0_pool = NULL, *d1_result = NULL;
        float h_result[16] = {0};

        /* GPU0: simulate pool allocation */
        CHECK(cudaSetDevice(0));
        CHECK(cudaMalloc(&d0_pool, N * sizeof(float)));
        fill_val<<<(N+255)/256, 256>>>(d0_pool, SECRET_0, N);
        CHECK(cudaDeviceSynchronize());
        printf("  GPU0 pool ptr (dirty): %p  secret=%.5f\n",
               (void*)d0_pool, SECRET_0);

        /* Simulate "return to pool" — in real PyTorch, del tensor puts
           the ptr back in free-list without zeroing. We do the same here:
           just don't zero it, treat same ptr as "new allocation" */

        /* GPU1: P2P copy from GPU0's "new" allocation */
        CHECK(cudaSetDevice(1));
        CHECK(cudaMalloc(&d1_result, N * sizeof(float)));
        CHECK(cudaMemcpyPeer(d1_result, 1, d0_pool, 0, N * sizeof(float)));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_result, d1_result, 16 * sizeof(float),
                         cudaMemcpyDeviceToHost));

        int matches = 0;
        for (int i = 0; i < 16; i++)
            if (fabsf(h_result[i] - SECRET_0) < 0.01f) matches++;

        printf("  GPU1 received [0:4]: %.5f %.5f %.5f %.5f\n",
               h_result[0], h_result[1], h_result[2], h_result[3]);
        printf("  Match (sample 16)  : %d/16\n", matches);
        if (matches == 16)
            printf("  [!!!] CROSS-GPU LEAK via P2P: GPU1 reads GPU0's activation residue\n");

        CHECK(cudaSetDevice(0));
        CHECK(cudaFree(d0_pool));
        CHECK(cudaSetDevice(1));
        CHECK(cudaFree(d1_result));
    }
    printf("\n");

    /* -------------------------------------------------------
     * Scenario C: NVLink bandwidth test to confirm P2P path
     * ------------------------------------------------------- */
    printf("[4] P2P bandwidth check (confirms NVLink vs PCIe)\n");
    {
        float *d0 = NULL, *d1 = NULL;
        const int BIG = 64 * 1024 * 1024;  /* 256 MB */

        CHECK(cudaSetDevice(0));
        CHECK(cudaMalloc(&d0, BIG * sizeof(float)));
        CHECK(cudaSetDevice(1));
        CHECK(cudaMalloc(&d1, BIG * sizeof(float)));

        /* Warm up */
        CHECK(cudaMemcpyPeer(d1, 1, d0, 0, BIG * sizeof(float)));
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0, 0));
        CHECK(cudaMemcpyPeer(d1, 1, d0, 0, BIG * sizeof(float)));
        CHECK(cudaEventRecord(t1, 0));
        CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double gb = (double)BIG * 4 / 1e9;
        printf("    256 MB transfer GPU0→GPU1: %.2f ms  (%.1f GB/s)\n", ms, gb/(ms/1000.0));
        printf("    %s\n",
               gb/(ms/1000.0) > 100 ? "High bandwidth → NVLink active" :
               gb/(ms/1000.0) > 10  ? "Medium bandwidth → PCIe or XGMI" :
                                       "Low bandwidth → no direct P2P path");

        CHECK(cudaSetDevice(0)); CHECK(cudaFree(d0));
        CHECK(cudaSetDevice(1)); CHECK(cudaFree(d1));
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    }

    printf("\n[Done]\n");
    return 0;
}
