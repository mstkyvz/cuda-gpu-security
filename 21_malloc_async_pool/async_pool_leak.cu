/*
 * Experiment 21: cudaMallocAsync / stream-ordered pool detailed test
 *
 * cudaMallocAsync uses the CUDA stream-ordered memory allocator — a separate
 * pool from cudaMallocAsync. This differs from torch's CUDACachingAllocator.
 *
 * The CUDA programming guide says:
 *   "Memory returned from cudaMallocAsync is uninitialized."
 *   "If the allocation is obtained from a pool that was created with
 *    cudaMemPoolAttrReuseAllowOpportunistic = 0, it may be zeroed."
 *
 * But PyTorch bypasses cudaMallocAsync entirely and uses its own allocator,
 * so this experiment isolates pure CUDA stream-ordered pool behavior.
 *
 * Tests:
 *   A. Default pool: fresh allocation — is it zeroed?
 *   B. Default pool: fill + free + realloc in same stream (no sync) — leaks?
 *   C. Default pool: fill + free + realloc with stream sync — still leaks?
 *   D. Custom pool with cudaMemPoolReuseAllowOpportunistic = 0 — safer?
 *   E. Cross-stream: victim fills on stream 0, attacker allocs on stream 1
 *   F. cudaMemPoolTrimTo(pool, 0) — does trim cause zeroing?
 *   G. cudaMemPoolSetAttribute threshold=0 impact
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

__global__ void fill_kernel(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void count_match(const float *buf, int *cnt, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fabsf(buf[i] - val) < 1e-3f) atomicAdd(cnt, 1);
}

static int gpu_count(const float *buf, float val, int n, cudaStream_t s) {
    int *d = NULL; int h = 0;
    CHECK(cudaMalloc(&d, sizeof(int)));
    CHECK(cudaMemset(d, 0, sizeof(int)));
    count_match<<<(n+255)/256, 256, 0, s>>>(buf, d, val, n);
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d));
    return h;
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Experiment 21: cudaMallocAsync Stream-Ordered Pool ===\n");
    printf("    Device : %s\n\n", prop.name);

    const int N = 256 * 1024;
    const float S = 1.73205f;
    const size_t BYTES = (size_t)N * sizeof(float);

    cudaStream_t s0, s1;
    CHECK(cudaStreamCreate(&s0));
    CHECK(cudaStreamCreate(&s1));

    /* ---- Part A: Fresh allocation from default pool ---- */
    printf("[Part A] Fresh cudaMallocAsync (default pool) — are pages zeroed?\n");
    {
        float *p = NULL;
        CHECK(cudaMallocAsync(&p, BYTES, s0));
        CHECK(cudaStreamSynchronize(s0));
        int m = gpu_count(p, 0.0f, N, s0);
        /* 0.0f match means zeroed */
        printf("  Zero-filled elements: %d/%d\n", m, N);
        printf("  Verdict: %s\n", m == N ? "SAFE — fresh pool alloc is zeroed" :
               "[!] Not all zeroed on fresh allocation");
        CHECK(cudaFreeAsync(p, s0));
        CHECK(cudaStreamSynchronize(s0));
    }
    printf("\n");

    /* ---- Part B: Same stream, no sync between free and realloc ---- */
    printf("[Part B] Same stream, no sync: fill → freeAsync → mallocAsync → read\n");
    {
        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocAsync(&p1, BYTES, s0));
        fill_kernel<<<(N+255)/256, 256, 0, s0>>>(p1, S, N);
        CHECK(cudaFreeAsync(p1, s0));
        /* No sync! */
        CHECK(cudaMallocAsync(&p2, BYTES, s0));
        int m = gpu_count(p2, S, N, s0);
        printf("  Matches: %d/%d (%.1f%%)\n", m, N, 100.0f*m/N);
        if (m == N) printf("  Verdict: [!!!] FULL LEAK — same stream no-sync path leaks\n");
        else if (m > 0) printf("  Verdict: [~] PARTIAL\n");
        else printf("  Verdict: SAFE — zeroed\n");
        CHECK(cudaFreeAsync(p2, s0));
        CHECK(cudaStreamSynchronize(s0));
    }
    printf("\n");

    /* ---- Part C: Same stream WITH sync between free and realloc ---- */
    printf("[Part C] Same stream WITH sync: fill → freeAsync → sync → mallocAsync → read\n");
    {
        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocAsync(&p1, BYTES, s0));
        fill_kernel<<<(N+255)/256, 256, 0, s0>>>(p1, S*2, N);
        CHECK(cudaFreeAsync(p1, s0));
        CHECK(cudaStreamSynchronize(s0));  /* ← sync here */
        CHECK(cudaMallocAsync(&p2, BYTES, s0));
        int m = gpu_count(p2, S*2, N, s0);
        printf("  Matches: %d/%d (%.1f%%)\n", m, N, 100.0f*m/N);
        if (m == N) printf("  Verdict: [!!!] LEAK even with stream sync\n");
        else if (m > 0) printf("  Verdict: [~] PARTIAL\n");
        else printf("  Verdict: SAFE — sync prevents reuse of stale data\n");
        CHECK(cudaFreeAsync(p2, s0));
        CHECK(cudaStreamSynchronize(s0));
    }
    printf("\n");

    /* ---- Part D: Custom pool with opportunistic reuse disabled ---- */
    printf("[Part D] Custom pool — cudaMemPoolReuseAllowOpportunistic = 0\n");
    {
        cudaMemPoolProps poolProps = {};
        poolProps.allocType = cudaMemAllocationTypePinned;
        poolProps.location.type = cudaMemLocationTypeDevice;
        poolProps.location.id = 0;
        cudaMemPool_t pool;
        CHECK(cudaMemPoolCreate(&pool, &poolProps));

        int disabled = 0;
        CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowOpportunistic, &disabled));

        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocFromPoolAsync(&p1, BYTES, pool, s0));
        fill_kernel<<<(N+255)/256, 256, 0, s0>>>(p1, S*3, N);
        CHECK(cudaFreeAsync(p1, s0));
        CHECK(cudaMallocFromPoolAsync(&p2, BYTES, pool, s0));
        int m = gpu_count(p2, S*3, N, s0);
        printf("  Matches (reuse disabled): %d/%d (%.1f%%)\n", m, N, 100.0f*m/N);
        if (m == N) printf("  Verdict: [!!!] LEAK even with opportunistic reuse OFF\n");
        else printf("  Verdict: SAFE — disabling opportunistic reuse prevents leak\n");

        CHECK(cudaFreeAsync(p2, s0));
        CHECK(cudaStreamSynchronize(s0));
        CHECK(cudaMemPoolDestroy(pool));
    }
    printf("\n");

    /* ---- Part E: Cross-stream — victim on s0, attacker on s1 ---- */
    printf("[Part E] Cross-stream: victim on stream-0, attacker alloc on stream-1\n");
    {
        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocAsync(&p1, BYTES, s0));
        fill_kernel<<<(N+255)/256, 256, 0, s0>>>(p1, S*4, N);
        CHECK(cudaFreeAsync(p1, s0));
        CHECK(cudaStreamSynchronize(s0));
        /* Attacker allocates on different stream */
        CHECK(cudaMallocAsync(&p2, BYTES, s1));
        int m = gpu_count(p2, S*4, N, s1);
        printf("  Matches: %d/%d (%.1f%%)\n", m, N, 100.0f*m/N);
        if (m == N) printf("  Verdict: [!!!] CROSS-STREAM LEAK\n");
        else if (m > 0) printf("  Verdict: [~] PARTIAL cross-stream\n");
        else printf("  Verdict: SAFE — cross-stream alloc gets zeroed page\n");
        CHECK(cudaFreeAsync(p2, s1));
        CHECK(cudaStreamSynchronize(s1));
    }
    printf("\n");

    /* ---- Part F: cudaMemPoolTrimTo(pool, 0) before realloc ---- */
    printf("[Part F] cudaMemPoolTrimTo(default_pool, 0) then realloc — does trim zero?\n");
    {
        cudaMemPool_t def_pool;
        CHECK(cudaDeviceGetDefaultMemPool(&def_pool, 0));

        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocAsync(&p1, BYTES, s0));
        fill_kernel<<<(N+255)/256, 256, 0, s0>>>(p1, S*5, N);
        CHECK(cudaFreeAsync(p1, s0));
        CHECK(cudaStreamSynchronize(s0));
        CHECK(cudaMemPoolTrimTo(def_pool, 0));  /* Release all free memory back to OS */
        CHECK(cudaMallocAsync(&p2, BYTES, s0));
        int m = gpu_count(p2, S*5, N, s0);
        printf("  Matches after TrimTo(0): %d/%d (%.1f%%)\n", m, N, 100.0f*m/N);
        if (m == N) printf("  Verdict: [!!!] LEAK even after pool trim\n");
        else printf("  Verdict: SAFE — TrimTo returns memory to OS, fresh pages zeroed\n");
        CHECK(cudaFreeAsync(p2, s0));
        CHECK(cudaStreamSynchronize(s0));
    }
    printf("\n");

    /* ---- Part G: ReleaseThreshold = 0 continuously ---- */
    printf("[Part G] ReleaseThreshold=0 on default pool — perpetual trim, then realloc\n");
    {
        cudaMemPool_t def_pool;
        CHECK(cudaDeviceGetDefaultMemPool(&def_pool, 0));
        uint64_t threshold = 0;
        CHECK(cudaMemPoolSetAttribute(def_pool, cudaMemPoolAttrReleaseThreshold, &threshold));

        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocAsync(&p1, BYTES, s0));
        fill_kernel<<<(N+255)/256, 256, 0, s0>>>(p1, S*6, N);
        CHECK(cudaFreeAsync(p1, s0));
        CHECK(cudaStreamSynchronize(s0));
        /* With threshold=0, free memory is released to OS after every sync */
        CHECK(cudaMallocAsync(&p2, BYTES, s0));
        int m = gpu_count(p2, S*6, N, s0);
        printf("  Matches (threshold=0): %d/%d (%.1f%%)\n", m, N, 100.0f*m/N);
        if (m == N) printf("  Verdict: [!!!] LEAK even with threshold=0\n");
        else printf("  Verdict: SAFE — ReleaseThreshold=0 prevents page reuse\n");
        CHECK(cudaFreeAsync(p2, s0));
        CHECK(cudaStreamSynchronize(s0));
    }

    CHECK(cudaStreamDestroy(s0));
    CHECK(cudaStreamDestroy(s1));
    printf("\n[Done]\n");
    return 0;
}
