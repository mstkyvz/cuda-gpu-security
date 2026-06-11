/*
 * Experiment 20: cudaMallocManaged (Unified Memory) Leak Test
 *
 * CVE-2024-53869: NVIDIA Unified Memory driver for Linux — uninitialized memory
 *   leak, CVSS 5.5, published 2025-01-28.
 *
 * Unified Memory (cudaMallocManaged) uses a special driver that migrates pages
 * between CPU and GPU on demand. The vulnerability class involves the migration
 * engine handing stale physical pages to a new allocation.
 *
 * Tests:
 *   A. Fresh cudaMallocManaged — are pages zeroed on first access?
 *   B. Free + same-size realloc — does the UM driver pool carry stale data?
 *   C. CPU-prefetch path — access before any GPU touch
 *   D. GPU-prefetch path — GPU writes, free, realloc, CPU reads
 *   E. Device-only hint (cudaMemAdviseSetAccessedBy) — changes behavior?
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

__global__ void fill_kernel(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void count_match_kernel(const float *buf, int *cnt, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fabsf(buf[i] - val) < 1e-3f) atomicAdd(cnt, 1);
}

int main() {
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=== Experiment 20: Unified Memory (cudaMallocManaged) Leak Test ===\n");
    printf("    Device  : %s\n", prop.name);
    printf("    Testing CVE-2024-53869 class (UM driver uninitialized pages)\n\n");

    const int N = 512 * 1024;  /* 2 MB */
    const float SECRET = 2.71828f;
    const size_t BYTES = (size_t)N * sizeof(float);

    /* ---- Part A: Fresh cudaMallocManaged — CPU access path ---- */
    printf("[Part A] Fresh cudaMallocManaged — CPU reads before any GPU touch\n");
    {
        float *p = NULL;
        CHECK(cudaMallocManaged(&p, BYTES));

        /* Read immediately (CPU fault path) */
        int nonzero = 0;
        for (int i = 0; i < N; i++) if (p[i] != 0.0f) nonzero++;
        printf("  Non-zero on fresh alloc (CPU first-touch): %d/%d\n", nonzero, N);
        printf("  Verdict: %s\n", nonzero == 0 ?
               "SAFE — UM driver zeroes on first CPU access" :
               "[!!!] LEAK — stale data on fresh UM allocation");
        CHECK(cudaFree(p));
    }
    printf("\n");

    /* ---- Part B: Free + realloc — UM page pool behavior ---- */
    printf("[Part B] cudaMallocManaged free + same-size realloc\n");
    {
        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocManaged(&p1, BYTES));

        /* Write secret from GPU */
        fill_kernel<<<(N+255)/256, 256>>>(p1, SECRET, N);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(p1));

        /* Reallocate */
        CHECK(cudaMallocManaged(&p2, BYTES));

        /* Read from CPU */
        int matches = 0, nonzero = 0;
        for (int i = 0; i < N; i++) {
            if (fabsf(p2[i] - SECRET) < 1e-3f) matches++;
            if (p2[i] != 0.0f) nonzero++;
        }
        printf("  Matches  : %d/%d  (%.1f%%)\n", matches, N, 100.0f*matches/N);
        printf("  Non-zero : %d/%d\n", nonzero, N);
        if (matches == N)
            printf("  Verdict : [!!!] FULL LEAK — UM pool reuse not zeroed\n");
        else if (nonzero > N/10)
            printf("  Verdict : [~] PARTIAL LEAK — residual stale data\n");
        else
            printf("  Verdict : SAFE — UM driver zeroes on reuse\n");
        CHECK(cudaFree(p2));
    }
    printf("\n");

    /* ---- Part C: Prefetch to GPU first, then CPU read after free+realloc ---- */
    printf("[Part C] Victim writes GPU, prefetches to GPU; attacker prefetches to CPU\n");
    {
        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocManaged(&p1, BYTES));
        fill_kernel<<<(N+255)/256, 256>>>(p1, SECRET * 2, N);
        CHECK(cudaDeviceSynchronize());
        /* Keep data on GPU side */
        CHECK(cudaMemPrefetchAsync(p1, BYTES, dev));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(p1));

        CHECK(cudaMallocManaged(&p2, BYTES));
        /* Attacker migrates to CPU */
        CHECK(cudaMemPrefetchAsync(p2, BYTES, cudaCpuDeviceId));
        CHECK(cudaDeviceSynchronize());

        int matches = 0;
        for (int i = 0; i < N; i++)
            if (fabsf(p2[i] - SECRET * 2) < 1e-3f) matches++;
        printf("  Matches  : %d/%d  (%.1f%%)\n", matches, N, 100.0f*matches/N);
        if (matches == N)
            printf("  Verdict : [!!!] FULL LEAK — GPU-resident UM pages leak via prefetch\n");
        else if (matches > N/10)
            printf("  Verdict : [~] PARTIAL LEAK\n");
        else
            printf("  Verdict : SAFE\n");
        CHECK(cudaFree(p2));
    }
    printf("\n");

    /* ---- Part D: GPU reads attacker's UM allocation after free+realloc ---- */
    printf("[Part D] GPU reads — kernel counts matches in reallocated UM buffer\n");
    {
        float *p1 = NULL, *p2 = NULL;
        int *d_cnt = NULL;
        CHECK(cudaMallocManaged(&p1, BYTES));
        fill_kernel<<<(N+255)/256, 256>>>(p1, SECRET * 3, N);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(p1));

        CHECK(cudaMallocManaged(&p2, BYTES));
        CHECK(cudaMalloc(&d_cnt, sizeof(int)));
        CHECK(cudaMemset(d_cnt, 0, sizeof(int)));
        count_match_kernel<<<(N+255)/256, 256>>>(p2, d_cnt, SECRET * 3, N);
        CHECK(cudaDeviceSynchronize());

        int h_cnt = 0;
        CHECK(cudaMemcpy(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  Matches (GPU read)  : %d/%d  (%.1f%%)\n", h_cnt, N, 100.0f*h_cnt/N);
        if (h_cnt == N)
            printf("  Verdict : [!!!] FULL LEAK via GPU read\n");
        else if (h_cnt > N/10)
            printf("  Verdict : [~] PARTIAL LEAK\n");
        else
            printf("  Verdict : SAFE\n");
        CHECK(cudaFree(p2));
        CHECK(cudaFree(d_cnt));
    }
    printf("\n");

    /* ---- Part E: cudaMemAdvise impact ---- */
    printf("[Part E] cudaMemAdviseSetPreferredLocation(device) — does hint affect zeroing?\n");
    {
        float *p1 = NULL, *p2 = NULL;
        CHECK(cudaMallocManaged(&p1, BYTES));
        CHECK(cudaMemAdvise(p1, BYTES, cudaMemAdviseSetPreferredLocation, dev));
        fill_kernel<<<(N+255)/256, 256>>>(p1, SECRET * 4, N);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(p1));

        CHECK(cudaMallocManaged(&p2, BYTES));
        CHECK(cudaMemAdvise(p2, BYTES, cudaMemAdviseSetPreferredLocation, dev));

        /* GPU read */
        int *d_cnt = NULL;
        CHECK(cudaMalloc(&d_cnt, sizeof(int)));
        CHECK(cudaMemset(d_cnt, 0, sizeof(int)));
        count_match_kernel<<<(N+255)/256, 256>>>(p2, d_cnt, SECRET * 4, N);
        CHECK(cudaDeviceSynchronize());
        int h_cnt = 0;
        CHECK(cudaMemcpy(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  Matches (preferred=GPU, GPU read): %d/%d  (%.1f%%)\n",
               h_cnt, N, 100.0f*h_cnt/N);
        if (h_cnt == N)
            printf("  Verdict : [!!!] FULL LEAK — Advise hint does not trigger re-zero\n");
        else
            printf("  Verdict : SAFE — Advise or re-zero triggered\n");

        CHECK(cudaFree(p2));
        CHECK(cudaFree(d_cnt));
    }

    printf("\n[Done]\n");
    return 0;
}
