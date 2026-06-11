/*
 * Experiment 19: cudaHostAlloc / cuMemHostAlloc — pinned (page-locked) host memory
 *
 * CVE-2011-0636: NVIDIA 260.19.21 cudaHostAlloc returned uninitialized kernel
 *   memory, exposing kernel heap data to user space.
 *
 * This test verifies whether the vulnerability is fixed in current drivers
 * (CUDA 12.8 / 13.x) AND whether PyTorch's pinned-memory host cache
 * (used for GPU→CPU transfers) can cause a similar host-side information leak.
 *
 * Tests:
 *   A. Fresh cudaHostAlloc — are new pages zeroed?
 *   B. cudaHostAlloc free + realloc (same size) — does reuse carry old data?
 *   C. PyTorch-style pinned alloc via cuMemHostAlloc — same check
 *   D. Multiple alloc/free cycles — entropy of leaked data
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)
#define CU_CHECK(call) do { CUresult _r=(call); if(_r!=CUDA_SUCCESS){ \
    const char *s="?"; cuGetErrorString(_r,&s); \
    fprintf(stderr,"[CU] %s:%d %s\n",__FILE__,__LINE__,s); exit(1); } } while(0)

static int count_nonzero_bytes(const unsigned char *buf, size_t n) {
    int c=0; for(size_t i=0;i<n;i++) if(buf[i]) c++; return c;
}

static int count_match_float(const float *buf, float val, int n) {
    int c=0; for(int i=0;i<n;i++) if(fabsf(buf[i]-val)<1e-3f) c++; return c;
}

int main() {
    CU_CHECK(cuInit(0));
    CHECK(cudaSetDevice(0));

    printf("=== Experiment 19: Pinned Host Memory Leak Test ===\n");
    printf("    Testing CVE-2011-0636 class vulnerability on modern CUDA drivers\n\n");

    const int N = 1024 * 1024; /* 4 MB */
    const float SECRET = 1.41421f;
    const size_t BYTES = (size_t)N * sizeof(float);

    /* ---- Part A: Fresh cudaHostAlloc — should be zeroed on a fixed driver ---- */
    printf("[Part A] Fresh cudaHostAlloc — are new pinned pages zeroed?\n");
    {
        float *h = NULL;
        CHECK(cudaHostAlloc(&h, BYTES, cudaHostAllocDefault));

        int nonzero = count_nonzero_bytes((unsigned char*)h, BYTES);
        printf("  Non-zero bytes on fresh alloc : %d / %zu\n", nonzero, BYTES);
        if (nonzero == 0)
            printf("  Verdict : SAFE — driver zeroes pinned pages on fresh alloc\n");
        else
            printf("  Verdict : [!!!] LEAK — uninitialized data in fresh cudaHostAlloc\n");

        /* Fill with secret, then free */
        memset(h, 0, BYTES);
        for (int i = 0; i < N; i++) h[i] = SECRET;
        CHECK(cudaFreeHost(h));
    }
    printf("\n");

    /* ---- Part B: cudaHostAlloc free + same-size realloc — CUDA driver pinned pool? ---- */
    printf("[Part B] cudaHostAlloc free + realloc (same size) — does reuse expose old data?\n");
    printf("  (PyTorch uses CachingHostAllocator — this is the raw driver level)\n");
    {
        float *h1 = NULL, *h2 = NULL;
        CHECK(cudaHostAlloc(&h1, BYTES, cudaHostAllocDefault));
        /* Fill h1 with secret */
        for (int i = 0; i < N; i++) h1[i] = SECRET;
        CHECK(cudaFreeHost(h1));

        /* Reallocate same size */
        CHECK(cudaHostAlloc(&h2, BYTES, cudaHostAllocDefault));

        int matches = count_match_float(h2, SECRET, N);
        int nonzero = count_nonzero_bytes((unsigned char*)h2, BYTES);
        printf("  Matches   : %d / %d  (%.1f%%)\n", matches, N, 100.0f*matches/N);
        printf("  Non-zero bytes : %d / %zu\n", nonzero, BYTES);
        if (matches == N)
            printf("  Verdict : [!!!] FULL LEAK — cudaHostAlloc reuse carries old data\n");
        else if (matches > N/10)
            printf("  Verdict : [~] PARTIAL LEAK — significant residual data\n");
        else
            printf("  Verdict : SAFE — pinned pool is zeroed on reuse\n");

        CHECK(cudaFreeHost(h2));
    }
    printf("\n");

    /* ---- Part C: cuMemHostAlloc (driver API) ---- */
    printf("[Part C] cuMemHostAlloc (driver API) free + realloc\n");
    {
        float *h1 = NULL, *h2 = NULL;
        CU_CHECK(cuMemHostAlloc((void**)&h1, BYTES, CU_MEMHOSTALLOC_PORTABLE));
        for (int i = 0; i < N; i++) h1[i] = SECRET * 2.0f;
        CU_CHECK(cuMemFreeHost(h1));

        CU_CHECK(cuMemHostAlloc((void**)&h2, BYTES, CU_MEMHOSTALLOC_PORTABLE));
        int matches = count_match_float(h2, SECRET * 2.0f, N);
        printf("  Matches   : %d / %d  (%.1f%%)\n", matches, N, 100.0f*matches/N);
        if (matches == N)
            printf("  Verdict : [!!!] FULL LEAK — cuMemHostAlloc reuse leaks\n");
        else if (matches > N/10)
            printf("  Verdict : [~] PARTIAL LEAK\n");
        else
            printf("  Verdict : SAFE — driver-level pinned pool is zeroed\n");

        CU_CHECK(cuMemFreeHost(h2));
    }
    printf("\n");

    /* ---- Part D: PyTorch CachingHostAllocator simulation ---- */
    printf("[Part D] Multi-cycle pinned alloc — entropy analysis\n");
    printf("  (10 cycles of fill-secret / free / realloc / read)\n");
    {
        const int CYCLES = 10;
        int total_matches = 0;
        for (int c = 0; c < CYCLES; c++) {
            float *h1 = NULL, *h2 = NULL;
            float sec = (float)(c + 1) * 0.12345f;
            CHECK(cudaHostAlloc(&h1, BYTES, cudaHostAllocDefault));
            for (int i = 0; i < N; i++) h1[i] = sec;
            CHECK(cudaFreeHost(h1));
            CHECK(cudaHostAlloc(&h2, BYTES, cudaHostAllocDefault));
            int m = count_match_float(h2, sec, N);
            total_matches += m;
            if (m > 0) printf("  Cycle %2d: %d/%d matches (secret=%.5f)\n", c, m, N, sec);
            CHECK(cudaFreeHost(h2));
        }
        printf("  Total matches across %d cycles: %d / %d\n",
               CYCLES, total_matches, N*CYCLES);
        if (total_matches == N*CYCLES)
            printf("  Verdict : [!!!] CONSISTENT LEAK — all cycles leak\n");
        else if (total_matches > 0)
            printf("  Verdict : [~] INTERMITTENT LEAK\n");
        else
            printf("  Verdict : SAFE — no pinned memory leakage\n");
    }

    printf("\n[Done]\n");
    return 0;
}
