/*
 * Experiment 32: NVLink P2P Residue
 *
 * GPU 0 and GPU 1 are connected via NVLink 4.0 (NV18 — 18 NVLink lanes,
 * 900 GB/s bidirectional). cudaMemcpyPeerAsync, cudaDeviceEnablePeerAccess,
 * and P2P mapped pointers all use this fabric.
 *
 * Questions:
 *   A. ZEROING: does a P2P peer memcpy zero the destination before transferring,
 *      or does residue from the prior owner remain?
 *      (GPU 0 writes SECRET to d_g0; GPU 1 copies from d_g0 to d_g1 via P2P)
 *
 *   B. POOL RESIDUE via P2P: victim on GPU 0 writes SECRET to pool block,
 *      frees it; attacker on GPU 1 reads the SAME physical pages via P2P
 *      mapped pointer (cudaDeviceEnablePeerAccess) without triggering a new alloc.
 *      Does the remote read bypass any zeroing that would happen on alloc?
 *
 *   C. CROSS-GPU POOL: GPU 0 pool alloc, fill SECRET, free; GPU 1 pool alloc
 *      of same size (different GPU pool) — does cross-GPU pool deliver residue?
 *      (Tests whether GPU 0 pool is accessible to GPU 1 after free)
 *
 *   D. P2P ACCESS ISOLATION: GPU 1 kernel directly reads GPU 0's device memory
 *      via P2P-enabled pointer. GPU 0 fills SECRET, device sync, GPU 1 reads.
 *      Tests: (1) does P2P access see correct data? (2) does GPU 0 cudaFree
 *      zero the buffer before GPU 1 reads?
 *
 * Compile:
 *   nvcc -O2 -arch=sm_90 -o nvlink_p2p_residue nvlink_p2p_residue.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

#define SECRET 3.14159f
#define N      (256 * 1024)   /* 1 MB = 256K floats */

static int count_near(const float *h, float v, float tol, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (fabsf(h[i] - v) < tol) c++;
    return c;
}

__global__ void fill_buf(float *p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void zero_buf(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

/* Read GPU 0 memory directly via P2P pointer from GPU 1 context */
__global__ void p2p_read(const float *remote, float *local, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) local[i] = remote[i];
}

/* Write SECRET to GPU 0 memory directly via P2P pointer from GPU 1 context */
__global__ void p2p_write(float *remote, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) remote[i] = v;
}

int main() {
    /* Check device count */
    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count));
    printf("=== Experiment 32: NVLink P2P Residue ===\n");
    printf("    Device count: %d\n", device_count);
    if (device_count < 2) {
        printf("    [SKIP] Need at least 2 GPUs. Only %d found.\n", device_count);
        return 0;
    }

    cudaDeviceProp prop0, prop1;
    CHECK(cudaGetDeviceProperties(&prop0, 0));
    CHECK(cudaGetDeviceProperties(&prop1, 1));
    printf("    GPU 0: %s\n", prop0.name);
    printf("    GPU 1: %s\n", prop1.name);

    /* Check P2P capability */
    int can_p2p_01 = 0, can_p2p_10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&can_p2p_01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&can_p2p_10, 1, 0));
    printf("    P2P: GPU0->GPU1: %s  GPU1->GPU0: %s\n",
           can_p2p_01 ? "YES" : "NO", can_p2p_10 ? "YES" : "NO");

    if (!can_p2p_01 || !can_p2p_10) {
        printf("    [WARN] P2P not fully available. Proceeding with available directions.\n");
    }

    /* Enable P2P access in both directions */
    CHECK(cudaSetDevice(0));
    if (can_p2p_10) CHECK(cudaDeviceEnablePeerAccess(1, 0));

    CHECK(cudaSetDevice(1));
    if (can_p2p_01) CHECK(cudaDeviceEnablePeerAccess(0, 0));

    printf("\n");

    float *h_buf = (float *)malloc(N * sizeof(float));

    for (int pass = 1; pass <= 5; pass++) {
        printf("========== PASS %d / 5 ==========\n", pass);

        /* ---- Test A: P2P memcpy zeroing ---- */
        printf("[Test A] P2P cudaMemcpyPeer: does destination get zeroed before copy?\n");
        {
            /* GPU 0: alloc + fill with SECRET */
            CHECK(cudaSetDevice(0));
            float *d_g0;
            CHECK(cudaMalloc(&d_g0, N * sizeof(float)));
            fill_buf<<<(N+255)/256, 256>>>(d_g0, SECRET, N);
            CHECK(cudaDeviceSynchronize());

            /* GPU 1: alloc + fill with DECOY (0x42 = 66.0f), then overwrite via P2P copy */
            CHECK(cudaSetDevice(1));
            float *d_g1;
            CHECK(cudaMalloc(&d_g1, N * sizeof(float)));
            fill_buf<<<(N+255)/256, 256>>>(d_g1, 66.0f, N);
            CHECK(cudaDeviceSynchronize());

            /* P2P copy: GPU0->GPU1 via NVLink */
            CHECK(cudaMemcpyPeer(d_g1, 1, d_g0, 0, N * sizeof(float)));
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(h_buf, d_g1, N * sizeof(float), cudaMemcpyDeviceToHost));
            int hits_secret = count_near(h_buf, SECRET, 0.01f, N);
            int hits_decoy  = count_near(h_buf, 66.0f, 0.01f, N);
            printf("  GPU0 writes SECRET, GPU1 writes DECOY=66.0, then P2P copy GPU0->GPU1\n");
            printf("  GPU1 d_g1 after P2P copy: SECRET=%d/%d  DECOY=%d/%d\n",
                   hits_secret, N, hits_decoy, N);
            if (hits_secret == N)
                printf("  [OK] P2P copy transfers data correctly (full SECRET visible on GPU1)\n");
            else
                printf("  [?] Unexpected: only %d/%d match\n", hits_secret, N);

            CHECK(cudaSetDevice(0)); CHECK(cudaFree(d_g0));
            CHECK(cudaSetDevice(1)); CHECK(cudaFree(d_g1));
        }

        /* ---- Test B: P2P direct pointer read (no copy) ---- */
        printf("[Test B] P2P direct pointer — GPU1 kernel reads GPU0 memory via peer pointer\n");
        {
            /* GPU 0: alloc + fill SECRET */
            CHECK(cudaSetDevice(0));
            float *d_g0;
            CHECK(cudaMalloc(&d_g0, N * sizeof(float)));
            fill_buf<<<(N+255)/256, 256>>>(d_g0, SECRET, N);
            CHECK(cudaDeviceSynchronize());

            /* GPU 1: alloc local buf, launch kernel that reads d_g0 via P2P */
            CHECK(cudaSetDevice(1));
            float *d_local;
            CHECK(cudaMalloc(&d_local, N * sizeof(float)));
            CHECK(cudaMemset(d_local, 0, N * sizeof(float)));

            /* d_g0 is GPU0 pointer — valid as P2P pointer when peer access is enabled */
            p2p_read<<<(N+255)/256, 256>>>(d_g0, d_local, N);
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(h_buf, d_local, N * sizeof(float), cudaMemcpyDeviceToHost));
            int hits = count_near(h_buf, SECRET, 0.01f, N);
            printf("  GPU0 d_g0=SECRET, GPU1 p2p_read(d_g0) -> d_local: hits=%d/%d\n", hits, N);
            if (hits == N)
                printf("  [OK] GPU1 can read GPU0 memory directly via P2P pointer\n");
            else
                printf("  [!] Only %d/%d match — partial or no P2P access\n", hits, N);

            CHECK(cudaSetDevice(0)); CHECK(cudaFree(d_g0));
            CHECK(cudaSetDevice(1)); CHECK(cudaFree(d_local));
        }

        /* ---- Test C: P2P read AFTER GPU0 cudaFree ---- */
        printf("[Test C] P2P residue after cudaFree — GPU0 frees; GPU1 tries P2P read\n");
        {
            /* GPU 0: alloc + fill SECRET, then FREE */
            CHECK(cudaSetDevice(0));
            float *d_g0;
            CHECK(cudaMalloc(&d_g0, N * sizeof(float)));
            fill_buf<<<(N+255)/256, 256>>>(d_g0, SECRET, N);
            CHECK(cudaDeviceSynchronize());

            /* Record the pointer value before free */
            uintptr_t ptr_val = (uintptr_t)d_g0;

            /* GPU 0: FREE (this returns memory to OS/driver — should unmap from GPU 1) */
            CHECK(cudaFree(d_g0));
            CHECK(cudaDeviceSynchronize());

            /* GPU 1: immediately re-alloc from GPU 0's pool via P2P? No — we can't
             * control GPU0's allocator from GPU1. Instead, we test a different angle:
             * GPU1 allocates its own buffer, then victim re-allocs on GPU0 to get
             * the same physical pages, and GPU1 pool-allocs same size — does it
             * get GPU0's recycled physical pages? Answer: No, pools are per-device.
             * This test documents that cross-device pool reuse doesn't happen. */
            CHECK(cudaSetDevice(1));
            float *d_g1;
            CHECK(cudaMallocAsync(&d_g1, N * sizeof(float), 0));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_buf, d_g1, N * sizeof(float), cudaMemcpyDeviceToHost));
            int hits = count_near(h_buf, SECRET, 0.01f, N);
            printf("  GPU0 alloc+fill+free; GPU1 pool alloc same size: SECRET=%d/%d\n", hits, N);
            if (hits > N / 4)
                printf("  [!!!] CROSS-DEVICE POOL RESIDUE — GPU0 SECRET reached GPU1 pool\n");
            else
                printf("  [SAFE] Cross-device pool alloc does not deliver GPU0 residue\n");
            printf("  (GPU0 ptr was 0x%lx; cross-GPU pool reuse physically impossible)\n",
                   (unsigned long)ptr_val);

            CHECK(cudaFreeAsync(d_g1, 0));
            CHECK(cudaDeviceSynchronize());
        }

        /* ---- Test D: P2P write from GPU1 -> GPU0, GPU0 reads ---- */
        printf("[Test D] P2P write — GPU1 kernel writes SECRET to GPU0 memory\n");
        {
            /* GPU 0: alloc + zero */
            CHECK(cudaSetDevice(0));
            float *d_g0;
            CHECK(cudaMalloc(&d_g0, N * sizeof(float)));
            zero_buf<<<(N+255)/256, 256>>>(d_g0, N);
            CHECK(cudaDeviceSynchronize());

            /* GPU 1: write SECRET into d_g0 via P2P pointer */
            CHECK(cudaSetDevice(1));
            p2p_write<<<(N+255)/256, 256>>>(d_g0, SECRET, N);
            CHECK(cudaDeviceSynchronize());

            /* GPU 0: read back d_g0 */
            CHECK(cudaSetDevice(0));
            CHECK(cudaMemcpy(h_buf, d_g0, N * sizeof(float), cudaMemcpyDeviceToHost));
            int hits = count_near(h_buf, SECRET, 0.01f, N);
            printf("  GPU0 zeroed d_g0, GPU1 p2p_write(d_g0, SECRET), GPU0 reads: hits=%d/%d\n",
                   hits, N);
            if (hits == N)
                printf("  [OK] GPU1 P2P write immediately visible on GPU0 (NVLink coherent)\n");
            else
                printf("  [!] %d/%d match — partial P2P write visibility\n", hits, N);

            CHECK(cudaSetDevice(0)); CHECK(cudaFree(d_g0));
        }

        /* ---- Test E: pool allocations NOT accessible via P2P; pool residue via staging ---- */
        printf("[Test E] P2P + pool residue: GPU0 pool residue staged to cudaMalloc, GPU1 reads\n");
        {
            /* GPU 0: pool alloc -> fill SECRET -> free (NO sync) -> re-alloc (residue present) */
            CHECK(cudaSetDevice(0));
            cudaStream_t s0;
            CHECK(cudaStreamCreate(&s0));

            float *d_pre;
            CHECK(cudaMallocAsync(&d_pre, N * sizeof(float), s0));
            fill_buf<<<(N+255)/256, 256, 0, s0>>>(d_pre, SECRET, N);
            CHECK(cudaStreamSynchronize(s0));
            CHECK(cudaFreeAsync(d_pre, s0));
            /* NO sync — same-stream pool block has SECRET */

            float *d_pool;
            CHECK(cudaMallocAsync(&d_pool, N * sizeof(float), s0));
            CHECK(cudaStreamSynchronize(s0));

            /* Verify pool residue on GPU0 */
            CHECK(cudaMemcpy(h_buf, d_pool, N * sizeof(float), cudaMemcpyDeviceToHost));
            int g0_hits = count_near(h_buf, SECRET, 0.01f, N);
            printf("  GPU0 pool residue (same-stream no-sync): SECRET=%d/%d\n", g0_hits, N);

            /* Stage pool block content into a cudaMalloc buffer (P2P requires cudaMalloc) */
            /* Note: cudaMallocAsync pool ptrs are NOT P2P-accessible — only cudaMalloc is. */
            /* We copy residue to a staging cudaMalloc buffer so GPU1 can P2P-read it. */
            float *d_staging;
            CHECK(cudaMalloc(&d_staging, N * sizeof(float)));
            /* D2D copy on GPU0: no zeroing, copies raw bytes including pool residue */
            CHECK(cudaMemcpy(d_staging, d_pool, N * sizeof(float), cudaMemcpyDeviceToDevice));

            /* GPU 1: P2P read d_staging (cudaMalloc — P2P-accessible) */
            CHECK(cudaSetDevice(1));
            float *d_local;
            CHECK(cudaMalloc(&d_local, N * sizeof(float)));
            CHECK(cudaMemset(d_local, 0, N * sizeof(float)));

            p2p_read<<<(N+255)/256, 256>>>(d_staging, d_local, N);
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(h_buf, d_local, N * sizeof(float), cudaMemcpyDeviceToHost));
            int p2p_hits = count_near(h_buf, SECRET, 0.01f, N);
            printf("  GPU1 P2P read of GPU0 residue (via staging cudaMalloc): SECRET=%d/%d\n",
                   p2p_hits, N);
            printf("  Note: cudaMallocAsync pool ptrs are NOT directly P2P-accessible;\n");
            printf("        only cudaMalloc ptrs support P2P. Pool residue requires staging.\n");
            if (p2p_hits > N / 4)
                printf("  [!!!] GPU0 pool residue propagated to GPU1 via P2P+staging\n");
            else
                printf("  [SAFE] GPU0 pool residue did not propagate via P2P+staging\n");

            CHECK(cudaSetDevice(0));
            CHECK(cudaFreeAsync(d_pool, s0));
            CHECK(cudaStreamSynchronize(s0));
            CHECK(cudaStreamDestroy(s0));
            CHECK(cudaFree(d_staging));
            CHECK(cudaSetDevice(1));
            CHECK(cudaFree(d_local));
        }
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("A: P2P cudaMemcpyPeer — correct data transfer, no zeroing artifacts\n");
    printf("B: P2P direct pointer — GPU1 kernel reads GPU0 memory via P2P\n");
    printf("C: Cross-device pool residue — GPU0 free, GPU1 pool alloc (should be SAFE)\n");
    printf("D: P2P write coherence — GPU1 writes visible on GPU0 via NVLink\n");
    printf("E: GPU0 pool residue + P2P read from GPU1 — does NVLink expose GPU0 pool residue?\n");
    printf("\n[Done]\n");

    free(h_buf);

    /* Disable P2P */
    CHECK(cudaSetDevice(0));
    if (can_p2p_10) cudaDeviceDisablePeerAccess(1);
    CHECK(cudaSetDevice(1));
    if (can_p2p_01) cudaDeviceDisablePeerAccess(0);

    return 0;
}
