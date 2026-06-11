/*
 * Experiment 28: Tensor Core (wmma) Fragment & Shared Memory Residue
 *
 * WMMA (Warp Matrix Multiply Accumulate) API uses Tensor Cores directly.
 * Key questions:
 *   A. Do wmma accumulator fragment REGISTERS get zeroed between kernel launches?
 *   B. Does __shared__ staging memory used by wmma operations persist?
 *   C. Does forced local-memory register spill carry SECRET across kernels?
 *
 * H100 Tensor Core: 4th-gen, supports m16n16k16 fp16/bf16/fp32, wgmma.
 * Fragment layout: accumulator<16,16,16,float> = 256 fp32 total, 8 fp32 per thread.
 *
 * Compile:
 *   nvcc -O2 -arch=sm_90 -o tensor_core_residue tensor_core_residue.cu
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

using namespace nvcuda;

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

/* wmma tile dimensions (m16n16k16) */
#define WM 16
#define WN 16
#define WK 16

/* Accumulator: 256 fp32 per warp, 8 fp32 per thread (32 threads per warp) */
#define ACCUM_PER_THREAD 8
#define WARP_SIZE        32
#define ACCUM_PER_WARP   (ACCUM_PER_THREAD * WARP_SIZE)  /* 256 */

/* Number of warps: saturate all 132 SMs */
#define N_WARPS  512

/* Secret values */
#define SECRET_H __float2half(3.14159f)
#define SECRET_F 3.14159f
/* After mma(SECRET × 1 + 0): each accumulator element = 3.14159 × 16 = 50.2654 */
#define SECRET_MMA (3.14159f * 16.0f)

/* Shared memory matrix size */
#define SHM_ELEMS (WM * WK)   /* 256 halfs = 512 bytes */

/* Local array size — large enough to force register spill to local mem (DRAM) */
#define LOCAL_N 512

/* ================================================================
 * Test A kernels: wmma accumulator fragment register residue
 * ================================================================ */

/* Victim: compute SECRET values into accumulator fragment, then exit WITHOUT storing */
__global__ void __noinline__ victim_fragment(int *sink) {
    wmma::fragment<wmma::matrix_a,    WM, WN, WK, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,    WM, WN, WK, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c;

    wmma::fill_fragment(a, SECRET_H);
    wmma::fill_fragment(b, __float2half(1.0f));
    wmma::fill_fragment(c, 0.0f);
    wmma::mma_sync(c, a, b, c);

    /* c now holds SECRET_MMA = 50.265... in all 256 elements (across warp) */
    /* Exit WITHOUT storing c to global memory — registers hold SECRET_MMA */
    if ((int)c.x[0] == 0xDEAD) sink[blockIdx.x] = (int)c.x[0];
}

/* Attacker: declare UNINITIALIZED fragment, store immediately */
__global__ void __noinline__ attacker_fragment(float *d_out) {
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c;
    /* No fill_fragment, no mma_sync — register state is "unspecified" */
    wmma::store_matrix_sync(d_out + (long long)blockIdx.x * ACCUM_PER_WARP,
                            c, WN, wmma::mem_row_major);
}

/* Sanity: store victim's fragment to verify SECRET_MMA is correct */
__global__ void __noinline__ victim_fragment_store(float *d_out) {
    wmma::fragment<wmma::matrix_a,    WM, WN, WK, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,    WM, WN, WK, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c;
    wmma::fill_fragment(a, SECRET_H);
    wmma::fill_fragment(b, __float2half(1.0f));
    wmma::fill_fragment(c, 0.0f);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(d_out + (long long)blockIdx.x * ACCUM_PER_WARP,
                            c, WN, wmma::mem_row_major);
}

/* ================================================================
 * Test B kernels: __shared__ residue from wmma staging memory
 * ================================================================ */

/* Victim: fill __shared__ with SECRET, load into wmma fragment, exit (shm not cleared) */
__global__ void __noinline__ victim_wmma_shared(int *sink) {
    __shared__ half sh_a[SHM_ELEMS];
    __shared__ half sh_b[SHM_ELEMS];

    int tid = threadIdx.x;
    for (int i = tid; i < SHM_ELEMS; i += WARP_SIZE) {
        sh_a[i] = SECRET_H;
        sh_b[i] = __float2half(1.0f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a,    WM, WN, WK, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,    WM, WN, WK, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c;
    wmma::load_matrix_sync(a, sh_a, WK);
    wmma::load_matrix_sync(b, sh_b, WK);
    wmma::fill_fragment(c, 0.0f);
    wmma::mma_sync(c, a, b, c);

    /* sh_a still holds SECRET_H, sh_b still holds 1.0h — kernel exits */
    if (tid == 0 && (int)__half2float(sh_a[0]) == 0xDEAD) sink[blockIdx.x] = 1;
}

/* Attacker: read UNINITIALIZED __shared__ — does it contain SECRET_H? */
__global__ void __noinline__ attacker_read_shared(float *d_out) {
    __shared__ half sh[SHM_ELEMS * 2];  /* same size as victim's sh_a+sh_b */
    int tid = threadIdx.x;
    __syncwarp();
    for (int i = tid; i < SHM_ELEMS; i += WARP_SIZE)
        d_out[(long long)blockIdx.x * SHM_ELEMS + i] = __half2float(sh[i]);
}

/* ================================================================
 * Test C kernels: register spill to local memory (DRAM) residue
 * ================================================================ */

/* Victim: fill large local array (forces spill to local mem in DRAM) with SECRET */
__global__ void __noinline__ victim_local_spill(int *sink) {
    float local[LOCAL_N];
    #pragma unroll 1
    for (int i = 0; i < LOCAL_N; i++) local[i] = SECRET_F;
    /* Force all values to be actually written (not optimized away) */
    float sum = 0;
    #pragma unroll 1
    for (int i = 0; i < LOCAL_N; i++) sum += local[i];
    /* Defeat DCE with opaque condition */
    if ((int)sum == 0xDEAD) sink[threadIdx.x + blockIdx.x * WARP_SIZE] = (int)sum;
    /* Local memory (DRAM) now holds SECRET_F when kernel exits */
}

/* Attacker: read local array WITHOUT initialization — does it contain SECRET? */
__global__ void __noinline__ attacker_local_read(float *d_out) {
    float local[LOCAL_N];
    float sum = 0;
    #pragma unroll 1
    for (int i = 0; i < LOCAL_N; i++) sum += local[i];
    /* Store average so we can detect SECRET pattern */
    d_out[threadIdx.x + blockIdx.x * WARP_SIZE] = sum / LOCAL_N;
}

/* ================================================================
 * Host helpers
 * ================================================================ */
static int count_near(const float *buf, float target, float tol, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (fabsf(buf[i] - target) < tol) c++;
    return c;
}

/* ================================================================
 * main
 * ================================================================ */
int main() {
    CHECK(cudaSetDevice(1));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 1));
    printf("=== Experiment 28: Tensor Core (wmma) Fragment Residue ===\n");
    printf("    Device: %s  SMs=%d  compute=%d.%d\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    for (int pass = 1; pass <= 5; pass++) {
        printf("========== PASS %d / 5 ==========\n", pass);

        /* ---- Test A: wmma accumulator fragment register residue ---- */
        printf("[Test A] wmma accumulator fragment register state (pass=%d)\n", pass);
        {
            int   total_accum = (long long)N_WARPS * ACCUM_PER_WARP;
            float *d_sanity, *d_out;
            int   *d_sink;
            CHECK(cudaMalloc(&d_sanity, (long long)N_WARPS * ACCUM_PER_WARP * sizeof(float)));
            CHECK(cudaMalloc(&d_out,    (long long)N_WARPS * ACCUM_PER_WARP * sizeof(float)));
            CHECK(cudaMalloc(&d_sink,   N_WARPS * sizeof(int)));
            CHECK(cudaMemset(d_out, 0, (long long)N_WARPS * ACCUM_PER_WARP * sizeof(float)));

            /* Sanity: victim stores result → confirm SECRET_MMA = 50.265... */
            victim_fragment_store<<<N_WARPS, WARP_SIZE, 0, stream>>>(d_sanity);
            CHECK(cudaStreamSynchronize(stream));
            float *h_sanity = (float *)malloc((long long)N_WARPS * ACCUM_PER_WARP * sizeof(float));
            CHECK(cudaMemcpy(h_sanity, d_sanity, (long long)N_WARPS * ACCUM_PER_WARP * sizeof(float), cudaMemcpyDeviceToHost));
            int sanity_hits = count_near(h_sanity, SECRET_MMA, 0.1f, total_accum);
            printf("  Sanity (victim stores): SECRET_MMA=%d/%d  (val=%.4f)\n",
                   sanity_hits, total_accum, h_sanity[0]);
            free(h_sanity);

            /* Actual test: victim exits WITHOUT storing → attacker reads uninit fragment */
            victim_fragment<<<N_WARPS, WARP_SIZE, 0, stream>>>(d_sink);
            attacker_fragment<<<N_WARPS, WARP_SIZE, 0, stream>>>(d_out);
            CHECK(cudaStreamSynchronize(stream));

            float *h_out = (float *)malloc((long long)N_WARPS * ACCUM_PER_WARP * sizeof(float));
            CHECK(cudaMemcpy(h_out, d_out, (long long)N_WARPS * ACCUM_PER_WARP * sizeof(float), cudaMemcpyDeviceToHost));
            int hits = count_near(h_out, SECRET_MMA, 0.1f, total_accum);
            int zeros = count_near(h_out, 0.0f, 0.01f, total_accum);
            printf("  Attacker uninit fragment: SECRET_MMA=%d/%d  zeros=%d/%d\n",
                   hits, total_accum, zeros, total_accum);
            if (hits > total_accum / 4)
                printf("  [!!!] WMMA FRAGMENT REGISTER LEAK — accumulator SECRET crosses kernel boundary!\n");
            else
                printf("  SAFE — wmma fragment registers zeroed between kernel launches\n");
            free(h_out);
            CHECK(cudaFree(d_sanity)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_sink));
        }

        /* ---- Test B: __shared__ residue from wmma staging ---- */
        printf("[Test B] __shared__ residue from wmma shared-memory staging (pass=%d)\n", pass);
        {
            long long total = (long long)N_WARPS * SHM_ELEMS;
            float *d_out;
            int   *d_sink;
            CHECK(cudaMalloc(&d_out,  total * sizeof(float)));
            CHECK(cudaMalloc(&d_sink, N_WARPS * sizeof(int)));

            victim_wmma_shared<<<N_WARPS, WARP_SIZE, 0, stream>>>(d_sink);
            attacker_read_shared<<<N_WARPS, WARP_SIZE, 0, stream>>>(d_out);
            CHECK(cudaStreamSynchronize(stream));

            float *h_out = (float *)malloc(total * sizeof(float));
            CHECK(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));
            int hits   = count_near(h_out, SECRET_F, 0.01f, (int)total);
            int nz     = 0;
            for (long long i = 0; i < total; i++) if (fabsf(h_out[i]) > 0.001f) nz++;
            printf("  Attacker read __shared__ (after victim wmma staging): SECRET=%d/%lld  nz=%d/%lld\n",
                   hits, total, nz, total);
            /* ~1/SM_count fraction of attacker blocks run on same SM as victim → threshold ≈ 10% */
            if (hits > (int)(total / 10))
                printf("  [!!!] WMMA SHARED STAGING LEAK — %d/%lld blocks see victim __shared__ (1/SM reuse)\n",
                       hits / SHM_ELEMS, total / SHM_ELEMS);
            else
                printf("  SAFE — no __shared__ residue from wmma staging\n");
            free(h_out);
            CHECK(cudaFree(d_out)); CHECK(cudaFree(d_sink));
        }

        /* ---- Test C: local memory spill residue ---- */
        printf("[Test C] Local memory (DRAM) spill residue (pass=%d)\n", pass);
        {
            long long total = (long long)N_WARPS * WARP_SIZE;
            float *d_out;
            int   *d_sink;
            CHECK(cudaMalloc(&d_out,  total * sizeof(float)));
            CHECK(cudaMalloc(&d_sink, N_WARPS * sizeof(int)));

            victim_local_spill<<<N_WARPS, WARP_SIZE, 0, stream>>>(d_sink);
            attacker_local_read<<<N_WARPS, WARP_SIZE, 0, stream>>>(d_out);
            CHECK(cudaStreamSynchronize(stream));

            float *h_out = (float *)malloc(total * sizeof(float));
            CHECK(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));
            int hits = count_near(h_out, SECRET_F, 0.01f, (int)total);
            int nz   = 0;
            for (long long i = 0; i < total; i++) if (fabsf(h_out[i]) > 0.001f) nz++;
            printf("  Attacker local avg (should be SECRET=%.3f if leaked): hits=%d/%lld  nz=%d/%lld\n",
                   SECRET_F, hits, total, nz, total);
            /* Also print min/max to see what values are present */
            float mn = h_out[0], mx = h_out[0];
            for (long long i = 0; i < total; i++) {
                if (h_out[i] < mn) mn = h_out[i];
                if (h_out[i] > mx) mx = h_out[i];
            }
            printf("  Local read range: min=%.4f max=%.4f\n", mn, mx);
            if (hits > total / 4)
                printf("  [!!!] LOCAL MEMORY SPILL LEAK — DRAM spill region contains SECRET!\n");
            else if (fabsf(mn - SECRET_F) < 0.1f && fabsf(mx - SECRET_F) < 0.1f)
                printf("  [!!!] ALL VALUES = SECRET_F — unambiguous local memory leak!\n");
            else
                printf("  SAFE — local memory zeroed or not reused\n");
            free(h_out);
            CHECK(cudaFree(d_out)); CHECK(cudaFree(d_sink));
        }
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("A: wmma accumulator fragment (registers) — zeroed between launches?\n");
    printf("B: __shared__ after wmma::load_matrix_sync staging — persists?\n");
    printf("C: Local memory (DRAM) spill for large register arrays — persists?\n");
    printf("\n[Done]\n");

    CHECK(cudaStreamDestroy(stream));
    return 0;
}
