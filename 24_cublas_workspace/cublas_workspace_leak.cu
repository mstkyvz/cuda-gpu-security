/*
 * Experiment 24: cuBLAS Workspace Leak Test
 *
 * cuBLAS allocates a scratch "workspace" buffer for some GEMM algorithms.
 * Tests whether workspace retains data between calls (potential info leak).
 *
 * Tests:
 *   A. FP32 cublasSgemm: is workspace used / does it have residue?
 *   B. FP16 cublasGemmEx COMPUTE_16F: same
 *   C. FP16 cublasGemmEx COMPUTE_32F_FAST_16F: LLM default path
 *   D. Victim FP16 GEMM (A=SECRET) → inspect workspace after
 *   E. Sequential GEMMs: GEMM-1 residue in GEMM-2?
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"[cuBLAS] %s:%d status=%d\n",__FILE__,__LINE__,(int)_s); exit(1); } } while(0)

__global__ void fill_fp32(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void fill_fp16(__half *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = __float2half(val);
}

static int count_nz(const unsigned char *p, size_t n) {
    int c = 0;
    for (size_t i = 0; i < n; i++) if (p[i]) c++;
    return c;
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Experiment 24: cuBLAS Workspace Leak Test ===\n");
    printf("    Device : %s\n\n", prop.name);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    /* Enable Tensor Cores */
    cublasMath_t math_mode = CUBLAS_TENSOR_OP_MATH;
    CUBLAS_CHECK(cublasSetMathMode(handle, math_mode));

    const int M = 4096, N = 4096, K = 4096;
    const float SECRET = 3.14159f;
    const size_t WS = 64 * 1024 * 1024; /* 64 MB workspace */

    /* Matrix buffers */
    float  *d_A32, *d_B32, *d_C32;
    __half *d_A16, *d_B16, *d_C16;
    CHECK(cudaMalloc(&d_A32, (size_t)M*K*sizeof(float)));
    CHECK(cudaMalloc(&d_B32, (size_t)M*K*sizeof(float)));
    CHECK(cudaMalloc(&d_C32, (size_t)M*N*sizeof(float)));
    CHECK(cudaMalloc(&d_A16, (size_t)M*K*sizeof(__half)));
    CHECK(cudaMalloc(&d_B16, (size_t)M*K*sizeof(__half)));
    CHECK(cudaMalloc(&d_C16, (size_t)M*N*sizeof(__half)));

    void *d_ws;
    CHECK(cudaMalloc(&d_ws, WS));
    unsigned char *h_ws = (unsigned char*)malloc(WS);

    int blocks = (M*K + 255) / 256;
    float alpha32 = 1.0f, beta32 = 0.0f;
    __half alpha16 = __float2half(1.0f), beta16 = __float2half(0.0f);

    /* ---- Part A: FP32 SGEMM ---- */
    printf("[Part A] FP32 cublasSgemm — workspace usage\n");
    {
        fill_fp32<<<blocks, 256>>>(d_A32, 1.0f, M*K);
        fill_fp32<<<blocks, 256>>>(d_B32, 1.0f, M*K);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_ws, 0, WS));
        CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, WS));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha32, d_A32, M, d_B32, K, &beta32, d_C32, M));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_ws, d_ws, WS, cudaMemcpyDeviceToHost));
        int nz = count_nz(h_ws, WS);
        printf("  Non-zero bytes in workspace: %d / %zu (%.2f%%)\n",
               nz, WS, 100.0f*nz/WS);
        printf("  Verdict: %s\n\n", nz > 0 ? "[~] workspace used — residue present" :
               "SAFE — cuBLAS did not use workspace for FP32 SGEMM on H100");
    }

    /* ---- Part B: FP16 GEMM COMPUTE_16F ---- */
    printf("[Part B] FP16 cublasGemmEx (COMPUTE_16F) — workspace usage\n");
    {
        fill_fp16<<<blocks, 256>>>(d_A16, 1.0f, M*K);
        fill_fp16<<<blocks, 256>>>(d_B16, 1.0f, M*K);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_ws, 0, WS));
        CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, WS));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha16, d_A16, CUDA_R_16F, M,
                                d_B16, CUDA_R_16F, K,
                     &beta16,  d_C16, CUDA_R_16F, M,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_ws, d_ws, WS, cudaMemcpyDeviceToHost));
        int nz = count_nz(h_ws, WS);
        printf("  Non-zero bytes: %d / %zu (%.2f%%)\n", nz, WS, 100.0f*nz/WS);
        printf("  Verdict: %s\n\n", nz > 0 ? "[~] workspace used" :
               "SAFE — no workspace usage");
    }

    /* ---- Part C: FP16 GEMM COMPUTE_32F_FAST_16F (LLM default) ---- */
    printf("[Part C] FP16 cublasGemmEx (COMPUTE_32F_FAST_16F) — LLM inference path\n");
    {
        fill_fp16<<<blocks, 256>>>(d_A16, 1.0f, M*K);
        fill_fp16<<<blocks, 256>>>(d_B16, 1.0f, M*K);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_ws, 0, WS));
        CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, WS));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha32, d_A16, CUDA_R_16F, M,
                                 d_B16, CUDA_R_16F, K,
                     &beta32,   d_C16, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_ws, d_ws, WS, cudaMemcpyDeviceToHost));
        int nz = count_nz(h_ws, WS);
        printf("  Non-zero bytes: %d / %zu (%.2f%%)\n", nz, WS, 100.0f*nz/WS);
        printf("  Verdict: %s\n\n", nz > 0 ? "[~] workspace used — residue present" :
               "SAFE — no workspace usage for this compute type");
    }

    /* ---- Part D: SECRET input → inspect workspace for input traces ---- */
    printf("[Part D] Victim GEMM A=SECRET(%.5f) → workspace residue inspection\n", SECRET);
    {
        fill_fp16<<<blocks, 256>>>(d_A16, SECRET, M*K);
        fill_fp16<<<blocks, 256>>>(d_B16, 1.0f, M*K);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_ws, 0, WS));
        CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, WS));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha32, d_A16, CUDA_R_16F, M,
                                 d_B16, CUDA_R_16F, K,
                     &beta32,   d_C16, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_ws, d_ws, WS, cudaMemcpyDeviceToHost));

        int nz = count_nz(h_ws, WS);

        /* Search for FP16 bit-pattern of SECRET */
        __half sh = __float2half(SECRET);
        uint16_t sb; memcpy(&sb, &sh, 2);
        int fp16_hits = 0;
        for (size_t i = 0; i + 1 < WS; i += 2) {
            uint16_t v; memcpy(&v, h_ws + i, 2);
            if (v == sb) fp16_hits++;
        }
        /* Search for FP32 SECRET pattern */
        uint32_t s32b; memcpy(&s32b, &SECRET, 4);
        int fp32_hits = 0;
        for (size_t i = 0; i + 3 < WS; i += 4) {
            uint32_t v; memcpy(&v, h_ws + i, 4);
            if (v == s32b) fp32_hits++;
        }

        printf("  Non-zero bytes          : %d / %zu (%.2f%%)\n", nz, WS, 100.0f*nz/WS);
        printf("  FP16 SECRET matches     : %d (bit pattern 0x%04x)\n", fp16_hits, sb);
        printf("  FP32 SECRET matches     : %d (bit pattern 0x%08x)\n", fp32_hits, s32b);
        if (fp16_hits > 1000 || fp32_hits > 1000)
            printf("  Verdict: [!!!] SECRET input pattern found in workspace residue\n\n");
        else if (nz > (int)(WS/2))
            printf("  Verdict: [~] Workspace used but no direct SECRET pattern\n\n");
        else
            printf("  Verdict: SAFE — workspace not used for this GEMM type\n\n");
    }

    /* ---- Part E: Sequential GEMMs — GEMM-1 residue in GEMM-2? ---- */
    printf("[Part E] Sequential GEMMs: does GEMM-1 residue survive into GEMM-2?\n");
    {
        const float S1 = 1.11111f, S2 = 2.22222f;

        /* GEMM 1 */
        fill_fp16<<<blocks, 256>>>(d_A16, S1, M*K);
        fill_fp16<<<blocks, 256>>>(d_B16, 1.0f, M*K);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_ws, 0, WS));
        CUBLAS_CHECK(cublasSetWorkspace(handle, d_ws, WS));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha32, d_A16, CUDA_R_16F, M, d_B16, CUDA_R_16F, K,
            &beta32, d_C16, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_ws, d_ws, WS, cudaMemcpyDeviceToHost));
        int nz1 = count_nz(h_ws, WS);
        printf("  GEMM-1 workspace nz = %d\n", nz1);

        /* GEMM 2 */
        fill_fp16<<<blocks, 256>>>(d_A16, S2, M*K);
        CHECK(cudaDeviceSynchronize());
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha32, d_A16, CUDA_R_16F, M, d_B16, CUDA_R_16F, K,
            &beta32, d_C16, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_ws, d_ws, WS, cudaMemcpyDeviceToHost));
        int nz2 = count_nz(h_ws, WS);

        __half s1h = __float2half(S1); uint16_t s1b; memcpy(&s1b, &s1h, 2);
        int s1_hits = 0;
        for (size_t i = 0; i + 1 < WS; i += 2) {
            uint16_t v; memcpy(&v, h_ws + i, 2);
            if (v == s1b) s1_hits++;
        }
        printf("  GEMM-2 workspace nz = %d\n", nz2);
        printf("  GEMM-1 S1=%.5f pattern in GEMM-2 workspace: %d hits\n", S1, s1_hits);
        if (s1_hits > 1000)
            printf("  Verdict: [!!!] GEMM-1 residue visible after GEMM-2\n\n");
        else if (nz1 == 0 && nz2 == 0)
            printf("  Verdict: SAFE — cuBLAS does not use workspace; no residue\n\n");
        else
            printf("  Verdict: SAFE — GEMM-1 residue not found after GEMM-2\n\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    free(h_ws);
    CHECK(cudaFree(d_ws));
    CHECK(cudaFree(d_A32)); CHECK(cudaFree(d_B32)); CHECK(cudaFree(d_C32));
    CHECK(cudaFree(d_A16)); CHECK(cudaFree(d_B16)); CHECK(cudaFree(d_C16));
    printf("[Done]\n");
    return 0;
}
