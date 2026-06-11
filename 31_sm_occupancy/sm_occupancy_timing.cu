/*
 * Experiment 31: SM Occupancy Timing Side-Channel (v2)
 *
 * Tests whether an attacker CUDA kernel can detect a victim kernel's presence
 * by measuring execution time degradation due to SM resource sharing.
 *
 * H100: 132 SMs, up to 64 warps/SM, 128KB L1/SM, 50MB L2 shared.
 *
 * Tests:
 *   A. SM Compute Throughput: attacker launches many blocks (2112 = 132×16),
 *      performs fixed FP32 work per block, measures total elapsed wall time.
 *      Victim also runs 2112 blocks. Both forced to share SMs.
 *      If SMs are shared: attacker total time increases.
 *
 *   B. HBM Bandwidth Contention: attacker sweeps its OWN 64MB buffer
 *      (larger than 50MB L2 → guaranteed HBM misses), measures cycles per
 *      cache-line-sized access. Victim sweeps its own SEPARATE 64MB buffer.
 *      If HBM bandwidth saturated: attacker latency increases.
 *
 *   C. Warp Scheduling Occupancy: attacker fills 1 SM fully (64 warps = 2 blocks
 *      of 1024 threads) and performs a fixed iteration loop, measuring cycle count
 *      via one designated "timer warp". Victim targets same SMs with many blocks.
 *      Co-scheduling pushes victim warps onto attacker's SM → scheduling slots
 *      divided → attacker completes later.
 *
 * Compile:
 *   nvcc -O2 -arch=sm_90 -o sm_occupancy_timing sm_occupancy_timing.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

/* H100 SM count */
#define N_SMS        132
/* Blocks per SM for the "many-block" tests (exceeds SM capacity → co-scheduling) */
#define BLOCKS_PER_SM 16
#define N_BLOCKS     (N_SMS * BLOCKS_PER_SM)   /* 2112 */
/* FP iterations per block in Test A */
#define FP_ITERS     (1 << 18)                 /* 262144 */
/* HBM buffer: 64MB > 50MB L2, guarantees cache misses on repeated sweep */
#define HBM_FLOATS   (16 * 1024 * 1024)        /* 64MB */

/* ================================================================
 * Helper: device __forceinline__ ldcg bypasses L1
 * ================================================================ */
__device__ __forceinline__ float ldcg(const float *p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];\n\tmembar.gl;\n\t"
                 : "=f"(v) : "l"(p) : "memory");
    return v;
}

/* ================================================================
 * Test A: attacker compute kernel
 * Each block performs FP_ITERS dependent FP32 MUL-ADD ops.
 * Total throughput is limited by SM availability.
 * ================================================================ */
__global__ void attacker_compute(float *d_guard, long long FP_ITERS_L) {
    float x = (float)(blockIdx.x * blockDim.x + threadIdx.x + 1);
    for (long long i = 0; i < FP_ITERS_L; i++) {
        x = x * 1.00001f + 0.000001f;
    }
    /* Anti-optimization: write result to prevent dead-code elimination */
    if ((long long)x == 0xDEADBEEF) d_guard[blockIdx.x] = x;
}

/* ================================================================
 * Victim compute: same configuration as attacker
 * ================================================================ */
__global__ void victim_compute(float *d_guard, long long FP_ITERS_L) {
    float x = (float)(blockIdx.x * blockDim.x + threadIdx.x + 2);
    for (long long i = 0; i < FP_ITERS_L; i++) {
        x = x * 1.00002f + 0.000002f;
    }
    if ((long long)x == 0xDEADBEEF) d_guard[blockIdx.x] = x;
}

/* ================================================================
 * Test B: attacker memory sweep kernel
 * Sweeps entire HBM_FLOATS buffer with ldcg (bypasses L1, misses L2
 * because buffer size > L2). Counts total cycles across all accesses.
 * Returns avg cycles per access in d_lat[0].
 * ================================================================ */
__global__ void attacker_hbm_sweep(float *buf, long long *d_lat) {
    /* Strided access pattern: each thread accesses distinct cache lines */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    long long t0 = clock64();
    float sum = 0.0f;
    for (int i = tid; i < HBM_FLOATS; i += stride) {
        sum += ldcg(buf + i);
    }
    long long t1 = clock64();

    if ((long long)sum == 0xDEAD) buf[0] = sum;

    /* Reduction: only thread 0 of block 0 writes final latency */
    __shared__ long long blk_lat;
    if (threadIdx.x == 0 && blockIdx.x == 0) blk_lat = t1 - t0;
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        /* Normalize to total accesses per thread */
        int accesses_per_thread = (HBM_FLOATS + stride - 1) / stride;
        d_lat[0] = (accesses_per_thread > 0) ? blk_lat / accesses_per_thread : 0;
    }
}

/* ================================================================
 * Victim HBM sweep (separate buffer to avoid L2 warming attacker)
 * Runs for target_clocks SM cycles to ensure overlap with attacker.
 * ================================================================ */
__global__ void victim_hbm_sweep(float *buf, long long target_clocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    long long start = clock64();
    while (clock64() - start < target_clocks) {
        for (int i = tid; i < HBM_FLOATS; i += stride) sum += ldcg(buf + i);
    }
    if ((long long)sum == 0xDEAD) buf[0] = sum;
}

/* ================================================================
 * Buffer initialization
 * ================================================================ */
__global__ void init_buf(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (float)(i % 997 + 1);
}

/* Wall-clock time in microseconds */
static long long wall_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

int main() {
    CHECK(cudaSetDevice(1));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 1));
    printf("=== Experiment 31: SM Occupancy Timing Side-Channel (v2) ===\n");
    printf("    Device: %s  SMs=%d  clockRate=%d MHz  L2=%d MB\n\n",
           prop.name, prop.multiProcessorCount,
           prop.clockRate / 1000, prop.l2CacheSize / (1024*1024));

    /* Allocations */
    float     *d_guard_a, *d_guard_v, *d_atk_buf, *d_vic_buf;
    long long *d_lat;
    CHECK(cudaMalloc(&d_guard_a, N_BLOCKS * sizeof(float)));
    CHECK(cudaMalloc(&d_guard_v, N_BLOCKS * sizeof(float)));
    CHECK(cudaMalloc(&d_atk_buf, (long long)HBM_FLOATS * sizeof(float)));
    CHECK(cudaMalloc(&d_vic_buf, (long long)HBM_FLOATS * sizeof(float)));
    CHECK(cudaMalloc(&d_lat,     sizeof(long long)));

    /* Initialize HBM buffers */
    init_buf<<<(HBM_FLOATS+255)/256, 256>>>(d_atk_buf, HBM_FLOATS);
    init_buf<<<(HBM_FLOATS+255)/256, 256>>>(d_vic_buf, HBM_FLOATS);
    CHECK(cudaDeviceSynchronize());

    for (int pass = 1; pass <= 5; pass++) {
        printf("========== PASS %d / 5 ==========\n", pass);

        /* ---- Test A: SM compute throughput ---- */
        printf("[Test A] SM compute throughput — %d blocks × 128 threads × %d FP iters\n",
               N_BLOCKS, FP_ITERS);
        {
            cudaStream_t sA, sV;
            CHECK(cudaStreamCreate(&sA));
            CHECK(cudaStreamCreate(&sV));

            /* Baseline: attacker alone */
            CHECK(cudaMemset(d_guard_a, 0, N_BLOCKS * sizeof(float)));
            long long w0 = wall_us();
            attacker_compute<<<N_BLOCKS, 128, 0, sA>>>(d_guard_a, (long long)FP_ITERS);
            CHECK(cudaStreamSynchronize(sA));
            long long base_us = wall_us() - w0;

            /* With victim: both run concurrently */
            CHECK(cudaMemset(d_guard_a, 0, N_BLOCKS * sizeof(float)));
            CHECK(cudaMemset(d_guard_v, 0, N_BLOCKS * sizeof(float)));
            /* Start victim first, then attacker */
            victim_compute<<<N_BLOCKS, 128, 0, sV>>>(d_guard_v, (long long)FP_ITERS);
            attacker_compute<<<N_BLOCKS, 128, 0, sA>>>(d_guard_a, (long long)FP_ITERS);
            long long wa0 = wall_us();
            CHECK(cudaStreamSynchronize(sA));
            long long load_us = wall_us() - wa0;
            CHECK(cudaStreamSynchronize(sV));

            printf("  Baseline (attacker alone): %lld ms\n", base_us / 1000);
            printf("  With victim (concurrent):  %lld ms (attacker portion)\n", load_us / 1000);
            double ratio = (double)load_us / (double)base_us;
            printf("  Slowdown: %.2fx", ratio);
            if (ratio > 1.3)
                printf("  [!!!] SM CO-SCHEDULING DETECTED — attacker slowed by %.2fx\n", ratio);
            else
                printf("  [~] No measurable SM co-scheduling slowdown\n");

            CHECK(cudaStreamDestroy(sA));
            CHECK(cudaStreamDestroy(sV));
        }

        /* ---- Test B: HBM bandwidth contention ---- */
        printf("[Test B] HBM bandwidth contention — attacker sweeps 64MB buffer (ldcg, bypasses L2)\n");
        {
            cudaStream_t sA, sV;
            CHECK(cudaStreamCreate(&sA));
            CHECK(cudaStreamCreate(&sV));
            long long h_lat;

            /* Baseline: attacker sweeps its buffer alone */
            CHECK(cudaMemset(d_lat, 0, sizeof(long long)));
            attacker_hbm_sweep<<<N_BLOCKS, 128, 0, sA>>>(d_atk_buf, d_lat);
            CHECK(cudaStreamSynchronize(sA));
            CHECK(cudaMemcpy(&h_lat, d_lat, sizeof(long long), cudaMemcpyDeviceToHost));
            long long base_lat = h_lat;

            /* With victim sweeping its SEPARATE buffer for ~1s (saturates HBM bandwidth) */
            long long clk_1s = (long long)prop.clockRate * 1000LL;
            victim_hbm_sweep<<<N_BLOCKS, 128, 0, sV>>>(d_vic_buf, clk_1s);
            usleep(50000);  /* 50ms: let victim start and run */
            CHECK(cudaMemset(d_lat, 0, sizeof(long long)));
            attacker_hbm_sweep<<<N_BLOCKS, 128, 0, sA>>>(d_atk_buf, d_lat);
            CHECK(cudaStreamSynchronize(sA));
            CHECK(cudaMemcpy(&h_lat, d_lat, sizeof(long long), cudaMemcpyDeviceToHost));
            long long load_lat = h_lat;
            CHECK(cudaStreamSynchronize(sV));

            printf("  Baseline latency (no victim): %lld cycles/access\n", base_lat);
            printf("  Latency with victim HBM load: %lld cycles/access\n", load_lat);
            double bw_ratio = (base_lat > 0) ? (double)load_lat / (double)base_lat : 1.0;
            printf("  Ratio: %.2fx", bw_ratio);
            if (bw_ratio > 1.3)
                printf("  [!!!] HBM CONTENTION DETECTED — %.2fx slower under victim HBM load\n", bw_ratio);
            else
                printf("  [~] No measurable HBM bandwidth contention\n");

            CHECK(cudaStreamDestroy(sA));
            CHECK(cudaStreamDestroy(sV));
        }
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("A: SM compute throughput — does attacker slow down when victim occupies SMs?\n");
    printf("B: HBM bandwidth — does victim's memory usage increase attacker's HBM latency?\n");
    printf("\n[Done]\n");

    CHECK(cudaFree(d_guard_a)); CHECK(cudaFree(d_guard_v));
    CHECK(cudaFree(d_atk_buf)); CHECK(cudaFree(d_vic_buf));
    CHECK(cudaFree(d_lat));
    return 0;
}
