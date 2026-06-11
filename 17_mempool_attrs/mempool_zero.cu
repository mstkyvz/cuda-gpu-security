/*
 * Experiment 17: cudaMemPool Attributes — Built-in Zeroing Support?
 *
 * CUDA 11.2+ stream-ordered memory pools expose attributes via:
 *   cudaMemPoolSetAttribute / cudaMemPoolGetAttribute
 *
 * Known attributes:
 *   cudaMemPoolAttrReleaseThreshold      — how much memory to hold vs return to OS
 *   cudaMemPoolReuseFollowEventDependencies
 *   cudaMemPoolReuseAllowOpportunistic
 *   cudaMemPoolReuseAllowInternalDependencies
 *
 * Question: Is there a "zero on release/reuse" attribute?
 * If so, setting it would make cudaMallocAsync safe without changing code.
 *
 * This experiment also tests:
 *   A. Default pool behavior (no zeroing → confirms Exp 1B)
 *   B. cudaMemPoolAttrReleaseThreshold = 0 (force return to OS → driver zeroes)
 *   C. Custom pool with different attributes
 *   D. cudaMemPoolTrimTo — does trimming to 0 then re-growing zero memory?
 *   E. The "no sync" vs "with sync" split: full table of all 4 cases
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA] %s:%d %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while(0)

__global__ void fill_secret(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

/* Count how many elements match val within tolerance */
int count_matches(float *h, int n, float val) {
    int cnt = 0;
    for (int i = 0; i < n; i++)
        if (__builtin_fabsf(h[i] - val) < 1e-3f) cnt++;
    return cnt;
}

int main() {
    const int N     = 4096;
    const float SEC = 2.71828f;
    float h[N];

    printf("=== cudaMemPool Attributes — Zeroing Behavior Test ===\n\n");

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    /* -------------------------------------------------------
     * Table: sync vs no-sync, same-stream vs diff-stream
     * ------------------------------------------------------- */
    printf("[Part A] Complete sync × stream matrix for cudaMallocAsync\n");
    printf("%-35s  %-8s  %-10s  %s\n",
           "Scenario", "Ptr-same", "Matches", "Verdict");
    printf("%s\n", "-----------------------------------------------------------------------");

    struct {
        const char *label;
        int same_stream;
        int do_sync;
    } cases[] = {
        {"same stream, no sync",   1, 0},
        {"same stream, with sync", 1, 1},
        {"diff stream, no sync",   0, 0},
        {"diff stream, with sync", 0, 1},
    };

    for (int c = 0; c < 4; c++) {
        cudaStream_t sW, sR;
        CHECK(cudaStreamCreate(&sW));
        CHECK(cudaStreamCreate(&sR));
        cudaStream_t writer_s = sW;
        cudaStream_t reader_s = cases[c].same_stream ? sW : sR;

        float *d_w = NULL, *d_r = NULL;
        CHECK(cudaMallocAsync(&d_w, N * sizeof(float), writer_s));
        fill_secret<<<(N+255)/256, 256, 0, writer_s>>>(d_w, SEC, N);
        CHECK(cudaFreeAsync(d_w, writer_s));

        if (cases[c].do_sync)
            CHECK(cudaStreamSynchronize(writer_s));

        CHECK(cudaMallocAsync(&d_r, N * sizeof(float), reader_s));
        CHECK(cudaMemcpyAsync(h, d_r, N * sizeof(float),
                              cudaMemcpyDeviceToHost, reader_s));
        CHECK(cudaStreamSynchronize(reader_s));

        int m = count_matches(h, N, SEC);
        printf("%-35s  %-8s  %4d/%-4d    %s\n",
               cases[c].label,
               d_w == d_r ? "YES" : "no",
               m, N,
               m == N ? "FULL LEAK" :
               m == 0 ? "SAFE (zeroed)" :
                        "PARTIAL");

        CHECK(cudaFreeAsync(d_r, reader_s));
        CHECK(cudaStreamSynchronize(reader_s));
        CHECK(cudaStreamDestroy(sW));
        if (!cases[c].same_stream) CHECK(cudaStreamDestroy(sR));
    }
    printf("\n");

    /* -------------------------------------------------------
     * Part B: cudaMemPoolAttrReleaseThreshold = 0
     * Force all freed memory to return to OS immediately.
     * OS/driver will zero on re-acquisition → should be safe.
     * ------------------------------------------------------- */
    printf("[Part B] cudaMemPoolAttrReleaseThreshold = 0 (force return to OS)\n");
    {
        cudaMemPool_t pool;
        CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));

        uint64_t threshold = 0;
        CHECK(cudaMemPoolSetAttribute(pool,
                                      cudaMemPoolAttrReleaseThreshold,
                                      &threshold));

        float *d_w = NULL, *d_r = NULL;
        CHECK(cudaMallocAsync(&d_w, N * sizeof(float), s));
        fill_secret<<<(N+255)/256, 256, 0, s>>>(d_w, SEC, N);
        CHECK(cudaFreeAsync(d_w, s));
        CHECK(cudaStreamSynchronize(s));  /* force memory back to OS */

        CHECK(cudaMallocAsync(&d_r, N * sizeof(float), s));
        CHECK(cudaMemcpyAsync(h, d_r, N * sizeof(float),
                              cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));

        int m = count_matches(h, N, SEC);
        printf("  Ptr same  : %s\n", d_w == d_r ? "YES" : "NO");
        printf("  Matches   : %d/%d\n", m, N);
        printf("  Verdict   : %s\n\n",
               m == 0 ? "SAFE (driver zeroes after OS return)" :
               m == N ? "LEAK (OS return didn't help)" : "PARTIAL");

        /* Restore threshold */
        threshold = UINT64_MAX;
        CHECK(cudaMemPoolSetAttribute(pool,
                                      cudaMemPoolAttrReleaseThreshold,
                                      &threshold));
        CHECK(cudaFreeAsync(d_r, s));
        CHECK(cudaStreamSynchronize(s));
    }

    /* -------------------------------------------------------
     * Part C: cudaMemPoolTrimTo — shrink pool to minimum,
     * then reallocate. New physical pages should be zeroed.
     * ------------------------------------------------------- */
    printf("[Part C] cudaMemPoolTrimTo(pool, 0) before reallocation\n");
    {
        cudaMemPool_t pool;
        CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));

        float *d_w = NULL, *d_r = NULL;
        CHECK(cudaMallocAsync(&d_w, N * sizeof(float), s));
        fill_secret<<<(N+255)/256, 256, 0, s>>>(d_w, SEC, N);
        CHECK(cudaFreeAsync(d_w, s));
        CHECK(cudaStreamSynchronize(s));

        /* Trim: return all unused pages to OS */
        CHECK(cudaMemPoolTrimTo(pool, 0));

        CHECK(cudaMallocAsync(&d_r, N * sizeof(float), s));
        CHECK(cudaMemcpyAsync(h, d_r, N * sizeof(float),
                              cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));

        int m = count_matches(h, N, SEC);
        printf("  Ptr same  : %s\n", d_w == d_r ? "YES" : "NO");
        printf("  Matches   : %d/%d\n", m, N);
        printf("  Verdict   : %s\n\n",
               m == 0 ? "SAFE (trim+OS return zeroes memory)" :
               m == N ? "LEAK (trim didn't return this block)" : "PARTIAL");

        CHECK(cudaFreeAsync(d_r, s));
        CHECK(cudaStreamSynchronize(s));
    }

    /* -------------------------------------------------------
     * Part D: Custom pool with all reuse flags disabled
     * cudaMemPoolReuseFollowEventDependencies = 0
     * cudaMemPoolReuseAllowOpportunistic = 0
     * cudaMemPoolReuseAllowInternalDependencies = 0
     * ------------------------------------------------------- */
    printf("[Part D] Custom pool with all reuse flags disabled\n");
    {
        cudaMemPoolProps props = {};
        props.allocType   = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id   = 0;

        cudaMemPool_t custom_pool;
        cudaError_t err = cudaMemPoolCreate(&custom_pool, &props);
        if (err != cudaSuccess) {
            printf("  cudaMemPoolCreate failed: %s\n", cudaGetErrorString(err));
        } else {
            /* Disable opportunistic reuse */
            int flag = 0;
            cudaMemPoolSetAttribute(custom_pool,
                                    cudaMemPoolReuseAllowOpportunistic, &flag);
            cudaMemPoolSetAttribute(custom_pool,
                                    cudaMemPoolReuseFollowEventDependencies, &flag);

            float *d_w = NULL, *d_r = NULL;
            err = cudaMallocFromPoolAsync(&d_w, N*sizeof(float), custom_pool, s);
            if (err == cudaSuccess) {
                fill_secret<<<(N+255)/256, 256, 0, s>>>(d_w, SEC, N);
                CHECK(cudaFreeAsync(d_w, s));
                CHECK(cudaStreamSynchronize(s));

                err = cudaMallocFromPoolAsync(&d_r, N*sizeof(float), custom_pool, s);
                if (err == cudaSuccess) {
                    CHECK(cudaMemcpyAsync(h, d_r, N*sizeof(float),
                                          cudaMemcpyDeviceToHost, s));
                    CHECK(cudaStreamSynchronize(s));

                    int m = count_matches(h, N, SEC);
                    printf("  Ptr same  : %s\n", d_w == d_r ? "YES" : "NO");
                    printf("  Matches   : %d/%d\n", m, N);
                    printf("  Verdict   : %s\n",
                           m == 0 ? "SAFE (reuse disabled + driver zeroes)" :
                           m == N ? "LEAK (reuse disabled, still same block)" :
                                    "PARTIAL");
                    CHECK(cudaFreeAsync(d_r, s));
                    CHECK(cudaStreamSynchronize(s));
                } else {
                    printf("  Second alloc failed: %s\n", cudaGetErrorString(err));
                }
            } else {
                printf("  First alloc failed: %s\n", cudaGetErrorString(err));
            }
            CHECK(cudaMemPoolDestroy(custom_pool));
        }
    }
    printf("\n");

    CHECK(cudaStreamDestroy(s));
    printf("[Done]\n");
    return 0;
}
