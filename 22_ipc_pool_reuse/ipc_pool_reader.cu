/*
 * Experiment 22: IPC + Pool reuse — Reader process
 *
 * Opens the IPC handle from the writer.
 * Also tests whether reader's own pool reuse can access the writer's pooled memory
 * at a different physical address (cross-process pool contamination).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

__global__ void count_match(const float *buf, int *cnt, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fabsf(buf[i] - val) < 1e-3f) atomicAdd(cnt, 1);
}

static int gpu_count_match(const float *buf, float val, int n) {
    int *d = NULL, h = 0;
    CHECK(cudaMalloc(&d, sizeof(int)));
    CHECK(cudaMemset(d, 0, sizeof(int)));
    count_match<<<(n+255)/256, 256>>>(buf, d, val, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d));
    return h;
}

int main(int argc, char **argv) {
    const char *hfile = argc > 1 ? argv[1] : "/tmp/ipc_pool_handle.bin";
    CHECK(cudaSetDevice(0));

    FILE *f = fopen(hfile, "rb");
    if (!f) { perror("fopen"); exit(1); }
    cudaIpcMemHandle_t handle;
    int N; float SECRET;
    (void)fread(&handle, sizeof(handle), 1, f);
    (void)fread(&N, sizeof(N), 1, f);
    (void)fread(&SECRET, sizeof(SECRET), 1, f);
    fclose(f);

    const size_t BYTES = (size_t)N * sizeof(float);
    printf("[reader] N=%d, WRITER_SECRET=%.5f, ANTI_SECRET=%.5f\n",
           N, SECRET, SECRET * -1.0f);

    /* Open IPC handle */
    float *p_ipc = NULL;
    CHECK(cudaIpcOpenMemHandle((void**)&p_ipc, handle, cudaIpcMemLazyEnablePeerAccess));
    printf("[reader] IPC ptr : %p\n", (void*)p_ipc);

    /* Read via IPC */
    float *h = (float*)malloc(BYTES);
    CHECK(cudaMemcpy(h, p_ipc, BYTES, cudaMemcpyDeviceToHost));

    int match_secret = 0, match_anti = 0, nonzero = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h[i] - SECRET) < 1e-3f) match_secret++;
        if (fabsf(h[i] - SECRET * -1.0f) < 1e-3f) match_anti++;
        if (h[i] != 0.0f) nonzero++;
    }
    printf("[reader] IPC read results:\n");
    printf("  Match SECRET      : %d/%d (writer pool secret — should be 0 if overwritten)\n",
           match_secret, N);
    printf("  Match ANTI-SECRET : %d/%d (writer overwrote with -SECRET)\n", match_anti, N);
    printf("  Non-zero          : %d/%d\n", nonzero, N);
    printf("\n");

    if (match_anti == N)
        printf("[IPC] As expected: writer's IPC block contains ANTI_SECRET (written after pool free)\n");
    if (match_secret > N/2)
        printf("[!!!] Pool residue leaked into IPC block — writer's pool contaminated cudaMalloc!\n");

    /* Reader's own pool: does reader's pool get contaminated by writer's pool block? */
    printf("[reader] Testing reader-side pool reuse contamination...\n");
    {
        cudaStream_t sr;
        CHECK(cudaStreamCreate(&sr));
        float *pr = NULL;
        CHECK(cudaMallocAsync(&pr, BYTES, sr));
        int ms = gpu_count_match(pr, SECRET, N);
        int ma = gpu_count_match(pr, SECRET * -1.0f, N);
        printf("  Fresh pool alloc in reader: match_secret=%d match_anti=%d\n", ms, ma);
        if (ms > N/2)
            printf("  [!!!] Cross-process pool contamination: reader got writer's SECRET\n");
        else if (ma > N/2)
            printf("  [!!!] Cross-process pool contamination: reader got writer's ANTI_SECRET\n");
        else
            printf("  SAFE: reader's pool alloc contains neither writer's value (pools are per-process)\n");
        CHECK(cudaFreeAsync(pr, sr));
        CHECK(cudaStreamSynchronize(sr));
        CHECK(cudaStreamDestroy(sr));
    }

    free(h);
    CHECK(cudaIpcCloseMemHandle(p_ipc));
    printf("\n[Done]\n");
    return 0;
}
