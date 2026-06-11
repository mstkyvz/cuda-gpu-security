/*
 * Experiment 22: IPC + Pool reuse combined attack
 *
 * Combines experiments 15 (IPC) and 1 (pool leak):
 *   - Writer uses PyTorch-style pool (cudaMallocAsync), fills with secret, "frees" (pooled)
 *   - Writer then does a fresh cudaMalloc (NOT pooled) and exports IPC handle
 *   - Reader opens the IPC handle — but also shares the SAME CUDA context pool
 *
 * The key question: if process A leaks via pool, and process B has an IPC handle
 * to process A's freshly-allocated block, can B use B's own pool reuse to access A's data?
 *
 * This tests the combination vector: IPC handle sharing + pool residue leak
 * in a realistic multi-tenant scenario (two containerized inference servers
 * sharing the same physical GPU).
 *
 * This is the writer process. Run BEFORE ipc_pool_reader.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

__global__ void fill_kernel(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

int main(int argc, char **argv) {
    const char *hfile = argc > 1 ? argv[1] : "/tmp/ipc_pool_handle.bin";
    CHECK(cudaSetDevice(0));

    const int N = 128 * 1024;
    const float SECRET = 3.14159f;
    const size_t BYTES = (size_t)N * sizeof(float);

    /* Step 1: Allocate via pool, fill with secret, "free" (stays pooled) */
    cudaStream_t s0;
    CHECK(cudaStreamCreate(&s0));
    float *p_pool = NULL;
    CHECK(cudaMallocAsync(&p_pool, BYTES, s0));
    fill_kernel<<<(N+255)/256, 256, 0, s0>>>(p_pool, SECRET, N);
    CHECK(cudaStreamSynchronize(s0));
    /* "Free" to pool — data still in physical memory */
    CHECK(cudaFreeAsync(p_pool, s0));
    CHECK(cudaStreamSynchronize(s0));
    printf("[writer] Pool alloc filled with SECRET=%.5f then freed to pool\n", SECRET);

    /* Step 2: Regular cudaMalloc — driver may hand back same physical page */
    float *p_ipc = NULL;
    CHECK(cudaMalloc(&p_ipc, BYTES));
    printf("[writer] cudaMalloc ptr : %p\n", (void*)p_ipc);

    /* Check if pool residue survived into cudaMalloc allocation */
    float *h = (float*)malloc(BYTES);
    CHECK(cudaMemcpy(h, p_ipc, BYTES, cudaMemcpyDeviceToHost));
    int m = 0;
    for (int i = 0; i < N; i++) if (fabsf(h[i] - SECRET) < 1e-3f) m++;
    printf("[writer] cudaMalloc self-check: %d/%d match (pool→cudaMalloc residue?)\n", m, N);
    free(h);

    /* Step 3: Export IPC handle */
    cudaIpcMemHandle_t handle;
    CHECK(cudaIpcGetMemHandle(&handle, p_ipc));

    FILE *f = fopen(hfile, "wb");
    if (!f) { perror("fopen"); exit(1); }
    fwrite(&handle, sizeof(handle), 1, f);
    fwrite(&N, sizeof(N), 1, f);
    fwrite(&SECRET, sizeof(SECRET), 1, f);
    fclose(f);

    printf("[writer] IPC handle written to %s\n", hfile);
    printf("[writer] Waiting 15 sec for reader...\n");
    fflush(stdout);

    /* Also fill the IPC allocation with a DIFFERENT value to distinguish pool residue */
    fill_kernel<<<(N+255)/256, 256>>>(p_ipc, SECRET * -1.0f, N);
    CHECK(cudaDeviceSynchronize());
    printf("[writer] IPC block overwritten with ANTI-SECRET=%.5f\n", SECRET * -1.0f);
    fflush(stdout);

    sleep(15);

    printf("[writer] Done.\n");
    CHECK(cudaFree(p_ipc));
    CHECK(cudaStreamDestroy(s0));
    return 0;
}
