/*
 * IPC Writer: Allocates GPU memory, fills with secret, exports IPC handle to file.
 * Run BEFORE ipc_reader.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"[CUDA] %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void fill_secret(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

int main(int argc, char **argv) {
    const char *handle_file = argc > 1 ? argv[1] : "/tmp/cuda_ipc_handle.bin";
    const int N = 64 * 1024;
    const float SECRET = 2.71828f;

    CHECK(cudaSetDevice(0));
    float *d = NULL;
    CHECK(cudaMalloc(&d, N * sizeof(float)));
    fill_secret<<<(N+255)/256, 256>>>(d, SECRET, N);
    CHECK(cudaDeviceSynchronize());

    cudaIpcMemHandle_t h;
    CHECK(cudaIpcGetMemHandle(&h, d));

    FILE *f = fopen(handle_file, "wb");
    if (!f) { perror("fopen"); exit(1); }
    fwrite(&h, sizeof(h), 1, f);
    fwrite(&N, sizeof(N), 1, f);
    fwrite(&SECRET, sizeof(SECRET), 1, f);
    fclose(f);

    printf("[writer] GPU ptr : %p\n", (void*)d);
    printf("[writer] Secret  : %.5f  (N=%d floats)\n", SECRET, N);
    printf("[writer] Handle written to: %s\n", handle_file);
    printf("[writer] Waiting 10 sec for reader...\n");
    fflush(stdout);

    /* Keep allocation alive while reader runs */
    sleep(10);

    printf("[writer] Done. Freeing GPU memory.\n");
    CHECK(cudaFree(d));
    return 0;
}
