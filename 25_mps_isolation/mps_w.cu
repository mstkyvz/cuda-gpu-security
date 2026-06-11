#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N (256*1024)
#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"[RT] %s\n",cudaGetErrorString(_e));exit(1);}} while(0)

__global__ void fill(float *b, float v, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) b[i]=v;
}

int main() {
    CHECK(cudaSetDevice(0));
    const float S = 3.14159f;
    cudaStream_t s; CHECK(cudaStreamCreate(&s));
    float *d; CHECK(cudaMallocAsync(&d, N*sizeof(float), s));
    fill<<<(N+255)/256,256,0,s>>>(d, S, N);
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaFreeAsync(d, s));
    CHECK(cudaStreamSynchronize(s));
    printf("[W] pool alloc ptr=%p, filled S=%.5f, freed to pool\n", (void*)d, S);
    fflush(stdout);
    FILE *f = fopen("/tmp/mps_w_done","w"); fclose(f);
    sleep(8);
    CHECK(cudaStreamDestroy(s));
    return 0;
}
