#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (256*1024)
#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"[RT] %s\n",cudaGetErrorString(_e));exit(1);}} while(0)

int main() {
    CHECK(cudaSetDevice(0));
    const float S = 3.14159f;
    cudaStream_t s; CHECK(cudaStreamCreate(&s));
    for (int attempt = 0; attempt < 5; attempt++) {
        float *d; CHECK(cudaMallocAsync(&d, N*sizeof(float), s));
        CHECK(cudaStreamSynchronize(s));
        float *h = (float*)malloc(N*sizeof(float));
        CHECK(cudaMemcpy(h, d, N*sizeof(float), cudaMemcpyDeviceToHost));
        int m=0; for(int i=0;i<N;i++) if(fabsf(h[i]-S)<1e-3f) m++;
        printf("[R] attempt %d: ptr=%p matches=%d/%d\n", attempt, (void*)d, m, N);
        if (m==N) printf("[!!!] MPS POOL LEAK — reader got writer pool memory!\n");
        free(h);
        CHECK(cudaFreeAsync(d, s));
        CHECK(cudaStreamSynchronize(s));
    }
    CHECK(cudaStreamDestroy(s));
    return 0;
}
