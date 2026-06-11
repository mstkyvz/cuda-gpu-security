/*
 * Experiment 29: CUDA IPC Cross-Process GPU Memory Isolation
 *
 * cudaIpcGetMemHandle / cudaIpcOpenMemHandle allow two separate OS processes to
 * share a GPU buffer without copying: process A exports a handle, process B
 * opens it and gets a device pointer to A's memory.
 *
 * Security question: when process A writes SECRET, then either:
 *   (a) shares the handle while buffer still contains SECRET
 *   (b) frees the buffer to pool, then re-exports from pool
 *   (c) exports before writing, B opens BEFORE A writes
 * — does process B see A's SECRET?
 *
 * Tests:
 *   A. Direct share: A writes SECRET, exports handle, B opens and reads.
 *   B. Async pool residue via IPC: A writes SECRET to pool alloc, frees,
 *      re-allocs from pool (gets same block), exports, B reads.
 *   C. Lazy-open: A exports empty buffer; A writes SECRET; B opens and reads
 *      (tests whether IPC maps physical memory directly or snaps on open).
 *
 * Architecture:
 *   This file compiles to a single binary with two modes:
 *     ./cuda_ipc_residue writer <handle_file>
 *     ./cuda_ipc_residue reader <handle_file>
 *   The test harness (main with fork+exec or pipes) orchestrates the sequence.
 *
 *   For simplicity: we use fork()+exec() within a single binary.
 *
 * Compile:
 *   nvcc -O2 -arch=sm_90 -o cuda_ipc_residue cuda_ipc_residue.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

#define N        (256 * 1024)   /* 1 MB buffer (256K floats) */
#define SECRET   3.14159f

static int count_match(const float *h, float v, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (fabsf(h[i] - v) < 1e-3f) c++;
    return c;
}

__global__ void fill_buf(float *p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void zero_buf(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

/* Shared memory between parent and child via mmap'd anonymous region */
typedef struct {
    cudaIpcMemHandle_t handle;
    volatile int       writer_done;   /* 1 = writer exported handle */
    volatile int       reader_done;   /* 1 = reader finished reading */
    int                result_hits;   /* written by reader, read by parent */
    int                result_n;
    uintptr_t          writer_ptr;    /* writer's d_buf pointer value */
} SharedState;

/* ================================================================
 * Writer: allocates GPU buffer, writes SECRET, exports handle
 * ================================================================ */
static void run_writer(SharedState *sh, int test_id) {
    cudaDeviceReset();
    CHECK(cudaSetDevice(1));
    float *d_buf;
    int blocks = (N + 255) / 256;

    if (test_id == 0) {
        /* Test A: write SECRET directly, export handle */
        CHECK(cudaMalloc(&d_buf, N * sizeof(float)));
        fill_buf<<<blocks, 256>>>(d_buf, SECRET, N);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaIpcGetMemHandle(&sh->handle, d_buf));
        sh->writer_ptr = (uintptr_t)d_buf;
        __sync_synchronize();
        sh->writer_done = 1;
        /* Wait for reader to finish, then free */
        while (!sh->reader_done) usleep(1000);
        CHECK(cudaFree(d_buf));

    } else if (test_id == 1) {
        /* Test B: pool residue via IPC
         * Phase 1: alloc from pool, write SECRET, free back to pool
         * Phase 2: re-alloc from pool (same block), export via IPC */
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        float *d_pre;
        CHECK(cudaMallocAsync(&d_pre, N * sizeof(float), stream));
        fill_buf<<<blocks, 256, 0, stream>>>(d_pre, SECRET, N);
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaFreeAsync(d_pre, stream));
        CHECK(cudaStreamSynchronize(stream));

        /* Re-alloc — same stream pool should return same block with SECRET */
        CHECK(cudaMallocAsync(&d_buf, N * sizeof(float), stream));
        CHECK(cudaStreamSynchronize(stream));

        /* Export IPC handle for the re-allocated block (contains SECRET) */
        /* Note: IPC only works with cudaMalloc, not cudaMallocAsync pools directly.
         * We copy to a cudaMalloc buffer for the IPC test. */
        float *d_ipc;
        CHECK(cudaMalloc(&d_ipc, N * sizeof(float)));
        /* Copy WITHOUT zeroing — pool block still has SECRET */
        CHECK(cudaMemcpy(d_ipc, d_buf, N * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK(cudaIpcGetMemHandle(&sh->handle, d_ipc));
        sh->writer_ptr = (uintptr_t)d_ipc;
        __sync_synchronize();
        sh->writer_done = 1;

        while (!sh->reader_done) usleep(1000);
        CHECK(cudaFreeAsync(d_buf, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaFree(d_ipc));
        CHECK(cudaStreamDestroy(stream));

    } else {
        /* Test C: export EMPTY buffer, then write SECRET, reader opens after */
        CHECK(cudaMalloc(&d_buf, N * sizeof(float)));
        zero_buf<<<blocks, 256>>>(d_buf, N);
        CHECK(cudaDeviceSynchronize());
        /* Export handle while buffer is ZERO */
        CHECK(cudaIpcGetMemHandle(&sh->handle, d_buf));
        sh->writer_ptr = (uintptr_t)d_buf;
        __sync_synchronize();
        sh->writer_done = 1;
        /* After signaling, write SECRET (reader opens AFTER this signal) */
        fill_buf<<<blocks, 256>>>(d_buf, SECRET, N);
        CHECK(cudaDeviceSynchronize());
        while (!sh->reader_done) usleep(1000);
        CHECK(cudaFree(d_buf));
    }
}

/* ================================================================
 * Reader: opens IPC handle, reads buffer, reports hits
 * ================================================================ */
static void run_reader(SharedState *sh) {
    /* Wait for writer to signal */
    while (!sh->writer_done) usleep(1000);

    cudaDeviceReset();
    CHECK(cudaSetDevice(1));
    float *d_remote;
    CHECK(cudaIpcOpenMemHandle((void**)&d_remote, sh->handle,
                               cudaIpcMemLazyEnablePeerAccess));

    float *h = (float *)malloc(N * sizeof(float));
    CHECK(cudaMemcpy(h, d_remote, N * sizeof(float), cudaMemcpyDeviceToHost));

    sh->result_hits = count_match(h, SECRET, N);
    sh->result_n    = N;
    free(h);

    CHECK(cudaIpcCloseMemHandle(d_remote));
    sh->reader_done = 1;
}

/* ================================================================
 * Run one test: fork writer and reader processes
 * ================================================================ */
static void run_test(int test_id, const char *test_name, int pass) {
    printf("[%s] pass=%d\n", test_name, pass);

    SharedState *sh = (SharedState *)mmap(NULL, sizeof(SharedState),
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (sh == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(sh, 0, sizeof(SharedState));

    fflush(stdout);
    pid_t writer_pid = fork();
    if (writer_pid == 0) {
        run_writer(sh, test_id);
        _exit(0);
    }

    fflush(stdout);
    pid_t reader_pid = fork();
    if (reader_pid == 0) {
        run_reader(sh);
        _exit(0);
    }

    int ws, rs;
    waitpid(writer_pid, &ws, 0);
    waitpid(reader_pid, &rs, 0);

    int hits = sh->result_hits;
    int n    = sh->result_n;
    printf("  Reader saw SECRET: %d/%d", hits, n);
    if (hits > n / 4)
        printf("  [!!!] CUDA IPC LEAK — cross-process GPU memory not isolated!\n");
    else if (hits > 0)
        printf("  [~] PARTIAL (some pages visible)\n");
    else
        printf("  SAFE — IPC reader sees zeroed/clean memory\n");

    munmap(sh, sizeof(SharedState));
}

int main() {
    /* Print device info without initializing CUDA context (avoids fork issues) */
    printf("=== Experiment 29: CUDA IPC Cross-Process GPU Memory Isolation ===\n");
    printf("    Device: H100 80GB HBM3  (child processes initialize CUDA independently)\n\n");

    for (int pass = 1; pass <= 5; pass++) {
        printf("========== PASS %d / 5 ==========\n", pass);

        run_test(0, "Test A: Direct share (write SECRET then export handle)", pass);
        run_test(1, "Test B: Pool residue via IPC (alloc-write-free-realloc-export)", pass);
        run_test(2, "Test C: Lazy-open (export empty, writer fills after signal)", pass);

        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("A: Direct share — A writes SECRET, exports handle, B reads\n");
    printf("B: Pool residue — A allocs pool block with SECRET, re-allocs same, B reads via IPC\n");
    printf("C: Lazy-open — A exports empty handle, A writes SECRET after, B opens later\n");
    printf("\n[Done]\n");
    return 0;
}
