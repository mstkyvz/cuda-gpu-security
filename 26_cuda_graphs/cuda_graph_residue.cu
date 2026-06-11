/*
 * Experiment 26: CUDA Graphs — Intermediate Buffer Residue Between Replays
 *
 * CUDA Graphs (cudaGraph_t) capture a sequence of kernels and replay them
 * without CPU overhead. Used by PyTorch (torch.cuda.CUDAGraph) for inference
 * speedup. Key question: when a graph is replayed, are intermediate buffers
 * (pre-graph pool allocations, graph-owned allocations) zeroed?
 *
 * Tests:
 *   A. External buffer — graph writes SECRET, replay, does data persist?
 *   B. Graph-owned pool alloc — after graph destroy, does pool contain SECRET?
 *   C. Pre-graph pool residue — alloc inside capture gets pre-graph SECRET data?
 *   D. __shared__ after graph kernel — does graph kernel leave __shared__ residue?
 *
 * Compile:
 *   nvcc -O2 -arch=sm_90 -o cuda_graph_residue cuda_graph_residue.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[RT] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

#define N (256 * 1024)
#define SECRET1 3.14159f
#define SECRET2 2.71828f

/* ---- Kernels ---- */

__global__ void write_secret(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void read_buf(const float *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

/* Fill __shared__ with secret, then read it back via global flag to prevent DCE */
#define SHM_N 12288
__global__ void fill_shared(float secret, int *flag) {
    __shared__ float sm[SHM_N];
    int tid = threadIdx.x, stride = blockDim.x;
    for (int i = tid; i < SHM_N; i += stride)
        sm[i] = secret;
    __syncthreads();
    if (tid == 0 && sm[0] != 0.0f) atomicAdd(flag, 1);
}

__global__ void read_shared(float *out, int n) {
    __shared__ float sm[SHM_N];
    int tid = threadIdx.x, stride = blockDim.x;
    __syncthreads();
    for (int i = tid; i < n && i < SHM_N; i += stride)
        out[i] = sm[i];
}

static int count_match(const float *buf, float val, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (fabsf(buf[i] - val) < 1e-3f) c++;
    return c;
}

/* ================================================================
 * Test A: External buffer — graph writes SECRET, multiple replays.
 *   Verifies that graphs correctly re-execute kernels on replay.
 *   Then checks: after graph destroy + pool alloc, is SECRET visible?
 * ================================================================ */
static void test_A(int pass) {
    printf("\n[Test A] External buffer in graph — replay correctness + post-destroy residue (pass=%d)\n", pass);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    float *d_work, *d_readout;
    CHECK(cudaMalloc(&d_work, N * sizeof(float)));
    CHECK(cudaMalloc(&d_readout, N * sizeof(float)));
    CHECK(cudaMemset(d_work, 0, N * sizeof(float)));

    /* Capture graph: write_secret to d_work */
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    int blocks = (N + 255) / 256;
    write_secret<<<blocks, 256, 0, stream>>>(d_work, SECRET1, N);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    /* Replay 1 */
    CHECK(cudaGraphLaunch(graphExec, stream));
    CHECK(cudaStreamSynchronize(stream));
    float *h = (float *)malloc(N * sizeof(float));
    CHECK(cudaMemcpy(h, d_work, N * sizeof(float), cudaMemcpyDeviceToHost));
    int m1 = count_match(h, SECRET1, N);
    printf("  Replay 1: d_work SECRET1 hits: %d/%d\n", m1, N);

    /* Replay 2 */
    CHECK(cudaMemset(d_work, 0, N * sizeof(float)));
    CHECK(cudaGraphLaunch(graphExec, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(h, d_work, N * sizeof(float), cudaMemcpyDeviceToHost));
    int m2 = count_match(h, SECRET1, N);
    printf("  Replay 2: d_work SECRET1 hits: %d/%d\n", m2, N);

    /* Destroy graph, then pool alloc */
    CHECK(cudaGraphExecDestroy(graphExec));
    CHECK(cudaGraphDestroy(graph));

    float *d_post;
    CHECK(cudaMallocAsync(&d_post, N * sizeof(float), stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(h, d_post, N * sizeof(float), cudaMemcpyDeviceToHost));
    int m_post = count_match(h, SECRET1, N);
    int nz_post = 0; for (int i = 0; i < N; i++) if (h[i] != 0.0f) nz_post++;
    printf("  Post-destroy pool alloc: SECRET1=%d/%d  nz=%d/%d\n", m_post, N, nz_post, N);

    CHECK(cudaFreeAsync(d_post, stream));
    CHECK(cudaStreamSynchronize(stream));
    free(h);
    CHECK(cudaFree(d_work));
    CHECK(cudaFree(d_readout));
    CHECK(cudaStreamDestroy(stream));
}

/* ================================================================
 * Test B: Graph-owned allocation (cudaMallocAsync INSIDE capture)
 *   Graph allocates d_internal, writes SECRET1, frees inside capture.
 *   After graph destroy, pool alloc — does it contain SECRET1?
 * ================================================================ */
static void test_B(int pass) {
    printf("\n[Test B] Graph-owned pool alloc (cudaMallocAsync inside capture) (pass=%d)\n", pass);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    float *d_internal = NULL;
    float *d_capture_result;
    CHECK(cudaMalloc(&d_capture_result, N * sizeof(float)));

    int blocks = (N + 255) / 256;
    cudaGraph_t g1;
    cudaGraphExec_t ge1;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaMallocAsync(&d_internal, N * sizeof(float), stream));
    write_secret<<<blocks, 256, 0, stream>>>(d_internal, SECRET1, N);
    read_buf<<<blocks, 256, 0, stream>>>(d_internal, d_capture_result, N);
    CHECK(cudaFreeAsync(d_internal, stream));
    CHECK(cudaStreamEndCapture(stream, &g1));
    CHECK(cudaGraphInstantiate(&ge1, g1, NULL, NULL, 0));

    /* Run graph 1 once */
    CHECK(cudaGraphLaunch(ge1, stream));
    CHECK(cudaStreamSynchronize(stream));

    float *h = (float *)malloc(N * sizeof(float));
    CHECK(cudaMemcpy(h, d_capture_result, N * sizeof(float), cudaMemcpyDeviceToHost));
    int m = count_match(h, SECRET1, N);
    printf("  Graph ran: capture_result SECRET1=%d/%d (sanity)\n", m, N);

    /* Destroy graph */
    CHECK(cudaGraphExecDestroy(ge1));
    CHECK(cudaGraphDestroy(g1));

    /* Pool alloc after destroy */
    float *d_after;
    CHECK(cudaMallocAsync(&d_after, N * sizeof(float), stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(h, d_after, N * sizeof(float), cudaMemcpyDeviceToHost));
    int m_after = count_match(h, SECRET1, N);
    int nz = 0; for (int i = 0; i < N; i++) if (h[i] != 0.0f) nz++;
    printf("  Post-destroy pool alloc: SECRET1=%d/%d  nz=%d/%d\n", m_after, N, nz, N);
    if (m_after > N / 4)
        printf("  [!!!] GRAPH-OWNED POOL RESIDUE — post-destroy alloc contains SECRET1!\n");
    else
        printf("  SAFE — post-destroy alloc does not contain graph's data\n");

    CHECK(cudaFreeAsync(d_after, stream));
    CHECK(cudaStreamSynchronize(stream));
    free(h);
    CHECK(cudaFree(d_capture_result));
    CHECK(cudaStreamDestroy(stream));
}

/* ================================================================
 * Test C: Pre-graph pool residue — alloc inside capture gets SECRET data?
 *   Phase 1: alloc from pool, write SECRET1, free to pool (pre-graph).
 *   Phase 2: capture graph that allocs from same pool — does it get SECRET1?
 *   This mimics PyTorch: warmup run fills pool, then graph capture replays.
 * ================================================================ */
static void test_C(int pass) {
    printf("\n[Test C] Pre-graph pool residue — graph alloc gets prior SECRET data? (pass=%d)\n", pass);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int blocks = (N + 255) / 256;
    float *h = (float *)malloc(N * sizeof(float));

    /* Phase 1: fill pool with SECRET1, then free */
    float *d_pre;
    CHECK(cudaMallocAsync(&d_pre, N * sizeof(float), stream));
    write_secret<<<blocks, 256, 0, stream>>>(d_pre, SECRET1, N);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaFreeAsync(d_pre, stream));
    CHECK(cudaStreamSynchronize(stream));
    printf("  Phase 1: SECRET1 written and freed to stream pool\n");

    /* Phase 2: capture graph, alloc inside capture, copy to output */
    float *d_in_graph = NULL;
    float *d_graph_out;
    CHECK(cudaMalloc(&d_graph_out, N * sizeof(float)));

    cudaGraph_t g;
    cudaGraphExec_t ge;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaMallocAsync(&d_in_graph, N * sizeof(float), stream));
    read_buf<<<blocks, 256, 0, stream>>>(d_in_graph, d_graph_out, N);
    CHECK(cudaFreeAsync(d_in_graph, stream));
    CHECK(cudaStreamEndCapture(stream, &g));
    CHECK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));

    /* Launch graph — does d_in_graph get the SECRET1 pool block? */
    CHECK(cudaGraphLaunch(ge, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaMemcpy(h, d_graph_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    int m = count_match(h, SECRET1, N);
    int nz = 0; for (int i = 0; i < N; i++) if (h[i] != 0.0f) nz++;
    printf("  Graph alloc (from pool): SECRET1=%d/%d  nz=%d/%d\n", m, N, nz, N);
    if (m > N / 4)
        printf("  [!!!] CUDA GRAPH POOL RESIDUE — graph alloc contains pre-graph SECRET1!\n");
    else
        printf("  SAFE — graph alloc does not contain pre-graph SECRET1\n");

    /* Replay same graph again — same pool block reused? */
    CHECK(cudaGraphLaunch(ge, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(h, d_graph_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    int m2 = count_match(h, SECRET1, N);
    printf("  Graph replay 2: SECRET1=%d/%d (same block reused in graph)\n", m2, N);

    CHECK(cudaGraphExecDestroy(ge));
    CHECK(cudaGraphDestroy(g));
    free(h);
    CHECK(cudaFree(d_graph_out));
    CHECK(cudaStreamDestroy(stream));
}

/* ================================================================
 * Test D: __shared__ residue from a graph kernel
 *   Graph kernel fills __shared__ with SECRET1.
 *   After graph launch, attacker kernel reads __shared__ (no init).
 *   Builds on Exp 23: confirms __shared__ residue persists even
 *   when the writer kernel was inside a CUDA Graph.
 * ================================================================ */
static void test_D(int pass) {
    printf("\n[Test D] __shared__ residue from graph kernel (pass=%d)\n", pass);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int *d_flag;
    float *d_shm_out;
    CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CHECK(cudaMalloc(&d_shm_out, SHM_N * sizeof(float)));

    /* Capture graph: fill __shared__ on all SMs */
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaMemsetAsync(d_flag, 0, sizeof(int), stream));
    fill_shared<<<512, 256, 0, stream>>>(SECRET1, d_flag);
    CHECK(cudaStreamEndCapture(stream, &g));
    CHECK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));

    /* Launch graph — fills __shared__ across all 132 SMs */
    CHECK(cudaGraphLaunch(ge, stream));
    CHECK(cudaStreamSynchronize(stream));

    int h_flag = 0;
    CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Graph filled __shared__ (flag=%d)\n", h_flag);

    /* Attacker: read uninitialized __shared__ immediately after graph */
    read_shared<<<512, 256>>>(d_shm_out, SHM_N);
    CHECK(cudaDeviceSynchronize());

    float *h = (float *)malloc(SHM_N * sizeof(float));
    CHECK(cudaMemcpy(h, d_shm_out, SHM_N * sizeof(float), cudaMemcpyDeviceToHost));
    int m = count_match(h, SECRET1, SHM_N);
    int nz = 0; for (int i = 0; i < SHM_N; i++) if (h[i] != 0.0f) nz++;
    printf("  Attacker after graph: __shared__ SECRET1=%d/%d  nz=%d/%d\n",
           m, SHM_N, nz, SHM_N);
    if (m == SHM_N)
        printf("  [!!!] GRAPH KERNEL __shared__ LEAK — attacker sees graph kernel's shared mem!\n");
    else if (m > SHM_N / 4)
        printf("  [~] PARTIAL graph kernel __shared__ leak\n");
    else
        printf("  SAFE — no __shared__ residue from graph kernel\n");

    CHECK(cudaGraphExecDestroy(ge));
    CHECK(cudaGraphDestroy(g));
    free(h);
    CHECK(cudaFree(d_flag));
    CHECK(cudaFree(d_shm_out));
    CHECK(cudaStreamDestroy(stream));
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Experiment 26: CUDA Graphs Intermediate Buffer Residue ===\n");
    printf("    Device: %s  (SMs: %d, compute %d.%d)\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    for (int pass = 0; pass < 4; pass++) {
        printf("\n========== PASS %d / 4 ==========\n", pass + 1);
        test_A(pass + 1);
        test_B(pass + 1);
        test_C(pass + 1);
        test_D(pass + 1);
    }

    printf("\n=== Summary ===\n");
    printf("A: External buffer — graph writes SECRET, does post-destroy pool contain it?\n");
    printf("B: Graph-owned alloc — after graph destroy, does pool alloc get SECRET?\n");
    printf("C: Pre-graph pool residue — graph alloc inside capture gets prior SECRET?\n");
    printf("D: __shared__ residue — graph kernel leaves __shared__ data for attacker?\n");
    printf("\n[Done]\n");
    return 0;
}
