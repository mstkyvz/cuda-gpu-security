/*
 * Experiment 30: cuDNN Workspace Residue
 *
 * cuDNN operations (convolutions, attention, normalization) require a temporary
 * "workspace" buffer allocated by the user and passed to the operation. This
 * mirrors the cuBLAS workspace pattern (Exp 24). Key question: if a workspace
 * buffer is filled with SECRET data, then freed to pool and re-allocated, does
 * the next cuDNN operation receive unzeroed workspace containing the SECRET?
 *
 * Tests use cuDNN v9 (cudnn.h, libcudnn.so.9):
 *   A. Direct residue: fill workspace with SECRET, run convolution, read workspace
 *      after operation (cuDNN does NOT zero the workspace after use)
 *   B. Same-stream pool residue: write SECRET to pool buffer, free, re-alloc
 *      (same pool block), pass to cuDNN as workspace — does cuDNN see SECRET?
 *   C. Cross-stream pool: free to pool, re-alloc on different stream — zeroed?
 *
 * Compile:
 *   nvcc -O2 -arch=sm_90 \
 *     -I/home/mustafa/tts-env/lib/python3.12/site-packages/nvidia/cudnn/include \
 *     -L/home/mustafa/tts-env/lib/python3.12/site-packages/nvidia/cudnn/lib \
 *     -lcudnn -o cudnn_workspace_residue cudnn_workspace_residue.cu \
 *     -Wl,-rpath,/home/mustafa/tts-env/lib/python3.12/site-packages/nvidia/cudnn/lib
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[CUDA] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

#define CDNN(call) do { cudnnStatus_t _e=(call); if(_e!=CUDNN_STATUS_SUCCESS){ \
    fprintf(stderr,"[cuDNN] %s:%d %s\n",__FILE__,__LINE__,cudnnGetErrorString(_e)); exit(1); } } while(0)

#define SECRET 3.14159f

/* Conv parameters: small enough to be fast, large enough for meaningful workspace */
#define N_IMG    4       /* batch size */
#define C_IN     64      /* input channels */
#define H_IN     28      /* height */
#define W_IN     28      /* width */
#define C_OUT    128     /* output channels */
#define K_H      3       /* kernel height */
#define K_W      3       /* kernel width */
#define H_OUT    26      /* = H_IN - K_H + 1 */
#define W_OUT    26      /* = W_IN - K_W + 1 */

static int count_near(const float *h, float v, float tol, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (fabsf(h[i] - v) < tol) c++;
    return c;
}

__global__ void fill_buf(float *p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

/* Create cuDNN convolution descriptor and query workspace size */
static size_t setup_conv(cudnnHandle_t hdl,
                         cudnnTensorDescriptor_t  *xDesc,
                         cudnnFilterDescriptor_t  *wDesc,
                         cudnnTensorDescriptor_t  *yDesc,
                         cudnnConvolutionDescriptor_t *convDesc,
                         cudnnConvolutionFwdAlgo_t *algo)
{
    CDNN(cudnnCreateTensorDescriptor(xDesc));
    CDNN(cudnnCreateFilterDescriptor(wDesc));
    CDNN(cudnnCreateTensorDescriptor(yDesc));
    CDNN(cudnnCreateConvolutionDescriptor(convDesc));

    CDNN(cudnnSetTensor4dDescriptor(*xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                    N_IMG, C_IN, H_IN, W_IN));
    CDNN(cudnnSetFilter4dDescriptor(*wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                    C_OUT, C_IN, K_H, K_W));
    CDNN(cudnnSetTensor4dDescriptor(*yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                    N_IMG, C_OUT, H_OUT, W_OUT));
    CDNN(cudnnSetConvolution2dDescriptor(*convDesc, 0, 0, 1, 1, 1, 1,
                                         CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    /* Use IMPLICIT_PRECOMP_GEMM — consistently requires workspace on NCHW layouts */
    *algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    size_t wsSize = 0;
    CDNN(cudnnGetConvolutionForwardWorkspaceSize(hdl, *xDesc, *wDesc, *convDesc, *yDesc,
                                                 *algo, &wsSize));
    return wsSize;
}

int main() {
    CHECK(cudaSetDevice(1));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 1));
    printf("=== Experiment 30: cuDNN Workspace Residue ===\n");
    printf("    Device: %s  compute=%d.%d\n", prop.name, prop.major, prop.minor);

    cudnnHandle_t hdl;
    CDNN(cudnnCreate(&hdl));

    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    size_t wsSize = setup_conv(hdl, &xDesc, &wDesc, &yDesc, &convDesc, &algo);

    printf("    Conv: N=%d C_in=%d H=%dx%d → C_out=%d H_out=%dx%d  filter=%dx%d\n",
           N_IMG, C_IN, H_IN, W_IN, C_OUT, H_OUT, W_OUT, K_H, K_W);
    printf("    Workspace size: %zu bytes (%.1f MB)\n\n", wsSize, (double)wsSize / (1024*1024));

    /* Allocate input/filter/output tensors (filled with ones for correctness) */
    size_t xSz = (size_t)N_IMG * C_IN  * H_IN  * W_IN  * sizeof(float);
    size_t wSz = (size_t)C_OUT * C_IN  * K_H   * K_W   * sizeof(float);
    size_t ySz = (size_t)N_IMG * C_OUT * H_OUT * W_OUT * sizeof(float);
    float *d_x, *d_w, *d_y;
    CHECK(cudaMalloc(&d_x, xSz)); CHECK(cudaMalloc(&d_w, wSz)); CHECK(cudaMalloc(&d_y, ySz));
    fill_buf<<<(xSz/4+255)/256, 256>>>(d_x, 1.0f, xSz/4);
    fill_buf<<<(wSz/4+255)/256, 256>>>(d_w, 0.01f, wSz/4);
    fill_buf<<<(ySz/4+255)/256, 256>>>(d_y, 0.0f, ySz/4);
    CHECK(cudaDeviceSynchronize());

    int ws_floats = (int)(wsSize / sizeof(float));
    float alpha = 1.0f, beta = 0.0f;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CDNN(cudnnSetStream(hdl, stream));

    for (int pass = 1; pass <= 5; pass++) {
        printf("========== PASS %d / 5 ==========\n", pass);

        /* ---- Test A: workspace content AFTER cuDNN operation ---- */
        printf("[Test A] Workspace content after cuDNN forward conv (pass=%d)\n", pass);
        {
            float *d_ws;
            CHECK(cudaMallocAsync(&d_ws, wsSize, stream));
            fill_buf<<<(ws_floats+255)/256, 256, 0, stream>>>(d_ws, SECRET, ws_floats);
            CHECK(cudaStreamSynchronize(stream));

            /* Run convolution with SECRET-filled workspace */
            CDNN(cudnnConvolutionForward(hdl, &alpha, xDesc, d_x, wDesc, d_w,
                                         convDesc, algo, d_ws, wsSize, &beta, yDesc, d_y));
            CHECK(cudaStreamSynchronize(stream));

            float *h_ws = (float *)malloc(wsSize);
            CHECK(cudaMemcpy(h_ws, d_ws, wsSize, cudaMemcpyDeviceToHost));
            int hits = count_near(h_ws, SECRET, 0.01f, ws_floats);
            int nz   = 0; for (int i = 0; i < ws_floats; i++) if (fabsf(h_ws[i]) > 0.001f) nz++;
            printf("  After conv (workspace pre-filled SECRET): SECRET=%d/%d  nz=%d/%d\n",
                   hits, ws_floats, nz, ws_floats);
            if (hits > ws_floats / 4)
                printf("  [NOTE] cuDNN did NOT overwrite full workspace (normal — ops only use what they need)\n");
            else
                printf("  [NOTE] cuDNN overwrote workspace (uses it for intermediate results)\n");
            free(h_ws);
            CHECK(cudaFreeAsync(d_ws, stream));
            CHECK(cudaStreamSynchronize(stream));
        }

        /* ---- Test B: same-stream pool residue (NO sync between free and alloc) ---- */
        printf("[Test B] Same-stream pool residue in cuDNN workspace NO-SYNC (pass=%d)\n", pass);
        {
            float *d_pre;
            CHECK(cudaMallocAsync(&d_pre, wsSize, stream));
            fill_buf<<<(ws_floats+255)/256, 256, 0, stream>>>(d_pre, SECRET, ws_floats);
            CHECK(cudaStreamSynchronize(stream));
            CHECK(cudaFreeAsync(d_pre, stream));
            /* NO sync between free and alloc — real-world pattern: pool reuse without gap */

            /* Re-alloc same pool block (should have SECRET from d_pre) */
            float *d_ws;
            CHECK(cudaMallocAsync(&d_ws, wsSize, stream));
            CHECK(cudaStreamSynchronize(stream));

            /* Read workspace content BEFORE cuDNN (verify pool residue) */
            float *h_pre = (float *)malloc(wsSize);
            CHECK(cudaMemcpy(h_pre, d_ws, wsSize, cudaMemcpyDeviceToHost));
            int pre_hits = count_near(h_pre, SECRET, 0.01f, ws_floats);
            free(h_pre);

            /* Run cuDNN with this residue-filled workspace */
            CDNN(cudnnConvolutionForward(hdl, &alpha, xDesc, d_x, wDesc, d_w,
                                         convDesc, algo, d_ws, wsSize, &beta, yDesc, d_y));
            CHECK(cudaStreamSynchronize(stream));

            /* Read after operation */
            float *h_ws = (float *)malloc(wsSize);
            CHECK(cudaMemcpy(h_ws, d_ws, wsSize, cudaMemcpyDeviceToHost));
            int post_hits = count_near(h_ws, SECRET, 0.01f, ws_floats);
            free(h_ws);

            printf("  Pool residue before conv: SECRET=%d/%d\n", pre_hits, ws_floats);
            printf("  Pool residue after conv:  SECRET=%d/%d\n", post_hits, ws_floats);
            if (pre_hits > ws_floats / 4)
                printf("  [!!!] CUDNN WORKSPACE POOL RESIDUE — workspace given to cuDNN contained SECRET!\n");
            else
                printf("  SAFE — pool block was zeroed before reaching cuDNN workspace\n");

            CHECK(cudaFreeAsync(d_ws, stream));
            CHECK(cudaStreamSynchronize(stream));
        }

        /* ---- Test C: cross-stream pool residue ---- */
        printf("[Test C] Cross-stream pool residue in cuDNN workspace (pass=%d)\n", pass);
        {
            cudaStream_t stream2;
            CHECK(cudaStreamCreate(&stream2));

            float *d_pre;
            CHECK(cudaMallocAsync(&d_pre, wsSize, stream));
            fill_buf<<<(ws_floats+255)/256, 256, 0, stream>>>(d_pre, SECRET, ws_floats);
            CHECK(cudaStreamSynchronize(stream));
            CHECK(cudaFreeAsync(d_pre, stream));
            CHECK(cudaStreamSynchronize(stream));

            /* Re-alloc on DIFFERENT stream */
            float *d_ws;
            CHECK(cudaMallocAsync(&d_ws, wsSize, stream2));
            CHECK(cudaStreamSynchronize(stream2));

            float *h_ws = (float *)malloc(wsSize);
            CHECK(cudaMemcpy(h_ws, d_ws, wsSize, cudaMemcpyDeviceToHost));
            int hits = count_near(h_ws, SECRET, 0.01f, ws_floats);
            free(h_ws);

            printf("  Cross-stream pool residue: SECRET=%d/%d", hits, ws_floats);
            if (hits > ws_floats / 4)
                printf("  [!!!] CROSS-STREAM RESIDUE IN CUDNN WORKSPACE\n");
            else
                printf("  SAFE — cross-stream allocation zeroed\n");

            CHECK(cudaFreeAsync(d_ws, stream2));
            CHECK(cudaStreamSynchronize(stream2));
            CHECK(cudaStreamDestroy(stream2));
        }
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("A: cuDNN workspace content after forward conv (does cuDNN clear workspace?)\n");
    printf("B: Same-stream pool residue as cuDNN workspace\n");
    printf("C: Cross-stream pool residue as cuDNN workspace\n");
    printf("\n[Done]\n");

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(hdl);
    CHECK(cudaFree(d_x)); CHECK(cudaFree(d_w)); CHECK(cudaFree(d_y));
    CHECK(cudaStreamDestroy(stream));
    return 0;
}
