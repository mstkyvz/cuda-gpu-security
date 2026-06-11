/*
 * Experiment 15: CUDA IPC Memory Handle Leak
 *
 * CUDA IPC (Inter-Process Communication) allows one process to share a
 * GPU memory allocation with another process via a handle. PyTorch uses
 * this for torch.multiprocessing shared tensors, and NCCL uses it for
 * multi-process GPU communication.
 *
 * Question: When process A creates an IPC handle to its tensor and
 * another process opens that handle, does the opener see the data?
 *
 * More importantly for security:
 *   - If a server exports IPC handles for efficiency (worker pool pattern),
 *     can a receiving process read memory that wasn't meant for it?
 *   - What happens if A allocates, fills, exports handle, then FREES
 *     the allocation while B holds the handle open?
 *
 * This test:
 *   Part A: Write to GPU, export IPC handle → read via IPC in SAME process
 *   Part B: Two-process test via fork() — child sees parent's GPU data
 *   Part C: IPC + pool reuse — does IPC handle survive cudaFree + realloc?
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA] %s:%d %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while(0)

__global__ void fill_secret(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void check_nonzero(const float *buf, int *cnt, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fabsf(buf[i] - val) < 0.01f)
        atomicAdd(cnt, 1);
}

/* Shared memory between parent and child via mmap */
typedef struct {
    cudaIpcMemHandle_t handle;
    float              secret_value;
    int                n;
    volatile int       parent_done;    /* parent signals child to start */
    volatile int       child_done;
} shared_ipc_t;

int main() {
    const int N = 64 * 1024;  /* 256 KB float */
    printf("=== CUDA IPC Memory Handle Leak Test ===\n");
    printf("    N = %d floats (%zu KB)\n\n", N, N*sizeof(float)/1024);

    /* -------------------------------------------------------
     * Part A: Same-process IPC round-trip
     * Open IPC handle within the same process → trivially works
     * but verifies the handle mechanism
     * ------------------------------------------------------- */
    printf("[Part A] Same-process IPC handle round-trip\n");
    {
        float *d_a = NULL, *d_reopen = NULL;
        cudaIpcMemHandle_t h;
        float h_out[8] = {0};

        CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        fill_secret<<<(N+255)/256, 256>>>(d_a, 2.71828f, N);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaIpcGetMemHandle(&h, d_a));
        CHECK(cudaIpcOpenMemHandle((void**)&d_reopen, h,
                                   cudaIpcMemLazyEnablePeerAccess));
        CHECK(cudaMemcpy(h_out, d_reopen, 8 * sizeof(float),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaDeviceSynchronize());

        int match = 1;
        for (int i = 0; i < 8; i++)
            if (fabsf(h_out[i] - 2.71828f) > 0.01f) match = 0;

        printf("  Writer ptr   : %p\n", (void*)d_a);
        printf("  IPC reopen   : %p\n", (void*)d_reopen);
        printf("  Values[0:4]  : %.5f %.5f %.5f %.5f\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);
        printf("  Match secret : %s\n\n", match ? "YES - data accessible via IPC" : "NO");

        CHECK(cudaIpcCloseMemHandle(d_reopen));
        CHECK(cudaFree(d_a));
    }

    /* -------------------------------------------------------
     * Part B: Cross-process IPC via fork()
     * Parent: allocate, fill with secret, export handle, wait
     * Child:  open handle, read data, report
     * ------------------------------------------------------- */
    printf("[Part B] Cross-process IPC (fork) - can child read parent's GPU data?\n");
    {
        /* Allocate shared memory region for IPC handle exchange */
        shared_ipc_t *shm = (shared_ipc_t*)mmap(NULL, sizeof(shared_ipc_t),
                                                  PROT_READ | PROT_WRITE,
                                                  MAP_SHARED | MAP_ANONYMOUS,
                                                  -1, 0);
        if (shm == MAP_FAILED) { perror("mmap"); exit(1); }

        shm->secret_value = 3.14159f;
        shm->n = N;
        shm->parent_done = 0;
        shm->child_done  = 0;

        float *d_parent = NULL;
        CHECK(cudaMalloc(&d_parent, N * sizeof(float)));
        fill_secret<<<(N+255)/256, 256>>>(d_parent, shm->secret_value, N);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaIpcGetMemHandle(&shm->handle, d_parent));

        printf("  Parent GPU ptr : %p  secret=%.5f\n",
               (void*)d_parent, shm->secret_value);

        pid_t pid = fork();
        if (pid == 0) {
            /* --- CHILD --- */
            float *d_child = NULL;
            cudaError_t err = cudaIpcOpenMemHandle((void**)&d_child,
                                                    shm->handle,
                                                    cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                fprintf(stderr, "  [child] IpcOpenMemHandle failed: %s\n",
                        cudaGetErrorString(err));
                shm->child_done = 1;
                exit(1);
            }

            float h_child[8] = {0};
            CHECK(cudaMemcpy(h_child, d_child, 8 * sizeof(float),
                             cudaMemcpyDeviceToHost));
            CHECK(cudaDeviceSynchronize());

            int matches = 0;
            for (int i = 0; i < 8; i++)
                if (fabsf(h_child[i] - shm->secret_value) < 0.01f) matches++;

            printf("  Child IPC ptr  : %p\n", (void*)d_child);
            printf("  Child read [0:4]: %.5f %.5f %.5f %.5f\n",
                   h_child[0], h_child[1], h_child[2], h_child[3]);
            printf("  Matches (8/8)  : %d/8\n", matches);
            if (matches == 8)
                printf("  [!!!] CROSS-PROCESS IPC READ CONFIRMED\n");

            CHECK(cudaIpcCloseMemHandle(d_child));
            shm->child_done = 1;
            exit(0);
        } else {
            /* --- PARENT waits --- */
            int status;
            waitpid(pid, &status, 0);
            CHECK(cudaFree(d_parent));
            munmap(shm, sizeof(shared_ipc_t));
        }
        printf("\n");
    }

    /* -------------------------------------------------------
     * Part C: IPC handle + cudaFree + pool realloc
     * Does the handle remain valid after the original allocation is freed?
     * This tests if attacker can hold a handle open while victim frees.
     * ------------------------------------------------------- */
    printf("[Part C] IPC handle durability after cudaFree + pool realloc\n");
    {
        float *d_orig = NULL, *d_reopen = NULL, *d_realloc = NULL;
        cudaIpcMemHandle_t h;
        float h_out[4] = {0};

        CHECK(cudaMalloc(&d_orig, N * sizeof(float)));
        fill_secret<<<(N+255)/256, 256>>>(d_orig, 1.41421f, N);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaIpcGetMemHandle(&h, d_orig));

        /* Open handle BEFORE free */
        cudaError_t err = cudaIpcOpenMemHandle((void**)&d_reopen, h,
                                               cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            printf("  IpcOpenMemHandle: %s\n", cudaGetErrorString(err));
        } else {
            /* Read before free */
            CHECK(cudaMemcpy(h_out, d_reopen, 4*sizeof(float),
                             cudaMemcpyDeviceToHost));
            printf("  Pre-free values : %.5f %.5f %.5f %.5f (secret=1.41421)\n",
                   h_out[0], h_out[1], h_out[2], h_out[3]);

            /* Now free original allocation */
            CHECK(cudaFree(d_orig));

            /* Reallocate from pool (could get same block) */
            CHECK(cudaMalloc(&d_realloc, N * sizeof(float)));
            fill_secret<<<(N+255)/256, 256>>>(d_realloc, 9.99999f, N);
            CHECK(cudaDeviceSynchronize());

            /* Try to read via old IPC handle — should it reflect new data? */
            memset(h_out, 0, sizeof(h_out));
            err = cudaMemcpy(h_out, d_reopen, 4*sizeof(float),
                             cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                printf("  Post-free+realloc IPC read: %.5f %.5f %.5f %.5f\n",
                       h_out[0], h_out[1], h_out[2], h_out[3]);
                if (fabsf(h_out[0] - 9.99999f) < 0.01f)
                    printf("  [!!!] IPC handle points to reallocated block (new data visible)\n");
                else if (fabsf(h_out[0] - 1.41421f) < 0.01f)
                    printf("  [!] IPC handle still shows freed data\n");
                else
                    printf("  [?] IPC handle shows unexpected data\n");
            } else {
                printf("  Post-free read failed: %s\n", cudaGetErrorString(err));
            }

            CHECK(cudaIpcCloseMemHandle(d_reopen));
            CHECK(cudaFree(d_realloc));
        }
    }

    printf("\n[Done]\n");
    return 0;
}
