/*
 * Experiment 18: CUDA Virtual Memory Management (VMM) Direct Test
 *
 * llama.cpp uses cuMemCreate / cuMemMap (VMM API) on modern GPUs (H100).
 * This is the ggml_cuda_pool_vmm implementation.
 *
 * VMM pool behavior (from ggml-cuda.cu):
 *   alloc(): ptr = pool_base + pool_used; pool_used += size; return ptr
 *   free():  pool_used -= size;  // just rewind — no zeroing!
 *
 * Questions:
 *   A. Does cuMemCreate zero new physical pages? (initial allocation)
 *   B. Does reusing a VMM range (free + alloc = unwind + advance) leak?
 *   C. Does cuMemUnmap + cuMemMap (physical remap) zero memory?
 *   D. cuMemCreate with different allocation types — pinned vs managed?
 *
 * This directly tests the allocator that llama-server uses on H100.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CU_CHECK(call)                                                       \
    do {                                                                     \
        CUresult _r = (call);                                                \
        if (_r != CUDA_SUCCESS) {                                            \
            const char *s = "?";                                             \
            cuGetErrorString(_r, &s);                                        \
            fprintf(stderr, "[CU] %s:%d  %s\n", __FILE__, __LINE__, s);     \
            exit(1);                                                         \
        }                                                                    \
    } while(0)

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[RT] %s:%d  %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while(0)

__global__ void fill_val(float *buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void count_match(const float *buf, int *cnt, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && __builtin_fabsf(buf[i] - val) < 1e-3f)
        atomicAdd(cnt, 1);
}

/* Get VMM granularity for device */
size_t get_granularity(int device) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;
    size_t gran = 0;
    cuMemGetAllocationGranularity(&gran, &prop,
                                   CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    return gran;
}

/* Allocate physical memory + map to virtual address */
static void vmm_alloc(CUdeviceptr *va, CUmemGenericAllocationHandle *handle,
                      size_t size, int device) {
    size_t gran = get_granularity(device);
    size_t rounded = gran * ((size + gran - 1) / gran);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;

    CU_CHECK(cuMemCreate(handle, rounded, &prop, 0));
    CU_CHECK(cuMemAddressReserve(va, rounded, 0, 0, 0));
    CU_CHECK(cuMemMap(*va, rounded, 0, *handle, 0));

    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id   = device;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(*va, rounded, &acc, 1));
}

static void vmm_free(CUdeviceptr va, CUmemGenericAllocationHandle handle,
                     size_t size) {
    size_t gran = get_granularity(0);
    size_t rounded = gran * ((size + gran - 1) / gran);
    CU_CHECK(cuMemUnmap(va, rounded));
    CU_CHECK(cuMemRelease(handle));
    CU_CHECK(cuMemAddressFree(va, rounded));
}

int main() {
    CU_CHECK(cuInit(0));
    int device = 0;
    CHECK(cudaSetDevice(device));

    printf("=== VMM Direct Test (cuMemCreate / cuMemMap) ===\n");

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    printf("    Device : %s\n", prop.name);

    size_t gran = get_granularity(device);
    printf("    VMM granularity: %zu KB\n\n", gran / 1024);

    const int N  = (int)(gran / sizeof(float));  /* exactly one granule */
    const float SEC_A = 2.71828f;
    const float SEC_B = 3.14159f;

    float *h = (float*)malloc(N * sizeof(float));
    if (!h) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* ------------------------------------------------------- */
    printf("[Part A] Initial cuMemCreate — does driver zero new pages?\n");
    {
        CUdeviceptr va;
        CUmemGenericAllocationHandle h_alloc;
        vmm_alloc(&va, &h_alloc, N * sizeof(float), device);

        CHECK(cudaMemcpy(h, (void*)va, N * sizeof(float), cudaMemcpyDeviceToHost));
        int nonzero = 0;
        for (int i = 0; i < N; i++) if (h[i] != 0.0f) nonzero++;

        printf("  Non-zero on fresh allocation : %d/%d\n", nonzero, N);
        printf("  Verdict : %s\n\n",
               nonzero == 0 ? "SAFE — cuMemCreate zeroes new physical pages" :
                              "LEAK — physical pages not zeroed on first map");
        vmm_free(va, h_alloc, N * sizeof(float));
    }

    /* ------------------------------------------------------- */
    printf("[Part B] VMM bump-alloc reuse (ggml_cuda_pool_vmm behavior)\n");
    printf("  Simulates: pool_used=0 → alloc → fill → pool_used=0 → alloc again\n");
    {
        /* Map a 2-slot region: [slot0][slot1] */
        size_t slot_size = (size_t)gran;
        size_t total     = slot_size * 2;

        CUdeviceptr va_pool;
        CUmemGenericAllocationHandle h_alloc;
        vmm_alloc(&va_pool, &h_alloc, total, device);

        /* Slot 0: victim fills with secret */
        float *slot0 = (float*)(uintptr_t)va_pool;
        fill_val<<<(N+255)/256, 256>>>(slot0, SEC_A, N);
        CHECK(cudaDeviceSynchronize());
        printf("  Slot0 ptr : %p  (filled with %.5f)\n", slot0, SEC_A);

        /* ggml free: just rewind pool_used → slot0 is "free"
           Next alloc gets slot0 again (same physical memory, not zeroed) */

        /* Slot 0 re-used as attacker */
        float *attacker = slot0;   /* same address — pool rewind */

        CHECK(cudaMemcpy(h, attacker, N * sizeof(float), cudaMemcpyDeviceToHost));
        int matches = 0;
        for (int i = 0; i < N; i++)
            if (__builtin_fabsf(h[i] - SEC_A) < 1e-3f) matches++;

        printf("  Attacker ptr : %p  (same: YES — pool rewind)\n", attacker);
        printf("  Matches   : %d/%d\n", matches, N);
        printf("  Verdict   : %s\n\n",
               matches == N ? "[!!!] FULL LEAK — ggml VMM pool does NOT zero on reuse" :
               matches == 0 ? "SAFE (zeroed)" : "PARTIAL");

        vmm_free(va_pool, h_alloc, total);
    }

    /* ------------------------------------------------------- */
    printf("[Part C] cuMemUnmap + cuMemMap to same VA — does remap zero?\n");
    printf("  (Maps fresh physical pages to the same virtual address)\n");
    {
        size_t slot_size = (size_t)gran;
        CUdeviceptr va;
        CUmemGenericAllocationHandle h1, h2;

        /* Create and map first physical block */
        {
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id   = device;
            CU_CHECK(cuMemCreate(&h1, slot_size, &prop, 0));
            CU_CHECK(cuMemCreate(&h2, slot_size, &prop, 0));
        }
        CU_CHECK(cuMemAddressReserve(&va, slot_size, 0, 0, 0));
        CU_CHECK(cuMemMap(va, slot_size, 0, h1, 0));

        CUmemAccessDesc acc = {};
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id   = device;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(va, slot_size, &acc, 1));

        /* Fill physical block h1 with secret */
        fill_val<<<(N+255)/256, 256>>>((float*)(uintptr_t)va, SEC_B, N);
        CHECK(cudaDeviceSynchronize());
        printf("  VA %p: filled h1 physical with %.5f\n",
               (void*)(uintptr_t)va, SEC_B);

        /* Unmap h1, map h2 (fresh physical) to same VA */
        CU_CHECK(cuMemUnmap(va, slot_size));
        CU_CHECK(cuMemMap(va, slot_size, 0, h2, 0));
        CU_CHECK(cuMemSetAccess(va, slot_size, &acc, 1));

        CHECK(cudaMemcpy(h, (void*)(uintptr_t)va, N * sizeof(float),
                         cudaMemcpyDeviceToHost));
        int nonzero = 0;
        for (int i = 0; i < N; i++) if (h[i] != 0.0f) nonzero++;

        printf("  After remap to fresh h2 physical:\n");
        printf("  Non-zero : %d/%d\n", nonzero, N);
        printf("  Verdict  : %s\n\n",
               nonzero == 0 ? "SAFE — fresh physical pages zeroed on new mapping" :
                              "LEAK — fresh physical pages contain old data");

        CU_CHECK(cuMemUnmap(va, slot_size));
        CU_CHECK(cuMemAddressFree(va, slot_size));
        CU_CHECK(cuMemRelease(h1));
        CU_CHECK(cuMemRelease(h2));
    }

    /* ------------------------------------------------------- */
    printf("[Part D] cuMemCreate → re-map SAME physical handle to new VA\n");
    printf("  (Reuse physical block under a new virtual address)\n");
    {
        size_t slot_size = (size_t)gran;
        CUdeviceptr va1, va2;
        CUmemGenericAllocationHandle phys;

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = device;
        CU_CHECK(cuMemCreate(&phys, slot_size, &prop, 0));

        CUmemAccessDesc acc = {};
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id   = device;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        /* Map to va1, fill with secret */
        CU_CHECK(cuMemAddressReserve(&va1, slot_size, 0, 0, 0));
        CU_CHECK(cuMemMap(va1, slot_size, 0, phys, 0));
        CU_CHECK(cuMemSetAccess(va1, slot_size, &acc, 1));
        fill_val<<<(N+255)/256,256>>>((float*)(uintptr_t)va1, SEC_A, N);
        CHECK(cudaDeviceSynchronize());
        printf("  va1 %p filled with %.5f\n", (void*)(uintptr_t)va1, SEC_A);

        /* Unmap from va1 */
        CU_CHECK(cuMemUnmap(va1, slot_size));
        CU_CHECK(cuMemAddressFree(va1, slot_size));

        /* Map same physical handle to va2 — SAME physical memory */
        CU_CHECK(cuMemAddressReserve(&va2, slot_size, 0, 0, 0));
        CU_CHECK(cuMemMap(va2, slot_size, 0, phys, 0));
        CU_CHECK(cuMemSetAccess(va2, slot_size, &acc, 1));

        CHECK(cudaMemcpy(h, (void*)(uintptr_t)va2, N * sizeof(float),
                         cudaMemcpyDeviceToHost));
        int matches = 0;
        for (int i = 0; i < N; i++)
            if (__builtin_fabsf(h[i] - SEC_A) < 1e-3f) matches++;

        printf("  va2 %p (same physical, new VA):\n", (void*)(uintptr_t)va2);
        printf("  Matches  : %d/%d\n", matches, N);
        printf("  Verdict  : %s\n",
               matches == N ? "[!!!] LEAK — physical reuse = data retention" :
               matches == 0 ? "SAFE — physical reuse zeroed" : "PARTIAL");

        CU_CHECK(cuMemUnmap(va2, slot_size));
        CU_CHECK(cuMemAddressFree(va2, slot_size));
        CU_CHECK(cuMemRelease(phys));
    }

    free(h);
    printf("\n[Done]\n");
    return 0;
}
