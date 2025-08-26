/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-cann.h"

#include <acl/acl.h>
#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ggml-backend-impl.h"
#include "ggml-cann/aclnn_ops.h"
#include "ggml-cann/ascend_graph.h"
#include "ggml-cann/common.h"
#ifdef GGML_USE_HCCL
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#endif

#define GGML_COMMON_DECL_C

#include <cstdlib>  // Required for getenv

#include "ggml-common.h"

#define GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func,
                   file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

/**
 * @brief Sets the device to be used by CANN.
 *
 * @param device The device ID to set.
 */
void ggml_cann_set_device(const int32_t device) {
    // TODO: uncomment these lines after empty context has fixed.
    // int current_device;
    // ACL_CHECK(aclrtGetDevice(&current_device));

    // if (device == current_device) {
    //   return;
    // }
    ACL_CHECK(aclrtSetDevice(device));
}

/**
 * @brief Retrieves the current device ID.
 *
 * @return The current device ID.
 */
int32_t ggml_cann_get_device() {
    int32_t id;
    ACL_CHECK(aclrtGetDevice(&id));
    return id;
}

/**
 * @brief Initialize the CANN device information.
 *
 * This function initializes the CANN device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrtGetDeviceCount((uint32_t*)&info.device_count);

    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__,
                       aclGetRecentErrMsg());
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CANN_MAX_DEVICES);

    for (int id = 0; id < info.device_count; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = id;
        prop.reserve = 0;
        ACL_CHECK(aclrtMemGetAllocationGranularity(
            &prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            &info.devices[id].vmm_granularity));

        size_t free, total;
        ggml_backend_cann_get_device_memory(id, &free, &total);
        info.devices[id].total_vram = free;
    }

    // TODO: add more device info later.
    return info;
}

/**
 * @brief Retrieve the CANN device information.
 *
 * This function returns a reference to a structure containing the CANN device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A reference to the structure containing the device information.
 */
const ggml_cann_device_info& ggml_cann_info() {
    static ggml_cann_device_info info = ggml_cann_init();
    return info;
}

// #define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(legacy).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
struct ggml_cann_pool_leg : public ggml_cann_pool {
    /**
     * @brief The maximum number of buffers in the pool.
     */
    static const int MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_cann_buffer {
        void* ptr = nullptr;  ///< Pointer to the buffer memory.
        size_t size = 0;      ///< Size of the buffer.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_buffer buffer_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_leg(int device) : device(device) {}

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_leg() {
        ggml_cann_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
        const size_t alignment = 128;
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }
#ifdef DEBUG_CANN_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CANN_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void* ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cann_buffer& b = buffer_pool[ibest];
            void* ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void* ptr;
        ggml_cann_set_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        *actual_size = size;
        pool_size += size;
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, "
            "requested %u MB\n",
            __func__, device, nnz, (uint32_t)(max_size / 1024 / 1024),
            (uint32_t)(pool_size / 1024 / 1024),
            (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        // memory should always buffered. these memory may still needed by
        // tasks in stream.
        // TODO, fix me.
        GGML_ABORT("Cann buffer pool full, increase MAX_CANN_BUFFERS\n");
    }
};

/**
 * @brief A pool of CANN buffers with virtual memory.
 *
 * This class manages a pool of CANN buffers with virtual memory for a specific
 * device.
 */
struct ggml_cann_pool_vmm : public ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    void* pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_used = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Allocation granularity for the virtual memory pool.
     */
    size_t granularity;

    /**
     * @brief Handles for the physical memory allocated.
     */
    std::vector<aclrtDrvMemHandle> handles;

    /**
     * @brief Offsets for the mapped memory regions.
     */
    std::vector<void*> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_vmm(int device)
        : device(device),
          granularity(ggml_cann_info().devices[device].vmm_granularity) {
        auto dev = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size = dev.total_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        if (pool_addr != 0) {
            for (auto& offset : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(offset));
            }
            for (auto& handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrtReleaseMemAddress(pool_addr));
        }
    }

    /**
     * @brief Allocate a buffer of the given size in the virtual memory pool.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        const size_t alignment = 128;
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = GGML_PAD(reserve_size, granularity);

            GGML_ASSERT(pool_size + reserve_size <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr = ACL_HBM_MEM_HUGE;
            prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            prop.reserve = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                ACL_CHECK(
                    aclrtReserveMemAddress(&pool_addr, max_size, 0, NULL, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((char*)pool_addr + pool_size, reserve_size, 0,
                                  handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((char*)pool_addr + pool_size);

            // add to the pool
            pool_size += reserve_size;

#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                device, (unsigned long long)(pool_size / 1024 / 1024),
                (unsigned long long)(reserve_size / 1024 / 1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void* ptr = (void*)((char*)pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device,
                      (unsigned long long)size, (unsigned long long)ptr);
#endif
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device,
                      (unsigned long long)size, (unsigned long long)ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void*)((char*)pool_addr + pool_used));
    }
};

/**
 * @brief Create a new CANN pool for a specific device.
 *
 * Factory method to create a new CANN pool object based on the device type.
 *
 * @param device The device ID for which to create the pool.
 * @return A unique pointer to the created CANN pool.
 */
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(
    int device) {
    return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_vmm(device));
}

// cann buffer
/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
struct ggml_backend_cann_buffer_context {
    int32_t device;  ///< The device ID associated with this buffer context.
    void* dev_ptr =
        nullptr;  ///< Pointer to the device memory allocated for the buffer.

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_buffer_context(int32_t device, void* dev_ptr)
        : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_buffer_context() { ACL_CHECK(aclrtFree(dev_ptr)); }
};

/**
 * @brief Check if a buffer is a CANN buffer.
 *
 * This function checks if a given buffer is a CANN buffer by comparing its
 * `get_name` function pointer to `ggml_backend_cann_buffer_get_name`.
 *
 * @param buffer The buffer to check.
 * @return true if the buffer is a CANN buffer, false otherwise.
 */
static bool ggml_backend_buft_is_cann(ggml_backend_buffer_type_t buft);
static bool ggml_backend_buffer_is_cann(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_cann(buffer->buft);
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
static void ggml_backend_cann_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    delete ctx;
}

/**
 * @brief Retrieve the base pointer of a CANN buffer.
 *
 * This function returns the base pointer of a CANN buffer, which points to the
 * device memory allocated for the buffer.
 *
 * @param buffer The CANN buffer whose base pointer is to be retrieved.
 * @return A pointer to the base of the device memory allocated for the buffer.
 */
static void* ggml_backend_cann_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    return ctx->dev_ptr;
}

/**
 * @brief Transform quantized Q4.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q4.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q4.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
#include <omp.h>

#include <chrono>
static void ggml_backend_cann_transform_q4_0(ggml_tensor* tensor,
                                             const void* src, void* dst) {
    // auto start = std::chrono::high_resolution_clock::now();

    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK4_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t) / 2;
    const auto* src_q4 = static_cast<const block_q4_0*>(src);

    // Get system's maximum thread count
    int max_threads = omp_get_max_threads();

    // Adjust thread count based on task size to ensure each thread handles
    // enough groups to justify the thread overhead
    int num_threads = std::min(max_threads, (int)(groups / 64 + 1));
    omp_set_num_threads(num_threads);

// Phase 1: Process quantization data and scales in parallel
#pragma omp parallel
    {
// Each thread processes its own chunk of the data
#pragma omp for
        for (int64_t i = 0; i < groups; ++i) {
            const block_q4_0* group = &src_q4[i];
            uint8_t* quant_offset = (uint8_t*)dst + i * QK4_0 / 2;
            uint16_t* scale_offset = (uint16_t*)((char*)dst + quant_bytes) + i;

            // Store scale
            *scale_offset = group->d;

            // Process 0-15 part (low 4 bits)
            for (int j = 0; j < QK4_0 / 2; j += 2) {
                quant_offset[j / 2] =
                    (group->qs[j] & 0x0F) | ((group->qs[j + 1] << 4));
            }

            // Process 16-31 part (high 4 bits)
            uint8_t* quant_offset2 = quant_offset + QK4_0 / 4;
            for (int j = 0; j < QK4_0 / 2; j += 2) {
                quant_offset2[j / 2] =
                    (group->qs[j] >> 4) | (group->qs[j + 1] & 0xF0);
            }
        }
    }

    // Phase 2: XOR transformation in parallel with larger chunks for better
    // efficiency
    const size_t chunk_size =
        4096;  // Process in larger chunks for better cache efficiency
    const size_t num_chunks = (quant_bytes + chunk_size - 1) / chunk_size;

#pragma omp parallel for
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        size_t start = chunk * chunk_size;
        size_t end = std::min(start + chunk_size, quant_bytes);

        for (size_t i = start; i < end; i++) {
            ((uint8_t*)dst)[i] ^= 0x88;
        }
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end
    // - start);

    // printf("ggml_backend_cann_transform_q4_0: n_elems = %ld, time = %.3f ms,
    // threads = %d\n",
    //        n_elems, duration.count() / 1000.0f, num_threads);
}

/**
 * @brief Transform CANN processed data back into quantized Q4.0 format.
 *
 * This function transforms CANN processed data back into quantized Q4.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q4_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q4.0 formatted data
 * will be stored.
 */
static void ggml_backend_cann_transform_back_q4_0(const ggml_tensor* tensor,
                                                  void* src, void* dst) {
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK4_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t) / 2;

    // Get system's maximum thread count
    int max_threads = omp_get_max_threads();

    // Adjust thread count based on task size to ensure each thread handles
    // enough groups to justify the thread overhead
    int num_threads = std::min(max_threads, (int)(groups / 64 + 1));
    omp_set_num_threads(num_threads);

    // Phase 1: XOR transformation in parallel with larger chunks for better
    // efficiency
    const size_t chunk_size =
        4096;  // Process in larger chunks for better cache efficiency
    const size_t num_chunks = (quant_bytes + chunk_size - 1) / chunk_size;

#pragma omp parallel for
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        size_t start = chunk * chunk_size;
        size_t end = std::min(start + chunk_size, quant_bytes);

        for (size_t i = start; i < end; i++) {
            ((uint8_t*)src)[i] ^= 0x88;
        }
    }

// Phase 2: Process conversion back to original format in parallel
#pragma omp parallel for
    for (int i = 0; i < groups; i++) {
        block_q4_0* group = (block_q4_0*)((char*)dst + i * sizeof(block_q4_0));
        const uint8_t* quant_offset = (uint8_t*)src + i * QK4_0 / 2;
        const uint16_t* scale_offset =
            (uint16_t*)((char*)src + quant_bytes) + i;

        // Set scale
        group->d = *scale_offset;

        // Process 0-15 part (low 4 bits)
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] = (quant_offset[j / 2] & 0x0F);
            group->qs[j + 1] = (quant_offset[j / 2] >> 4);
        }

        // Process 16-31 part (high 4 bits)
        const uint8_t* quant_offset2 = quant_offset + QK4_0 / 4;
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] |= (quant_offset2[j / 2] << 4);
            group->qs[j + 1] |= (quant_offset2[j / 2] & 0xF0);
        }
    }
}

/**
 * @brief Transform quantized Q8.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q8.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q8.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
static void ggml_backend_cann_transform_q8_0(ggml_tensor* tensor,
                                             const void* src, void* dst) {
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK8_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t);
    const auto* src_q8 = static_cast<const block_q8_0*>(src);

    // Get system's maximum thread count
    int max_threads = omp_get_max_threads();

    // Adjust thread count based on task size
    int num_threads = std::min(max_threads, (int)(groups / 64 + 1));
    omp_set_num_threads(num_threads);

// Parallel processing of quantization groups
#pragma omp parallel for
    for (int64_t i = 0; i < groups; i++) {
        const block_q8_0* group = &src_q8[i];
        uint8_t* quant_offset = (uint8_t*)dst + i * QK8_0;
        uint16_t* scale_offset = (uint16_t*)((char*)dst + quant_bytes) + i;

        // Store scale
        *scale_offset = group->d;

        // Copy quantized values
        memcpy(quant_offset, group->qs, QK8_0 * sizeof(uint8_t));
    }
}

/**
 * @brief Transform CANN processed data back into quantized Q8.0 format.
 *
 * This function transforms CANN processed data back into quantized Q8.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q8_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q8.0 formatted data
 * will be stored.
 */
static void ggml_backend_cann_transform_back_q8_0(const ggml_tensor* tensor,
                                                  const void* src, void* dst) {
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK8_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t);

    // Get system's maximum thread count
    int max_threads = omp_get_max_threads();

    // Adjust thread count based on task size
    int num_threads = std::min(max_threads, (int)(groups / 64 + 1));
    omp_set_num_threads(num_threads);

// Parallel processing of groups
#pragma omp parallel for
    for (int i = 0; i < groups; i++) {
        block_q8_0* group = (block_q8_0*)((char*)dst + i * sizeof(block_q8_0));
        const uint8_t* quant_offset = (const uint8_t*)src + i * QK8_0;
        const uint16_t* scale_offset =
            (const uint16_t*)((const char*)src + quant_bytes) + i;

        // Set scale
        group->d = *scale_offset;

        // Copy quantized values
        memcpy(group->qs, quant_offset, QK8_0 * sizeof(uint8_t));
    }
}

/**
 * @brief Transform tensor data based on its type for CANN processing.
 *
 * This function transforms tensor data based on its quantization type for CANN
 * processing. It dispatches the transformation based on the tensor's type to
 * specialized functions handling Q4.0 and Q8.0 formats.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data to be transformed.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
static void ggml_backend_cann_transform(ggml_tensor* tensor, const void* src,
                                        void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            break;
        default:
            GGML_ASSERT(false &&
                        "ggml_backend_cann_transform: unsupported type");
            break;
    }
}

/**
 * @brief Transform CANN processed data back into tensor data based on its type.
 *
 * This function transforms CANN processed data back into tensor data based on
 * its quantization type for Q4.0 and Q8.0 formats. It dispatches the
 * transformation based on the tensor's type to specialized functions.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data containing CANN processed data.
 * @param dst Pointer to the destination buffer where transformed tensor data
 * will be stored.
 */
static void ggml_backend_cann_transform_back(const ggml_tensor* tensor,
                                             void* src, void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_back_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_back_q8_0(tensor, src, dst);
            break;
        default:
            GGML_ASSERT(false &&
                        "ggml_backend_cann_transform_back: unsupported type");
            break;
    }
}

/**
 * @brief Check if transformation is needed for a given tensor type.
 *
 * This function checks if transformation is needed for a given tensor type
 * to prepare data for CANN processing.
 *
 * @param type The tensor type to check.
 * @return true if transformation is needed, false otherwise.
 */
static bool need_transform(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Initialize a tensor using data from a CANN buffer.
 *
 * This function initializes a tensor using data from a CANN buffer.
 * It handles special cases such as views and quantization.
 *
 * @param buffer The CANN buffer from which to initialize the tensor.
 * @param tensor Pointer to the tensor to be initialized.
 */
static enum ggml_status ggml_backend_cann_buffer_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    // TODO: can backend doesn't support quantized yet. Just leave the code
    // here.
    if (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size =
            ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            size_t memset_size = padded_size - original_size;
            ACL_CHECK(aclrtMemset((char*)tensor->data + original_size,
                                  memset_size, 0, memset_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

// TODO: need handle tensor which has paddings.
/**
 * @brief Set tensor data in a CANN buffer.
 *
 * This function sets tensor data in a CANN buffer, handling transformations
 * if needed based on the tensor's type.
 *
 * @param buffer The CANN buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param data Pointer to the source data to be copied into the tensor.
 * @param offset Offset in the source data from where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_cann_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor* tensor,
                                                const void* data, size_t offset,
                                                size_t size) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    // TODO: refer to cann(#6017), it use thread's default stream.
    // For acl, synchronous functions use this default stream.
    // Why aclrtSynchronizeDevice?

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy((char*)tensor->data + offset, size, data, size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        void* transform_buffer = malloc(size);
        ggml_backend_cann_transform(tensor, data, transform_buffer);

        ACL_CHECK(aclrtMemcpy((char*)tensor->data + offset, size,
                              transform_buffer, size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        free(transform_buffer);
    }
}

/**
 * @brief Get tensor data from a CANN buffer.
 *
 * This function retrieves tensor data from a CANN buffer, handling
 * transformations if needed based on the tensor's type.
 *
 * @param buffer The CANN buffer from which to retrieve tensor data.
 * @param tensor Pointer to the tensor whose data will be retrieved.
 * @param data Pointer to the destination buffer where the tensor data will be
 * copied.
 * @param offset Offset in the destination buffer where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_cann_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor* tensor,
                                                void* data, size_t offset,
                                                size_t size) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(data, size, (char*)tensor->data + offset, size,
                              ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        void* transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpy(transform_buffer, size,
                              (char*)tensor->data + offset, size,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        ggml_backend_cann_transform_back(tensor, transform_buffer, data);
        free(transform_buffer);
    }
}

/**
 * @brief Copy tensor data between CANN buffers if possible.
 *
 * This function copies tensor data between CANN buffers if the source and
 * destination buffers are CANN buffers and they meet the necessary conditions
 * (same device or devices can access each other).
 *
 * @param buffer The destination CANN buffer where the tensor data will be
 * copied.
 * @param src Pointer to the source tensor whose data will be copied.
 * @param dst Pointer to the destination tensor where the data will be copied.
 * @return true if the copy operation succeeded, false otherwise.
 */
static bool ggml_backend_cann_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor* src,
                                                ggml_tensor* dst) {
    if (ggml_backend_buffer_is_cann(src->buffer)) {
        ggml_backend_cann_buffer_context* src_ctx =
            (ggml_backend_cann_buffer_context*)src->buffer->context;
        ggml_backend_cann_buffer_context* dst_ctx =
            (ggml_backend_cann_buffer_context*)buffer->context;

        size_t memcpy_size = ggml_nbytes(src);
        // Same device.
        if (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                  (const char*)src->data, memcpy_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE));
            return true;
        } else {
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device,
                                               dst_ctx->device));
            if (canAccessPeer) {
                ggml_cann_set_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                      (const char*)src->data, memcpy_size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Clear a CANN buffer by setting all its memory to a specified value.
 *
 * This function clears a CANN buffer by setting all its memory to a specified
 * value.
 *
 * @param buffer The CANN buffer to be cleared.
 * @param value The value to which each byte in the buffer will be set.
 */
static void ggml_backend_cann_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
static const ggml_backend_buffer_i ggml_backend_cann_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cann_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cann_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cann_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cann_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cann_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cann_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_buffer_clear,
    /* .reset           = */ NULL,
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
struct ggml_backend_cann_buffer_type_context {
    int32_t
        device; /**< Device identifier associated with the buffer context. */
    std::string name; /**< Name associated with the buffer context. */
};

/**
 * @brief Retrieves the name associated with a CANN buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char* ggml_backend_cann_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    ggml_backend_cann_buffer_type_context* buft_ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;

    return buft_ctx->name.c_str();
}

/**
 * @brief Allocates a new CANN buffer of the specified type and size.
 *
 * This function allocates a new CANN buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_cann_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cann_buffer_type_context* buft_ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;

    ggml_cann_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1);

    void* dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR(
            "%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx->device,
            aclGetRecentErrMsg());
        return nullptr;
    }

    ggml_backend_cann_buffer_context* ctx =
        new ggml_backend_cann_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cann_buffer_interface,
                                    ctx, size);
}

/**
 * @brief Retrieves the memory alignment requirement for CANN buffers of this
 * type.
 *
 * This function returns the alignment requirement in bytes for memory allocated
 * by the CANN buffer type.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @return The alignment requirement in bytes (fixed at 128 bytes for CANN
 * buffers).
 */
static size_t ggml_backend_cann_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

/**
 * @brief Calculates the allocation size required for a tensor in a CANN buffer.
 *
 * Computes the total allocation size needed for storing the tensor's data in a
 * CANN buffer, considering any necessary padding or adjustments for quantized
 * types.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @param tensor Pointer to the tensor for which the allocation size is
 * calculated.
 * @return The total allocation size in bytes required for the tensor in the
 * CANN buffer.
 */
static size_t ggml_backend_cann_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);

    // TODO: not support quantized yet.
    // TODO: consider un-continue tensor.
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(
                tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cann_buffer_type_is_host(
    ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
static const ggml_backend_buffer_type_i
    ggml_backend_cann_buffer_type_interface = {
        /* .get_name         = */ ggml_backend_cann_buffer_type_name,
        /* .alloc_buffer     = */ ggml_backend_cann_buffer_type_alloc_buffer,
        /* .get_alignment    = */ ggml_backend_cann_buffer_type_get_alignment,
        /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
        /* .get_alloc_size   = */ ggml_backend_cann_buffer_type_get_alloc_size,
        /* .is_host          = */ ggml_backend_cann_buffer_type_is_host,
};

/**
 * @brief Retrieves the CANN buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
ggml_backend_buffer_type_t ggml_backend_cann_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cann_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_cann_buffer_types[GGML_CANN_MAX_DEVICES];

    static bool ggml_backend_cann_buffer_type_initialized = false;

    if (!ggml_backend_cann_buffer_type_initialized) {
        for (int32_t i = 0; i < ggml_cann_info().device_count; i++) {
            ggml_backend_cann_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cann_buffer_type_interface,
                /* .device    = */
                ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                new ggml_backend_cann_buffer_type_context{
                    i, "CANN" + std::to_string(i)},
            };
        }
        ggml_backend_cann_buffer_type_initialized = true;
    }

    return &ggml_backend_cann_buffer_types[device];
}

/**
 * @brief Retrieves the name associated with a CANN host buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN host buffer type context.
 *
 * @param buft Pointer to the host buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char* ggml_backend_cann_host_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    return "CANN_Host";

    GGML_UNUSED(buft);
}

/**
 * @brief Retrieves the name associated with a CANN host buffer.
 *
 * This function returns the descriptive name associated with the specified
 * CANN host buffer context.
 *
 * @param buft Pointer to the host buffer context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char* ggml_backend_cann_host_buffer_name(
    ggml_backend_buffer_t buffer) {
    return "CANN_Host";

    GGML_UNUSED(buffer);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer,
 * including its context.
 *
 * @param buffer The CANN host buffer to free.
 */
static void ggml_backend_cann_host_buffer_free(ggml_backend_buffer_t buffer) {
    ACL_CHECK(aclrtFreeHost(buffer->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
static void* ggml_cann_host_malloc(size_t size) {
    if (getenv("GGML_CANN_NO_PINNED") != nullptr) {
        return nullptr;
    }

    const size_t alignment = 128;
    size = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }

    void* hostPtr = nullptr;
    aclError err = aclrtMallocHost((void**)&hostPtr, size);
    if (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n",
                      __func__, size / 1024.0 / 1024.0, aclGetRecentErrMsg());
        return nullptr;
    }
    return hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if
 * allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_cann_host_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    void* hostPtr = ggml_cann_host_malloc(size);

    if (hostPtr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),
                                              size);
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_cpu_buffer_from_ptr(hostPtr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cann_host_buffer_free;

    return buffer;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_buffer_type_t ggml_backend_cann_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cann_buffer_type_host =
        {
            /* .iface    = */ {
                /* .get_name         = */
                ggml_backend_cann_host_buffer_type_name,
                /* .alloc_buffer     = */
                ggml_backend_cann_host_buffer_type_alloc_buffer,
                /* .get_alignment    = */
                ggml_backend_cpu_buffer_type()->iface.get_alignment,
                /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
                                                 /* .get_alloc_size   = */
                ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                /* .is_host          = */
                ggml_backend_cpu_buffer_type()->iface.is_host,
            },
            /* .device   = */
            ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
            /* .context  = */ nullptr,
        };

    return &ggml_backend_cann_buffer_type_host;
}

/**
 * @brief Computes the forward operation for a given tensor using CANN
 * operations.
 *
 * This function selects the appropriate CANN operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The CANN context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @param key A unique identifier string for the operation, used for caching.
 * @return true if the computation was successful; false otherwise.
 */
static bool ggml_cann_compute_forward(ggml_backend_cann_context& ctx,
                                      struct ggml_tensor* dst,
                                      std::string key = "") {
    GGML_UNUSED(key);
    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cann_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_cann_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cann_mul_div<aclnnDivGetWorkspaceSize, aclnnDiv>(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cann_activation<aclnnSiluGetWorkspaceSize, aclnnSilu>(
                        ctx, dst);
                    break;
                // TODO: Use faster gelu??
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cann_activation<aclnnTanhGetWorkspaceSize, aclnnTanh>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cann_activation<aclnnReluGetWorkspaceSize, aclnnRelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cann_activation<aclnnHardsigmoidGetWorkspaceSize,
                                         aclnnHardsigmoid>(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cann_activation<aclnnHardswishGetWorkspaceSize,
                                         aclnnHardswish>(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cann_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_FUSED:
            ggml_cann_rms_norm_fused(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            break;
        //     return false;
        case GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cann_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cann_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cann_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cann_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            break;
#ifdef GGML_USE_HCCL
        case GGML_OP_ALL_REDUCE_SUM:
            ggml_cann_allreduce_sum(ctx, dst);
            break;
#endif
        case GGML_OP_TO_ZERO:
            ggml_cann_set_tensor_to_zero(ctx, dst);
            break;
        case GGML_OP_MOE_FUSED:
            ggml_cann_moe_fused(ctx, dst);
            break;
        case GGML_OP_MOE_FUSED_CPU:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_PROMPT:
            ggml_cann_flash_attn_prompt(ctx, dst);
            break;
#ifdef LLAMA_JITTOR_OPS_SUPPORT
        case GGML_OP_FLASH_ATTN_JITTOR_V1:
            ggml_cann_flash_attn_jittor_v1(ctx, dst);
            break;
#endif
        case GGML_OP_GET_SLICE:
            ggml_cann_get_slice(ctx, dst);
            break;
        case GGML_OP_SCATTER_UPDATE:
            ggml_cann_scatter_update(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

// backend
/**
 * @brief Retrieves the name associated with the CANN backend.
 *
 * This function returns the name assigned to the CANN backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the CANN backend structure.
 * @return A pointer to a constant string representing the backend name.
 */
static const char* ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
static void ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    // finalize when last backend freed.
    if (cann_ctx->device == ggml_backend_cann_get_device_count() - 1) {
        ACL_CHECK(aclFinalize());
    }

    delete cann_ctx;
    delete backend;
}

/**
 * @brief Sets tensor data asynchronously in the CANN backend.
 *
 * This function asynchronously sets tensor data in the CANN backend. Depending
 * on the tensor type, it may perform data transformations before copying data
 * to the device.
 *
 * @param backend Pointer to the CANN backend structure.
 * @param tensor Pointer to the tensor structure to set data for.
 * @param data Pointer to the host data to copy to the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_cann_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor* tensor,
                                               const void* data, size_t offset,
                                               size_t size) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpyAsync((char*)tensor->data + offset, size, data,
                                   size, ACL_MEMCPY_HOST_TO_DEVICE,
                                   cann_ctx->stream()));
    } else {
        void* transform_buffer = malloc(size);
        ggml_backend_cann_transform(tensor, data, transform_buffer);

        ACL_CHECK(aclrtMemcpyAsync(
            (char*)tensor->data + offset, size, transform_buffer, size,
            ACL_MEMCPY_HOST_TO_DEVICE, cann_ctx->stream()));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
        free(transform_buffer);
    }
}

static void ggml_backend_cann_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor* tensor,
                                               void* data, size_t offset,
                                               size_t size) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ggml_backend_buffer_t buf =
        tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cann_buffer_type(cann_ctx->device) &&
                "unsupported buffer type");

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpyAsync(data, size, (char*)tensor->data + offset,
                                   size, ACL_MEMCPY_DEVICE_TO_HOST,
                                   cann_ctx->stream()));
    } else {
        void* transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpyAsync(
            transform_buffer, size, (char*)tensor->data + offset, size,
            ACL_MEMCPY_DEVICE_TO_HOST, cann_ctx->stream()));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
        ggml_backend_cann_transform_back(tensor, transform_buffer, data);
        free(transform_buffer);
    }
}

/**
 * @brief Asynchronously copies tensor data between CANN backends.
 *
 * This function copies tensor data asynchronously between two CANN backends. It
 * checks if both tensors reside in CANN buffers and whether the devices support
 * peer-to-peer access for direct copying. If not, it returns false.
 *
 * @param backend_src Pointer to the source CANN backend structure.
 * @param backend_dst Pointer to the destination CANN backend structure.
 * @param src Pointer to the source tensor to copy data from.
 * @param dst Pointer to the destination tensor to copy data to.
 * @return true if the copy operation succeeds, false otherwise.
 */
static bool ggml_backend_cann_cpy_tensor_async(ggml_backend_t backend_src,
                                               ggml_backend_t backend_dst,
                                               const ggml_tensor* src,
                                               ggml_tensor* dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) ||
                ggml_backend_is_cann(backend_dst));

    if (!ggml_backend_buffer_is_cann(src->buffer) ||
        !ggml_backend_buffer_is_cann(dst->buffer)) {
        return false;
    }

    ggml_backend_buffer_t buf_src =
        src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst =
        dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_cann_context* cann_ctx_src =
        (ggml_backend_cann_context*)backend_src->context;
    ggml_backend_cann_context* cann_ctx_dst =
        (ggml_backend_cann_context*)backend_dst->context;

    size_t copy_size = ggml_nbytes(dst);
    if (backend_src != backend_dst) {
        ggml_backend_cann_buffer_context* buf_ctx_src =
            (ggml_backend_cann_buffer_context*)buf_src->context;
        ggml_backend_cann_buffer_context* buf_ctx_dst =
            (ggml_backend_cann_buffer_context*)buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device,
                                           cann_ctx_dst->device));
        if (!canAccessPeer) {
            return false;
        }

        // need open both directions for memcpyasync between devices.
        ggml_cann_set_device(cann_ctx_dst->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_set_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_src->stream()));

        // TODO: workaround for Event didn`t work here.
        aclrtSynchronizeStream(cann_ctx_src->stream());
    } else {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    return true;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
static void ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
}

/**
 * @brief 在CANN后端执行计算图的核心函数，使用aclnn_ops.h中的算子
 *
 * @param backend CANN后端实例
 * @param cgraph 要执行的GGML计算图
 * @return 执行状态，成功或失败
 */
static enum ggml_status ggml_backend_cann_graph_compute_aclnn(
    ggml_backend_t backend, ggml_cgraph* cgraph) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }

        std::string key;
        if (node->op == GGML_OP_ADD) {
            key += std::to_string(node->op) +
                   std::to_string(node->src[0]->ne[0]) +
                   std::to_string(node->src[0]->ne[1]) +
                   std::to_string(node->src[0]->ne[2]) +
                   std::to_string(node->src[0]->ne[3]);
            key += std::to_string(node->src[1]->ne[0]) +
                   std::to_string(node->src[1]->ne[1]) +
                   std::to_string(node->src[1]->ne[2]) +
                   std::to_string(node->src[1]->ne[3]);
        }

        bool ok = ggml_cann_compute_forward(*cann_ctx, node, key);

        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__,
                           node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief 在CANN后端执行计算图的核心函数
 *
 * 该函数负责将GGML计算图转换为华为昇腾(Ascend)计算图并执行计算
 * 支持图缓存优化，避免重复构建相同结构的图
 *
 * @param backend CANN后端实例
 * @param cgraph 要执行的GGML计算图
 * @return 执行状态，成功或失败
 */
static enum ggml_status ggml_backend_cann_graph_compute(ggml_backend_t backend,
                                                        ggml_cgraph* cgraph) {
    // 获取环境变了，USE_ACLNN 为1时，使用aclnn_ops.h中的算子
    // 否则使用ge中的算子
    // 如果cgraph->flags为1，则使用aclnn_ops.h中的算子
    if ((cgraph->flags & 1) == 0 ||
        (getenv("USE_ACLNN") != nullptr && atoi(getenv("USE_ACLNN")) == 1)) {
        if (cgraph->flags & 2) {
            return GGML_STATUS_SUCCESS;
        }
        return ggml_backend_cann_graph_compute_aclnn(backend, cgraph);
    }

    // 从后端获取CANN上下文，包含Ascend图和会话信息
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    // 从backend里面获取device_id
    int device_id = cann_ctx->device;

    // 第一部分：初始化Ascend图和GE(Graph Engine)环境
    // --------------------------------------------------------------
    // 检查Ascend图是否已初始化，如果未初始化则创建新的图和会话
    if (cann_ctx->ascend_graph == nullptr) {
        // 配置GE初始化参数
        std::map<AscendString, AscendString> config = {
            {"ge.exec.deviceId",
             AscendString(
                 std::to_string(device_id)
                     .c_str())},  // AscendString(const char_t *const name);
            {"ge.graphRunMode", "0"},              // 设置图运行模式
            {"ge.tiling_schedule_optimize", "1"},  // 优化tilling
            // {"ge.opDebugLevel", "1"},  // 调试级别，当前已注释
            // {"ge.exec.enable_exception_dump","1"}, // 异常转储，当前已注释
            {"ge.exec.precision_mode", "allow_fp32_to_fp16"},
            {"ge.exec.reuseZeroCopyMemory", "1"}
            // 允许FP32到FP16的自动转换，提高性能
        };

        // 初始化GE环境
        ge::Status ret = ge::GEInitialize(config);
        if (ret != 0) {
            return GGML_STATUS_ABORTED;  // 初始化失败则返回中止状态
        }

        // 创建Ascend图结构和会话
        cann_ctx->ascend_graph = std::make_unique<ggml_ascend_graph>();
        // cann_ctx->ascend_graph->graphs = std::vector<Graph>();  //
        // 注释掉的图列表

        // 创建处理过的图的缓存映射，用size_t(地址+时间戳hash)作为键，图索引作为值
        cann_ctx->ascend_graph->processed_graphs =
            std::unordered_map<size_t, uint32_t>();

        // 创建Ascend会话，用于管理和执行图
        std::map<ge::AscendString, ge::AscendString> options;
        cann_ctx->ascend_graph->session = new Session(options);
    }

    // 第二部分：图缓存和复用机制
    // --------------------------------------------------------------
    //  // 使用图的内存地址加时间戳作为唯一标识符，用于图缓存
    //  auto graph_address =
    //  reinterpret_cast<size_t>(static_cast<void*>(cgraph));

    //  // 创建复合键：地址 + 时间戳字符串的hash
    //  size_t time_hash = std::hash<std::string>{}(cgraph->graph_name_by_time);
    //  size_t graph_key = graph_address ^ (time_hash << 1);

    size_t graph_key = std::hash<std::string>{}(cgraph->graph_name_by_time);

    uint32_t graph_idx = 0;  // 图索引，用于在会话中标识特定图

    // 获取当前流
    aclrtStream stream = cann_ctx->stream();

    // 第三部分：图构建或复用逻辑
    // --------------------------------------------------------------
    // 检查当前图是否已经处理过（是否在缓存中）
    if (cann_ctx->ascend_graph->processed_graphs.find(graph_key) ==
        cann_ctx->ascend_graph->processed_graphs.end()) {
        // 图未处理过，需要构建新图

        cann_ctx->input_init.clear();
        cann_ctx->output_init.clear();

        // 分配新的图索引
        graph_idx =
            (uint32_t)cann_ctx->ascend_graph->processed_graphs.size() + 1;

        // 构建Ascend图，同时初始化输入输出Tensor
        auto new_graph = build_ascend_graph(
            cgraph, *cann_ctx, cann_ctx->input_init, cann_ctx->output_init);

        // 缓存新构建的图，记录其地址和索引的映射关系
        cann_ctx->ascend_graph->processed_graphs[graph_key] = graph_idx;

        // 从环境变量获取路径 in
        // {WORK_PATH}/ggml/src/ggml-cann/fusion_switch.cfg
        const char* fusion_switch_path_env =
            std::getenv("FUSION_SWITCH_FILE_PATH");
        ge::Status ret;
        if (fusion_switch_path_env != nullptr &&
            std::strlen(fusion_switch_path_env) > 0) {
            std::map<AscendString, AscendString> global_options = {
                {ge::ir_option::FUSION_SWITCH_FILE, fusion_switch_path_env},
            };
            // 将新图添加到会话中
            ret = cann_ctx->ascend_graph->session->AddGraph(
                graph_idx, new_graph, global_options);
        } else {
            // 不传入 global_options
            ret =
                cann_ctx->ascend_graph->session->AddGraph(graph_idx, new_graph);
        }

        if (ret != SUCCESS) {
            printf("Graph add failed: %d\n", ret);
            return GGML_STATUS_ABORTED;  // 添加图失败则返回中止状态
        }
        ret = cann_ctx->ascend_graph->session->CompileGraph(graph_idx);
        if (ret != SUCCESS) {
            printf("Graph compile failed: %d\n", ret);
            return GGML_STATUS_ABORTED;  // 编译失败则返回中止状态
        }
        ret = cann_ctx->ascend_graph->session->LoadGraph(graph_idx, {}, stream);
        if (ret != SUCCESS) {
            printf("Graph load failed: %d\n", ret);
            return GGML_STATUS_ABORTED;  // 加载失败则返回中止状态
        }
    } else {
        // 图已经处理过，获取缓存的图索引
        graph_idx = cann_ctx->ascend_graph->processed_graphs[graph_key];

        // 重建图的输入输出Tensor，但不重建图结构
        // 这是为了处理输入数据可能发生变化的情况
        auto ret = reuse_ascend_graph(
            graph_idx, cann_ctx->ascend_graph->session, cgraph, stream,
            cann_ctx->input_init, cann_ctx->output_init);
        if (ret != SUCCESS) {
            printf("Graph reuse failed: %d\n", ret);
            return GGML_STATUS_ABORTED;  // 重建失败则返回中止状态
        }
    }

    if (cgraph->flags & 2) {
        return GGML_STATUS_SUCCESS;
    }

    // 异步执行图计算
    // ret = cann_ctx->ascend_graph->session->RunGraphWithStreamAsync(
    //     graph_idx, stream, input_init, output_init);
    // if (ret != SUCCESS) {
    //     printf("Graph async run failed: %d\n", ret);
    //     return GGML_STATUS_ABORTED;  // 运行失败则返回中止状态
    // }

    ge::Status ret =
        cann_ctx->ascend_graph->session->ExecuteGraphWithStreamAsync(
            graph_idx, stream, cann_ctx->input_init, cann_ctx->output_init);
    if (ret != SUCCESS) {
        printf("Graph async execute failed: %d\n", ret);
        return GGML_STATUS_ABORTED;  // 运行失败则返回中止状态
    }

    // 同步流，确保任务执行完成
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // 计算成功完成
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Checks if the CANN backend supports a specific operation.
 *
 * This function checks whether the specified operation is supported by the
 * CANN backend.
 *
 * @param backend Pointer to the CANN backend structure to check support for
 *                the operation.
 * @param op Pointer to the tensor representing the operation to check.
 * @return bool Returns true if the operation is supported by the backend,
 *              otherwise false.
 */
static bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev,
                                          const ggml_tensor* op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT: {
            switch (op->src[0]->type) {
                case GGML_TYPE_Q8_0:
                    // Current groupsize should not be greater than k-1 in
                    // aclnnWeightQuantBatchMatmulV2GetWorkspaceSize
                    if (op->src[0]->ne[0] <= QK8_0) {
                        return false;
                    }
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                case GGML_TYPE_Q4_0:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_MUL_MAT_ID:
            return true;
            //  return false;
        // embedding
        case GGML_OP_GET_ROWS: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        } break;
        case GGML_OP_CPY: {
            switch (op->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I32:
                case GGML_TYPE_I64:
                case GGML_TYPE_I8:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_CONT: {
            // TODO: support GGML_TYPE_BF16
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_ROPE: {
            // TODO: with ops-test v == 1
            float* ext_factor = (float*)((int32_t*)op->op_params + 7);
            // TODO: n_dims <= ne0
            if (op->src[0]->ne[0] != op->op_params[1]) {
                return false;
            }
            // TODO: ext_factor != 0
            if (*ext_factor != 0) {
                // printf("ROPE False ext_factor: %f\n", *ext_factor);
                //  return false;
                return true;
            }

            const int mode = ((const int32_t*)op->op_params)[2];
            if (mode & GGML_ROPE_TYPE_MROPE) {
                return false;
            }
            if (mode & GGML_ROPE_TYPE_VISION) {
                return false;
            }

            return true;
        }
        case GGML_OP_UPSCALE: {
            // aclnnUpsampleNearest2dGetWorkspaceSize not support
            // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
            if (op->src[0]->ne[2] * op->ne[3] !=
                op->src[0]->ne[3] * op->ne[2]) {
                return false;
            }
            return true;
        }
#ifdef GGML_USE_HCCL
        case GGML_OP_ALL_REDUCE_SUM:
            return true;
#endif
        case GGML_OP_IM2COL:
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_MOE_FUSED:
        case GGML_OP_FLASH_ATTN_PROMPT:
#ifdef LLAMA_JITTOR_OPS_SUPPORT
        case GGML_OP_FLASH_ATTN_JITTOR_V1:
#endif
        case GGML_OP_TO_ZERO:
        case GGML_OP_SCATTER_UPDATE:
        case GGML_OP_GET_SLICE:
        case GGML_OP_RMS_NORM_FUSED:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

/**
 * @brief Checks if the backend buffer type is associated with the CANN backend.
 *
 * This function checks whether the provided backend buffer type is associated
 * with the CANN backend based on the comparison of its name retrieval function
 * pointer.
 *
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the buffer type is associated with the CANN
 * backend, otherwise false.
 */
static bool ggml_backend_buft_is_cann(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cann_buffer_type_name;
}

/**
 * @brief Determines if a tensor operation should be offloaded to the CANN
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * CANN backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the CANN backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
static bool ggml_backend_cann_offload_op(ggml_backend_dev_t dev,
                                         const ggml_tensor* op) {
    const int min_batch_size = 32;
    GGML_UNUSED(dev);

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
static void ggml_backend_cann_event_record(ggml_backend_t backend,
                                           ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ACL_CHECK(aclrtRecordEvent((aclrtEvent)event->context, cann_ctx->stream()));
}

/**
 * @brief Waits for a recorded event to complete on the CANN backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * ACL runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
static void ggml_backend_cann_event_wait(ggml_backend_t backend,
                                         ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    if (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx->stream(),
                                       (aclrtEvent)event->context));
    } else {
        GGML_ABORT("fatal error");
    }
}

/**
 * @brief Structure defining the interface for the CANN backend.
 *
 * This structure contains function pointers for various operations
 * supported by the CANN backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static const ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_cann_guid() {
    static ggml_guid guid = {0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                             0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64};
    return &guid;
}

// backend device
struct ggml_backend_cann_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char* ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context* ctx =
        (ggml_backend_cann_device_context*)dev->context;
    return ctx->name.c_str();
}

static const char* ggml_backend_cann_device_get_description(
    ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context* ctx =
        (ggml_backend_cann_device_context*)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_cann_device_get_memory(ggml_backend_dev_t dev,
                                                size_t* free, size_t* total) {
    ggml_backend_cann_device_context* ctx =
        (ggml_backend_cann_device_context*)dev->context;
    ggml_backend_cann_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_cann_device_get_type(
    ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cann_device_get_props(ggml_backend_dev_t dev,
                                               ggml_backend_dev_props* props) {
    props->name = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_memory(dev, &props->memory_free,
                                        &props->memory_total);

    bool host_buffer = getenv("GGML_CANN_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev,
                                                    const char* params) {
    GGML_UNUSED(params);
    ggml_backend_cann_device_context* ctx =
        (ggml_backend_cann_device_context*)dev->context;
    return ggml_backend_cann_init(ctx->device, params);
}

/**
 * @brief Checks if the CANN backend supports a specific backend buffer type.
 *
 * This function determines whether the CANN backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the CANN backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the CANN backend supports the buffer type,
 *              otherwise false.
 */
static bool ggml_backend_cann_supports_buft(ggml_backend_dev_t dev,
                                            ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context* dev_ctx =
            (ggml_backend_cann_device_context*)dev->context;
        ggml_backend_cann_buffer_type_context* buft_ctx =
            (ggml_backend_cann_buffer_type_context*)buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    return false;
}

static ggml_backend_buffer_type_t ggml_backend_cann_device_get_buffer_type(
    ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context* ctx =
        (ggml_backend_cann_device_context*)dev->context;
    return ggml_backend_cann_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cann_device_get_host_buffer_type(
    ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cann_host_buffer_type();
}

/**
 * @brief Creates a new event for the CANN backend device.
 *
 * This function initializes a new event for the CANN backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the CANN backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
static ggml_backend_event_t ggml_backend_cann_device_event_new(
    ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context* dev_ctx =
        (ggml_backend_cann_device_context*)dev->context;

    ggml_cann_set_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    return new ggml_backend_event{
        /* .device = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(),
                                                 dev_ctx->device),
        /* .context = */ event,
    };
}

/**
 * @brief Frees a CANN backend event.
 *
 * This function destroys the ACL runtime event associated with the given CANN
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
static void ggml_backend_cann_device_event_free(ggml_backend_dev_t dev,
                                                ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent)event->context));

    delete event;
    GGML_UNUSED(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_cann_device_event_synchronize(
    ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent)event->context));

    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_memory,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called
                                                                     // for
                                                                     // every
                                                                     // card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_buffer_type,
    /* .get_host_buffer_type    = */
    ggml_backend_cann_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
struct ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char* ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CANN_NAME;
}

static size_t ggml_backend_cann_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context* ctx =
        (ggml_backend_cann_reg_context*)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cann_reg_get_device(
    ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cann_reg_context* ctx =
        (ggml_backend_cann_reg_context*)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

#ifdef GGML_USE_HCCL
struct ggml_comm_initializer {
    HcclRootInfo rootinfo;
    int num_devices;
    int rank;
};

static size_t ggml_backend_cann_comm_init_overhead() {
    return sizeof(ggml_comm_initializer);
}

static void ggml_backend_cann_comm_init(void* comm_init, int num_devices) {
    ggml_comm_initializer* initializer = (ggml_comm_initializer*)comm_init;
    initializer->num_devices = num_devices;
    initializer->rank = 0;
    HCCL_CHECK(HcclGetRootInfo(&initializer->rootinfo));
}

static void ggml_backend_cann_comm_set_rank(void* comm_init, int rank) {
    ggml_comm_initializer* initializer = (ggml_comm_initializer*)comm_init;
    initializer->rank = rank;
}

static ggml_comm_initializer ggml_comm_initializer_from_params(void* params) {
    return *(ggml_comm_initializer*)params;
}

#endif

static void* ggml_backend_cann_reg_get_proc_address(ggml_backend_reg_t reg,
                                                    const char* name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
#ifdef GGML_USE_HCCL
    if (strcmp(name, "ggml_backend_comm_init_overhead") == 0) {
        return (void*)ggml_backend_cann_comm_init_overhead;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void*)ggml_backend_cann_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_set_rank") == 0) {
        return (void*)ggml_backend_cann_comm_set_rank;
    }
#endif
    // reserved for future use
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_address,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            aclInit(nullptr);
            ggml_backend_cann_reg_context* ctx =
                new ggml_backend_cann_reg_context;

            for (int i = 0; i < ggml_cann_info().device_count; i++) {
                ggml_backend_cann_device_context* dev_ctx =
                    new ggml_backend_cann_device_context();
                dev_ctx->description = aclrtGetSocName();
                dev_ctx->device = i;
                dev_ctx->name = GGML_CANN_NAME + std::to_string(i);
                ggml_cann_set_device(i);
                ggml_backend_dev_t dev = new ggml_backend_device{
                    /* .iface   = */ ggml_backend_cann_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx};
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cann_reg_interface,
                /* .context     = */ ctx};
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device, const char* params) {
    aclInit(nullptr);
    if (device < 0 || device >= ggml_backend_cann_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cann_context* ctx = new ggml_backend_cann_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }
    ggml_cann_set_device(ctx->device);

#ifdef GGML_USE_HCCL
    if (params != nullptr) {
        ggml_comm_initializer init =
            ggml_comm_initializer_from_params((void*)params);
        ctx->init_comm(init.rootinfo, init.num_devices, init.rank);
        GGML_LOG_INFO(
            "%s: initialized communication backend %d on CANN device %d\n",
            __func__, init.rank, device);
    }
#endif

    ggml_backend_t cann_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_cann_guid(),
        /* .interface = */ ggml_backend_cann_interface,
        /* .device    = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
        /* .context   = */ ctx};

    return cann_backend;
}

bool ggml_backend_is_cann(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_count() {
    return ggml_cann_info().device_count;
}

void ggml_backend_cann_get_device_description(int32_t device, char* description,
                                              size_t description_size) {
    ggml_cann_set_device(device);
    const char* soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name);
}

void ggml_backend_cann_get_device_memory(int32_t device, size_t* free,
                                         size_t* total) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)

void ggml_backend_cann_presample(ggml_backend_t backend,
                                 const ggml_tensor* logits, ggml_tensor* values,
                                 ggml_tensor* indices, size_t k) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ggml_cann_topk(*cann_ctx, logits, values, indices, k);
}
