#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* a,
    unsigned int* precount,
    unsigned int n,
    unsigned int k
)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int from = ((1u << (k-1)) - 1) + index * (1u << k);
    unsigned int to = from + (1u << (k - 1));
    if (to < n) {
        precount[to] += precount[from];
    }

    for (int ii = 0; ii < FUSE_N; ++ii) {
        from = to;
        to = from + (1u << (k + ii));
        if (!(index & ((1u << (ii + 1)) - 1)) && (to < n)) {
            precount[to] += precount[from];
        }
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &precount, unsigned int n, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), precount.cuptr(), n, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
