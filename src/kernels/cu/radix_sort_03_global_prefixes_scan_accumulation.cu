#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* precount,
    unsigned int* ans,
    unsigned int n
)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int acc = 0;
    unsigned int bit_acc = 0;
    if (index <= n) {
        for (int bit_ind = 31; bit_ind >= 0; --bit_ind) {
            if (index & (1u << bit_ind)) {
                acc += precount[bit_acc + (1u << bit_ind) - 1];
                bit_acc += (1u << bit_ind);
            }
        }
        ans[index - 1] = acc;
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &precount, gpu::gpu_mem_32u &ans, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(precount.cuptr(), ans.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
