#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* a,
    unsigned int* buffer_zeros,
    unsigned int num_bit,
    unsigned int n)
{
    const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        buffer_zeros[index] = 1 - ((a[index] >> num_bit) & 1);
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &buffer_zeros, unsigned int num_bit, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), buffer_zeros.cuptr(), num_bit, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
