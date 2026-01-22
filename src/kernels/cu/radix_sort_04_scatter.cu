#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* a,
    const unsigned int* buffer_pref,
          unsigned int* buffer,
    unsigned int num_bit,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        if ((a[index] >> num_bit) & 1) { // bit - 1
            buffer[buffer_pref[n - 1] + index - buffer_pref[index]] = a[index];
        } else {
            buffer[buffer_pref[index] - 1] = a[index];
        }
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &buffer_pref, gpu::gpu_mem_32u &buffer, unsigned int num_bit, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), buffer_pref.cuptr(), buffer.cuptr(), num_bit, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
