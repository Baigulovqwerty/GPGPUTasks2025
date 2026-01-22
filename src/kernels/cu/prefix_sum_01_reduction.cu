#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
// prefix_sum_01_sum_reduction
__global__ void prefix_sum_01_precount(
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
    // // объединение двух запусков кернела в один
    // const unsigned int to_new = to + (1u << k);
    // if (!(index & 1) && (to_new < n)) {
    //     precount[to_new] += precount[to];
    // }
    // // объединение двух запусков кернела в один
    // const unsigned int to_new_new = to_new + (1u << k + 1);
    // if (!(index & 3) && (to_new_new < n)) {
    //     precount[to_new_new] += precount[to_new];
    // }
    // // объединение двух запусков кернела в один
    // const unsigned int to_new_new_new = to_new_new + (1u << k + 2);
    // if (!(index & 7) && (to_new_new_new < n)) {
    //     precount[to_new_new_new] += precount[to_new_new];
    // }
    // // объединение двух запусков кернела в один
    // const unsigned int to_new_new_new_new = to_new_new_new + (1u << k + 3);
    // if (!(index & 15) && (to_new_new_new_new < n)) {
    //     precount[to_new_new_new_new] += precount[to_new_new_new];
    // }
}

namespace cuda {
void prefix_sum_01_precount(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &precount, unsigned int n, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_precount<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), precount.cuptr(), n, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
