#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <mma.h>
#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const half* a, // rows=h x cols=k
                       const half* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int t = 0; t < k; t += 16) {
        const half* Aptr = a + warpM * 16 * k + t;
        const half* Bptr = b + warpN * 16 + t * w;
        wmma::load_matrix_sync(a_frag, Aptr, k);
        wmma::load_matrix_sync(b_frag, Bptr, w);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* Cptr = c + warpM * 16 * w + warpN * 16;
    wmma::store_matrix_sync(Cptr, c_frag, w, wmma::mem_row_major);
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_16f &a, const gpu::gpu_mem_16f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(reinterpret_cast<const half*>(a.cuptr()),
    reinterpret_cast<const half*>(b.cuptr()), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda

