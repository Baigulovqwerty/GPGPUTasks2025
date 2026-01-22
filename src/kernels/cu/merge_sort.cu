#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const unsigned int array_pos = i >> sorted_k;
    const unsigned int friend_array_pos = array_pos ^ 1;

    const unsigned int arr_len = 1 << sorted_k;
    const unsigned int array_start = array_pos * arr_len;
    const unsigned int start = friend_array_pos * arr_len;

    if (arr_len == n || start >= n) {
        output_data[i] = input_data[i];
        return;
    }

    int l = start - 1;
    int r = start + arr_len < n ? start + arr_len : n;

    while (r - l > 1) {
        int m = l + (r - l) / 2;
        
        if ((input_data[i] < input_data[m]) || ((input_data[i] == input_data[m]) && (friend_array_pos & 1))) {
            r = m;
        } else {
            l = m;
        }
    }

    unsigned int ii = i % arr_len;
    unsigned int rr = r - start;
    unsigned int new_arr_start = friend_array_pos % 2 == 0 ? start : array_start;

    output_data[new_arr_start + ii + rr] = input_data[i];
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
