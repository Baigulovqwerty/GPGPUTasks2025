#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void prefix_sum_01_precount(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &precount, unsigned int n, unsigned int k);
void prefix_sum_02_final(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &precount, gpu::gpu_mem_32u &ans, unsigned int n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}
