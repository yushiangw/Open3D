// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

void CumSum(benchmark::State& state, const Device& device) {
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};
    Tensor src(shape, Dtype::Int64, device);
    Tensor warm_up = src.CumSum(1);
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = src.CumSum(1);
    }
}

BENCHMARK_CAPTURE(CumSum, CPU, Device("CPU:0"))->Unit(benchmark::kMillisecond);

// TODO: CUDA.
//#ifdef BUILD_CUDA_MODULE
// BENCHMARK_CAPTURE(CumSum, CUDA, Device("CUDA:0"))
//        ->Unit(benchmark::kMillisecond);
//#endif

}  // namespace core
}  // namespace open3d