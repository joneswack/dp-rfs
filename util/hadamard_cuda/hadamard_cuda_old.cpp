/*  Copyright (c) 2019
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation, either version 3 of the License, or (at your
 *  option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *  Authors:
 *      Simone Rossi <simone.rossi@eurecom.fr>
 *      Maurizio Filippone <maurizio.filippone@eurecom.fr>
 */

#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>

#define _OPENMP
#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
void forward_fwht_vector_cpu(scalar_t *output, int log2N)
{
    const long int N = 1 << log2N;

    //Cycle through stages with different butterfly strides
    for (int stride = N / 2; stride >= 1; stride >>= 1)
    {
        //Cycle through subvectors of (2 * stride) elements
        for (int base = 0; base < N; base += 2 * stride)

            //Butterfly index within subvector of (2 * stride) size
            for (int j = 0; j < stride; j++)
            {
                int i0 = base + j + 0;
                int i1 = base + j + stride;

                auto T1 = output[i0];
                auto T2 = output[i1];
                output[i0] = T1 + T2;
                output[i1] = T1 - T2;
            }
    }
}

at::Tensor forward_fwht_cpu(at::Tensor input)
{
    auto shape = input.sizes();
    AT_CHECK(shape.size() == 2, "Input tensor should be 2D. Please, reshape it as 2D.\n")
    int batch_size = shape[0];
    int log2d = (int)log2(shape[1]);
    AT_CHECK((float)log2d == log2(shape[1]), "Number of input feature has to be an integer power of 2.\n")
    at::Tensor output = input.clone();
    int64_t grain_size = 1;
    
    at::parallel_for(0, batch_size, grain_size, [&](int64_t b_begin, int64_t b_end) {
        for (int i = b_begin; i < b_end; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(input.type(), "fwtBatchCPU", ([&] {
                forward_fwht_vector_cpu(output[i].data<scalar_t>(), log2d);
                }));
        }
    });
    return output;
}

#ifdef IS_CUDA_AVAILABLE
//at::Tensor forward_fwht_cuda(at::Tensor output);
void forward_fwht_cuda(float* output, int batchSize, int log2N);

at::Tensor forward_fwht_gpu(at::Tensor input)
{
    AT_CHECK(input.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    auto n = input.size(-1);
    auto log2N = long(log2(n));
    AT_CHECK(n == 1 << log2N, "n must be a power of 2");
    auto output = input.clone();  // Cloning makes it contiguous.
    auto batchSize = input.numel() / (1 << log2N);
    forward_fwht_cuda(output.data<float>(), batchSize, log2N); // , batchSize, log2N .data_ptr<float>()
    return output;
}
#endif

at::Tensor forward_fwht(at::Tensor input){
    if (input.type().is_cuda())
#ifdef IS_CUDA_AVAILABLE
        return forward_fwht_gpu(input);
#else
        AT_ERROR("FWHT is not compiled with CUDA but the input tensor is on GPU\n");
#endif
    return forward_fwht_cpu(input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hadamard_transform", &forward_fwht, "Implements the Fast Walsh Hadamard transform on CPU or GPU (if available)");
}

/*
at::Tensor fwtBatchGPU(at::Tensor d_Data);

torch::Tensor hadamard_transform(torch::Tensor x) {
  AT_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
  auto n = x.size(-1);
  auto log2N = long(log2(n));
  AT_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = x.clone();  // Cloning makes it contiguous.
  auto batchSize = x.numel() / (1 << log2N);
  return fwtBatchGPU(output); // , batchSize, log2N .data_ptr<float>()
  // return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_transform", &hadamard_transform, "Fast Hadamard transform");
}
*/
