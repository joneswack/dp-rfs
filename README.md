# Random Features for Polynomial Kernels and Dot Product Kernels

This repository contains PyTorch implementations of various random feature maps for polynomial and general dot product kernels. In particular, we provide implementations using complex random projections that can improve the kernel approximation significantly. For dot product kernels, we provide an algorithm that minimizes the variance of the random feature approximation and can also be used in conjunction with the Gaussian kernel.

The basic building block of random features for dot product kernels are polynomial sketches that approximate the polynomial kernel of degree *p*. Such random projections can be seen as linear random projections applied to *p*-times self-tensored input features. If *p=1*, these sketches reduce to standard linear random projections.

For more information and a description of the novelties of this work, consult the associated technical reports:

```bibtex
@article{wacker2022a,
  title={Improved Random Features for Dot Product Kernels},
  author={Wacker, Jonas and Kanagawa, Motonobu and Filippone, Maurizio},
  journal={arXiv preprint arXiv:2201.08712},
  year={2022}
}
@article{wacker2022b,
  title={Complex-to-Real Random Features for Polynomial Kernels},
  author={Wacker, Jonas and Ohana, Ruben and Filippone, Maurizio},
  journal={arXiv preprint},
  year={2022}
}
```

Please cite these works if you find them useful.

In addition, this repository implements and extends works from the following related papers:

* [Random Feature Maps for Dot Product Kernels](http://proceedings.mlr.press/v22/kar12/kar12.pdf)
* [Fast and Scalable Polynomial Kernels via
Explicit Feature Maps](https://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf)
* [Spherical Random Features for polynomial kernels](https://papers.nips.cc/paper/2015/file/f7f580e11d00a75814d2ded41fe8e8fe-Paper.pdf)
* [Oblivious Sketching of High-Degree Polynomial Kernels](https://arxiv.org/abs/1909.01410)
* [Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
* [Orthogonal Random Features](https://papers.nips.cc/paper/2016/file/53adaf494dc89ef7196d73636eb2451b-Paper.pdf)
* [The Unreasonable Effectiveness of Structured
Random Orthogonal Embeddings](https://arxiv.org/pdf/1703.00864.pdf)

## Requirements

We recommend:

* Python 3.6 or higher
* PyTorch 1.8 or higher
* For GPU support: CUDA 9.0 or higher

Multiple complex operations on tensors that we use in our code have been introduced in PyTorch version 1.8.
In case you have a lower version, you cannot use the complex projections in our code.

### A note on using GPU-accelerated SRHT sketches

PyTorch does not natively support the Fast Walsh Hadamard Transform. This repository contains an implementation including a CUDA kernel in `util/hadamard_cuda`. This kernel is needed for GPU-accelerated TensorSRHT sketches.
If you want to use this implementation, you need to specify your cuda path inside `util/fwht/__init__.py` for its compilation:
```python
if torch.cuda.is_available():
    sources.extend([filedir + 'fwht_kernel.cu'])
    flags.extend(['-DIS_CUDA_AVAILABLE'])
    if os.environ.get('CUDA_HOME', '-1') == '-1':
        warnings.warn('CUDA_HOME variable not set. Setting CUDA_HOME=/usr/local/cuda-9.0...',)
        os.environ['CUDA_HOME'] = '/usr/local/cuda-9.0'
```
We used cuda version 9.0 in our experiments but other versions should work too.

## Getting started

### Polynomial sketches

A polynomial sketch can be used on its own or as a module inside a larger deep learning framework.
It is initialized as follows:

```python
from random_features.polynomial_sketch import PolynomialSketch
feature_encoder = PolynomialSketch(
    input_dimension, # data input dimension (power of 2 for srht projection_type)
    projection_dimension, # output dimension of the random sketch
    degree=degree, # degree of the polynomial kernel
    bias=bias, # bias parameter of the polynomial kernel
    lengthscale=lengthscale, # inverse scale of the data (like lengthscale for Gaussian kernel)
    projection_type='gaussian'/'rademacher'/'srht'/'countsketch_scatter',
    complex_weights=False/True, # whether to use complex random projections (without complex_real outputs are complex-valued)
    complex_real=False/True, # whether to use complex-to-real sketches (outputs are real-valued)
    hierarchical=False/True, # whether to use hierarchical sketching as proposed in <https://arxiv.org/abs/1909.01410>
    device='cpu'/'cuda', # whether to use CPU or GPU
)

feature_encoder.resample() # initialize random feature sample
feature_encoder.forward(input_data) # project input data
```

`projection_type` has a strong impact on the approximation quality and computation speed. `projection_type='srht'` uses the subsampled randomized Hadamard transform that makes use of structured matrix products. These are faster (especially on the GPU). They also give lower variances than Rademacher and Gaussian sketches most of the time. `countsketch_scatter` uses CountSketches as the base projection to yield TensorSketch (<https://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf>).

Depending on the scaling of your data, high-degree sketches can give very large variances. `complex_weights` usually improve the approximation significantly in this case. `hierarchical` sketches (<https://arxiv.org/abs/1909.01410>) are not the focus of this work, but can be helpful too.
Complex random features can be computed about as fast as real ones, in particular when using `projection_type='srht'`. However, the downstream task is usually more expensive when working with complex data. Complex-to-Real (CtR) (`complex_real=True`) sketches return real-valued random features instead and thus alleviate this problem.

### Spherical Random Features (SRF)

We also provide an implementation of [Spherical Random Features for polynomial kernels](https://papers.nips.cc/paper/2015/file/f7f580e11d00a75814d2ded41fe8e8fe-Paper.pdf). They can be run similar to the above code but the random feature distribution needs to be optimized first. Please have a look at the code at the bottom of `random_features/spherical.py` to understand the procedure for obtaining this distribution and the random feature map afterwards. Once the distribution is obtained, it is saved under `saved_models` (some parameterizations are contained in this repository already).

### Approximating the Gaussian kernel using polynomial sketches

The Gaussian kernel can be approximated well through a randomized Maclaurin expansion with polynomial sketches assuming that the data is zero-centered and a proper lengthscale is used.

Note: This method is not implemented for Complex-to-Real (CtR) sketches yet.

The approximator is initialized as follows:

```python

from random_features.gaussian_approximator import GaussianApproximator
feature_encoder = GaussianApproximator(
    input_dimension, # data input dimension (power of 2 for srht projection_type)
    projection_dimension, # output dimension of the random sketch
    approx_degree=10, # maximum degree of the taylor approximation
    lengthscale=lengthscale, # lengthscale of the Gaussian kernel
    var=kernel_var, # variance of the Gaussian kernel
    method='maclaurin', # approximation method (other possibilities: rff/poly_sketch)
    projection_type='gaussian'/'rademacher'/'srht'/'countsketch_scatter',
    hierarchical=False/True,
    complex_weights=False/True
)

# find optimal sampling distribution when using optimized maclaurin
feature_encoder.initialize_sampling_distribution(input_data_sample, min_sampling_degree=2)
feature_encoder.resample() # initialize random feature sample
feature_encoder.forward(input_data) # project input data
```

Here, we chose the optimized Maclaurin method (`method='maclaurin'`).
`initialize_sampling_distribution` finds the optimal random feature distribtion between degrees `min_sampling_degree=2` and `approx_degree=10`. `method='rff'` (random Fourier features) and `method='poly_sketch'` do not require this step.
`method='maclaurin_p'` gives the Maclaurin approximation according to <http://proceedings.mlr.press/v22/kar12/kar12.pdf>, which is less optimal than the optimized Maclaurin method but requires no preprocessing.

## Reproducing the Gaussian Process Classification/Regression experiments

If you would like to reproduce the results from the experiments in our paper, download the desired datasets from the [UCI machine learning repository](https://archive.ics.uci.edu/). The code_rna dataset is available at <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html>. Then load the associated CSV files and save them using PyTorch. The dataset paths are in `config/datasets` and the datasets to be used in the experiments are in `config/active_datasets.json`.

Then run the following command:

```sh
python run_rf_gp_experiments.py --rf_parameter_file [rf_parameter_config] --datasets_file config/active_datasets.json --use_gpu
```

`--use_gpu` is optional. `[rf_parameter_config]` needs to be replaced by one of the files located in config/rf_parameters depending on which kernel should be approximated. These files can be easily adapted according to your needs.

The output logs of the experiments are saved in the csv and logs folder.

The bar plots can be created from the csv files using the Jupyter notebook `notebooks/bar-plot-visualization.ipynb`.
