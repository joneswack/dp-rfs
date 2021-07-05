# Beyond Random Fourier Features: Improved Random Sketches for Dot Product Kernels

This repository contains PyTorch implementations of various random feature maps for polynomial and general dot product kernels. In particular, we provide implementations for complex-valued features that improve the kernel approximation significantly.
Furthermore, it allows to reproduce the Gaussian Process experiments from the associated paper: ...

The basic building block of random features for dot product kernels are polynomial sketches that approximate the polynomial kernel of degree $p$. Such random projections can be seen as an extension of the Johnson-Lindenstrauss Lemma to degree-$p$ tensored versions of the input feature space and are therefore quite general. If $p=1$, these sketches reduce to linear random projections. Furthermore, they can be used to approximate the Gaussian kernel via a truncated Taylor series expansion.

## Requirements

We recommend:

* Python 3.6 or higher
* PyTorch 1.8 or higher

Multiple complex operations on tensors that we use in our code have been introduced in PyTorch version 1.8.
In case you have a lower version, you cannot use the complex projections in our code.

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
    projection_type='gaussian'/'rademacher'/'srht'/'countsketch_scatter',
    hierarchical=False/True,
    complex_weights=False/True
)

feature_encoder.resample() # initialize random feature sample
feature_encoder.cuda() # only for GPU
feature_encoder.move_submodules_to_cuda() # only for GPU
feature_encoder.forward(input_data) # project input data
```

`projection_type` has a strong impact on the approximation quality and computation speed. `projection_type=srht` uses the subsampled randomized Hadamard transform that makes use of structured matrix products. These are faster (especially on the GPU). They also give lower variances for odd degrees than Rademacher and Gaussian sketches.

Depending on the scaling of your data, high-degree sketches can give very large variances. `complex_weights` usually improve the approximation significantly in this case. `hierarchical` sketches <https://arxiv.org/abs/1909.01410> can be helpful too.
Both come at a higher computational cost for the downstream task.

At the bottom of `random_features/polynomial_sketch.py`, we show how to evaluate the unbiasedness of the approximation. You can directly run the script.

### Spherical Random Features (SRF)

We also provide an implementation of [Spherical Random Features for polynomial kernels](https://papers.nips.cc/paper/2015/file/f7f580e11d00a75814d2ded41fe8e8fe-Paper.pdf). They can be run similar to the above code but the random feature distribution needs to be optimized first. Please have a look at the code at the bottom of `random_features/spherical.py` to understand the procedure for obtaining this distribution and the random feature map afterwards. Once the distribution is obtained, it is saved under `saved_models` (some parameterizations are contained in this repository already).

### Approximating the Gaussian kernel using polynomial sketches

The Gaussian kernel can be approximated well through a randomized Maclaurin expansion with polynomial sketches assuming that the data is zero-centered and a proper lengthscale is used.

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

# find optimal sampling distribution using optimized maclaurin
feature_encoder.initialize_sampling_distribution(input_data_sample, min_sampling_degree=2)

feature_encoder.resample() # initialize random feature sample
feature_encoder.feature_encoder.cuda() # only for GPU
feature_encoder.feature_encoder.move_submodules_to_cuda() # only for GPU
feature_encoder.forward(input_data) # project input data
```

Here, we chose the optimized Maclaurin method (`method='maclaurin'`).
`initialize_sampling_distribution` finds the optimal random feature distribtion between degrees `min_sampling_degree=2` and `approx_degree=10`. `method='rff'` (random Fourier features) and `method='poly_sketch'` do not require this step.
`method='maclaurin_p'` gives the Maclaurin approximation according to <http://proceedings.mlr.press/v22/kar12/kar12.pdf> if benchmarking is desired.

## Reproducing the Gaussian Process Classification/Regression experiments

If you would like to reproduce the results from the experiments in our paper, run the following command:

```sh
python run_rf_gp_experiments.py --rf_parameter_file [rf_parameter_config] --datasets_file config/active_datasets.json --use_gpu
```

`--use_gpu` is optional. `[rf_parameter_config]` needs to be replaced by one of the files located in config/rf_parameters depending on whether the Gaussian or the degree 20 polynomial kernel should be approximated. These files can be easily adapted according to your needs.

The output logs of the experiments are saved in the csv and logs folder.

The bar plots can be created from the csv files using the Jupyter notebook `notebooks/bar-plot-visualization.ipynb`.
