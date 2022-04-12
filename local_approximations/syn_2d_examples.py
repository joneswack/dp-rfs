import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.io import loadmat

from models.het_gp import HeteroskedasticGP, predictive_dist_exact
from random_features.gaussian_approximator import GaussianApproximator
from util.helper_functions import classification_scores, regression_scores

from util.kernels import gaussian_kernel
from util.helper_functions import optimize_marginal_likelihood

from util.measures import Exponential_Measure

save_name = 'sinc_2d_example'
D = 100 # number of RFs
# we need more points for high frequency
n_points = [100, 100, 100] # 50
frequencies = [3, 1, 0.5]

noise_var = 0.01
seed = 8
lr = 1e-2
num_its = 50

start = -3
end = 3

configs = [
    {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'hierarchical': False, 'complex_weights': False},
    # {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True},
    {'method': 'maclaurin', 'proj': 'srht', 'degree': 10, 'hierarchical': False, 'complex_weights': True},
    # {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 10, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True}
]

params = {
    'legend.fontsize': 'medium',
    'figure.figsize': (4*len(frequencies), 3*(len(configs)+1)), # 2.2*len(csvs)
    'axes.labelsize': 'medium',
    'axes.titlesize':'medium',
    'xtick.labelsize':'large',
    'ytick.labelsize':'large'
}
pylab.rcParams.update(params)

def generate_sinosoids(fn, n, noise_var, seed):
    np.random.seed(seed=seed)

    # Generate sample data (0-centered)
    # X = np.random.uniform(low=-1.5, high=1.5, size=(n, 1))
    X = np.random.uniform(low=start, high=end, size=(n, 2))
    y = fn(X[:,0], X[:,1])
    y += np.random.normal(loc=0, scale=np.sqrt(noise_var), size=y.shape)  # add noise

    return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)


if __name__ == '__main__':

    fig1, f1_axes = plt.subplots(ncols=len(frequencies), nrows=len(configs)+1)

    for i, freq in enumerate(frequencies):
        fn = lambda x, y: 0.5 * np.sinc(freq * x) + 0.5 * np.sinc(freq * y)

        train_data, train_labels = generate_sinosoids(fn, n_points[i], noise_var, seed)
        train_data = train_data.float()
        train_labels = train_labels.float()

        num_points = 30

        a = torch.linspace(start, end, num_points)
        b = torch.linspace(start, end, num_points)
        test_data = torch.stack(torch.meshgrid(a, b), dim=-1).view(-1,2)
        test_labels = fn(test_data[:,0], test_data[:,1])

        log_noise_var = torch.nn.Parameter((torch.ones(1) * noise_var).log(), requires_grad=False)
        log_lengthscale = torch.nn.Parameter(torch.cdist(train_data, train_data).median().log(), requires_grad=True)
        log_var = torch.nn.Parameter((torch.ones(1) * train_labels.var()), requires_grad=True)

        # kernel_fun = lambda x, y: rbf_kernel(x, y, lengthscale=1.)
        kernel_fun = lambda x, y: log_var.exp() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp())
        optimize_marginal_likelihood(
            train_data, train_labels, kernel_fun, log_lengthscale, log_var, log_noise_var, num_iterations=num_its, lr=lr
        )

        print('Lengthscale:', log_lengthscale.exp().item())
        print('Kernel var:', log_var.exp().item())
        print('Noise var:', noise_var)
    
        # plt.hist((train_data @ train_data.t() / log_lengthscale.data.exp().pow(2)).view(-1))
        # plt.show()

        kernel_fun = lambda x, y: log_var.exp().item() * gaussian_kernel(x, y, lengthscale=log_lengthscale.exp().item())
        f_test_mean_ref, f_test_stds_ref = predictive_dist_exact(
            train_data, test_data, train_labels, noise_var * torch.ones_like(train_labels), kernel_fun
        )

        label_range = f_test_mean_ref.max() - f_test_mean_ref.min()
        level_space = np.linspace(f_test_mean_ref.min() - 0.2 * label_range, f_test_mean_ref.max() + 0.2 * label_range, 15)

        cur_ax = f1_axes[0, i]

        cntr1 = cur_ax.contourf(
            test_data[:,0].reshape(num_points,num_points),
            test_data[:,1].reshape(num_points,num_points),
            f_test_mean_ref.reshape(num_points,num_points),
            levels=level_space,
            cmap="RdBu_r"
        )
        cur_ax.plot(train_data[:,0], train_data[:,1], 'ko', ms=1.5)
        fig1.colorbar(cntr1, ax=cur_ax)

        title = '$y=sinc({} x)$'.format(frequencies[i]) + '\n'
        title += '$l={:.2f}, '.format(log_lengthscale.exp().item())
        title += '\\sigma^2={:.2f}, '.format(log_var.exp().item())
        title += '\\sigma^2_{{noise}} = {:.2f}$'.format(noise_var)
        cur_ax.set_title(title)
        cur_ax.set_xlim(start, end)
        cur_ax.set_ylim(start, end)


        for j, config in enumerate(configs):
            feature_encoder = GaussianApproximator(train_data.shape[1], D,
                                                approx_degree=config['degree'],
                                                lengthscale=1., var=1.,
                                                trainable_kernel=False,
                                                method=config['method'], projection_type=config['proj'], hierarchical=config['hierarchical'],
                                                complex_weights=config['complex_weights'])

            feature_encoder.log_lengthscale.data = log_lengthscale.data
            feature_encoder.log_var.data = log_var.data

            if config['method'] == 'maclaurin':
                feature_encoder.initialize_sampling_distribution(train_data)
                # feature_encoder.feature_encoder.measure = Exponential_Measure(True)
                # feature_encoder.feature_encoder.measure.distribution = np.array(D * [1])
                # feature_dist = feature_encoder.feature_encoder.measure.distribution

                feature_encoder.resample()

                # solve one GP per test input
                test_means = []
                test_stds = []
                for test_point in test_data:
                    train_features = feature_encoder.forward(train_data - test_point) #  
                    test_features = feature_encoder.forward(torch.zeros_like(test_point).unsqueeze(0)) # 

                    het_gp = HeteroskedasticGP(None)

                    f_test_mean, f_test_stds = het_gp.predictive_dist(
                        train_features, test_features,
                        train_labels, noise_var * torch.ones_like(train_labels)
                    )

                    test_means.append(f_test_mean)
                    test_stds.append(f_test_stds)

                f_test_mean = torch.hstack(test_means)[0]
                f_test_stds = torch.hstack(test_stds)[0]
                    
            else:
                feature_dist = None
                feature_encoder.resample()
                train_features = feature_encoder.forward(train_data)
                test_features = feature_encoder.forward(test_data)

                het_gp = HeteroskedasticGP(None)

                f_test_mean, f_test_stds = het_gp.predictive_dist(
                    train_features, test_features,
                    train_labels, noise_var * torch.ones_like(train_labels)
                )

            cur_ax = f1_axes[j+1, i]

            cntr1 = cur_ax.contourf(
                test_data[:,0].reshape(num_points,num_points),
                test_data[:,1].reshape(num_points,num_points),
                f_test_mean.reshape(num_points,num_points),
                levels=level_space,
                cmap="RdBu_r"
            )
            cur_ax.plot(train_data[:,0], train_data[:,1], 'ko', ms=1.5)
            if config['method'] == 'maclaurin':
                cur_ax.set_title(str(feature_encoder.feature_encoder.measure.distribution))
            fig1.colorbar(cntr1, ax=cur_ax)

            if i == 0:
                cur_ax.set_ylabel('$\\bf{{{}}}$'.format(config['method']))

            cur_ax.set_xlim(start, end)
            cur_ax.set_ylim(start, end)

        # test_error, test_mnll = regression_scores(f_test_mean, f_test_stds**2 + noise_var, test_labels)

    plt.tight_layout()
    handles, labels = f1_axes[0,0].get_legend_handles_labels()
    legend = plt.figlegend(handles=handles, labels=labels, loc='lower center', ncol=5, bbox_to_anchor = (0,-0.075,1.03,1.0), bbox_transform = plt.gcf().transFigure)

    #plt.savefig('figures/{}.pdf'.format(save_name), dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()