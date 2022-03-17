import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.io import loadmat

from models.het_gp import HeteroskedasticGP, predictive_dist_exact
from random_features.gaussian_approximator import GaussianApproximator
from util.helper_functions import classification_scores, regression_scores

from util.kernels import gaussian_kernel
from util.helper_functions import cholesky_solve

from util.LBFGS import FullBatchLBFGS

save_name = 'sinc_1d_example'
D = 10 # number of RFs
# we need more points for high frequency
n_points = [15, 25, 10] # 50
frequencies = [5, 2, 0.5]

noise_var = 0.01
seed = 8
lr = 1e-2
num_its = 100

start = -1.5
end = 1.5

configs = [
    {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'hierarchical': False, 'complex_weights': False},
    # {'method': 'rff', 'proj': 'gaussian', 'degree': 4, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True},
    {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 10, 'hierarchical': False, 'complex_weights': False},
    # {'method': 'maclaurin', 'proj': 'rademacher', 'degree': 10, 'bias': 0, 'lengthscale': True, 'hierarchical': False, 'complex_weights': True}
]

params = {
    'legend.fontsize': 'medium',
    'figure.figsize': (3*len(frequencies), 4), # 2.2*len(csvs)
    'axes.labelsize': 'medium',
    'axes.titlesize':'medium',
    'xtick.labelsize':'large',
    'ytick.labelsize':'large'
}
pylab.rcParams.update(params)

def load_snelson():
    data = loadmat('snelson1d.mat')
    Xtrain = torch.from_numpy(data['X'])
    Ytrain = torch.from_numpy(data['Y'])

    return Xtrain, Ytrain

def generate_sinosoids(fn, n, noise_var, seed):
    np.random.seed(seed=seed)

    # Generate sample data (0-centered)
    # X = np.random.uniform(low=-1.5, high=1.5, size=(n, 1))
    X = np.random.uniform(low=start, high=end, size=(n, 1))
    y = fn(X)
    y += np.random.normal(loc=0, scale=np.sqrt(noise_var), size=y.shape)  # add noise

    return torch.from_numpy(X), torch.from_numpy(y)

def plot_app_vs_full_with_div(ax, i, j, test_inputs, test_labels, original_inputs, original_labels,
                                app_mean, app_stds, full_mean, full_stds, lengthscale, var, noise_var, feature_dist):

    test_inputs = test_inputs.view(-1).numpy()
    original_inputs = original_inputs.view(-1).numpy()
    original_labels = original_labels.view(-1).numpy()

    app_mean = app_mean.view(-1).numpy()
    app_stds = app_stds.view(-1).numpy()
    full_mean = full_mean.view(-1).numpy()
    full_stds = full_stds.view(-1).numpy()

    # Training points
    ax.plot(original_inputs, original_labels, 'ko', markersize=3, label='Training Points')

    # Approx. GP
    ax.plot(test_inputs, app_mean, '--', lw=2, color='royalblue', label=r'Approx. GP $\mu$')
    ax.plot(test_inputs, app_mean-2*app_stds, color='royalblue', lw=1,
                    alpha=1, label=r'Approx. GP $\mu \pm 2 \sigma$')
    ax.plot(test_inputs, app_mean+2*app_stds, color='royalblue', lw=1,
                    alpha=1)
    # ax.plot(test_inputs, test_labels, 'b-', lw=2, label='Target function', alpha=0.1)

    # Full GP
    ax.plot(test_inputs, full_mean, 'k-', lw=2, label=r'Full GP $\mu$', alpha=0.75)
    ax.fill_between(test_inputs, full_mean-2*full_stds, full_mean+2*full_stds, color='grey', 
                    alpha=0.25, label=r'Full GP $\mu \pm 2 \sigma$')

    ax.set_xlabel(r'$x$')
    ax.set_xlim(1.5*start, 1.5*end)

    # if feature_dist is not None:
        # title += '\n' + 'Feature dist.: {}'.format(feature_dist)

    if j == 0:
        title = '$y=sinc({} x)$'.format(frequencies[i]) + '\n'
        title += '$l={:.2f}, '.format(lengthscale)
        title += '\\sigma^2={:.2f}, '.format(var)
        title += '\\sigma^2_{{noise}} = {:.2f}$'.format(noise_var)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        # ax.legend(loc='lower right')

    if i == 0:
        if j == 0:
            ax.set_ylabel('$\\bf{{RFF \\, Gaussian}}$')
        else:
            ax.set_ylabel('$\\bf{{Maclaurin \\, Radem.}}$')



def exact_marginal_log_likelihood(kernel_train, training_labels, log_noise_var):
    n = len(training_labels)
    L_train = torch.cholesky(kernel_train + torch.exp(log_noise_var) * torch.eye(n, dtype=torch.float), upper=False)
    alpha = cholesky_solve(training_labels, L_train)
    mll = -0.5 * training_labels.t().mm(alpha) - L_train.diagonal().log().sum() - (n / 2) * np.log(2*np.pi)

    return mll

def optimize_marginal_likelihood(training_data, training_labels, kernel_fun, log_lengthscale, log_var, log_noise_var, num_iterations=10, lr=1e-3):
    trainable_params = [log_lengthscale, log_var, log_noise_var]

    for iteration in range(num_iterations):
        print('### Iteration {} ###'.format(iteration))
        optimizer = FullBatchLBFGS(trainable_params, lr=lr, history_size=10, line_search='Wolfe')

        def closure():
            optimizer.zero_grad()

            kernel_train = kernel_fun(training_data, training_data)
            loss = - exact_marginal_log_likelihood(kernel_train, training_labels, log_noise_var)
            print('Loss: {}'.format(loss.item()))

            return loss

        loss = closure()
        loss.backward()
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)


if __name__ == '__main__':

    # print('Loading Snelson dataset')
    # train_data, train_labels = load_snelson()

    # train_size = int(0.5 * len(train_labels))
    # perm = torch.randperm(len(train_labels))
    # train_idxs = perm[:train_size]
    # train_data = train_data[train_idxs]
    # train_labels = train_labels[train_idxs]

    # filter out center points between 2.5 and 3.5
    # indices = (train_data.view(-1) <= 2.0) | (train_data.view(-1) >= 4.0)
    # train_data = train_data[indices]
    # train_labels = train_labels[indices]

    # train_data = (train_data - train_data.mean(0)) #/ train_data.std()
    # train_labels = train_labels - train_labels.mean(0)

    fig1, f1_axes = plt.subplots(ncols=len(frequencies), nrows=len(configs))

    for i, freq in enumerate(frequencies):
        fn = lambda x: np.sinc(freq * x)

        train_data, train_labels = generate_sinosoids(fn, n_points[i], noise_var, seed)
        train_data = train_data.float()
        train_labels = train_labels.float()

        x_delta = torch.abs(train_data.max() - train_data.min())
        # we have 200 training points, therefore we fill twice the space with 400 test points
        test_data = torch.linspace(
            # start=train_data.min() - 0.5 * x_delta, end=train_data.max() + 0.5 * x_delta,
            start = start*1.5, end=end*1.5,
            steps=400, dtype=torch.float
        ).view(-1, 1)
        test_labels = fn(test_data)

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
                # feature_encoder.initialize_sampling_distribution(train_data)
                feature_encoder.feature_encoder.measure.distribution = np.array(D * [1])
                feature_dist = feature_encoder.feature_encoder.measure.distribution

                feature_encoder.resample()

                # solve one GP per test input
                test_means = []
                test_stds = []
                for test_point in test_data:
                    train_features = feature_encoder.forward(train_data - test_point)
                    test_features = feature_encoder.forward(torch.zeros(1, 1, dtype=test_point.dtype))

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

            cur_ax = f1_axes[j, i]

            plot_app_vs_full_with_div(
                cur_ax, i, j,
                test_data, test_labels,
                train_data, train_labels,
                f_test_mean, f_test_stds,
                f_test_mean_ref, f_test_stds_ref,
                log_lengthscale.exp().item(),
                log_var.exp().item(),
                noise_var, feature_dist
            )

            # test_error, test_mnll = regression_scores(f_test_mean, f_test_stds**2 + noise_var, test_labels)

    plt.tight_layout()
    handles, labels = f1_axes[0,0].get_legend_handles_labels()
    legend = plt.figlegend(handles=handles, labels=labels, loc='lower center', ncol=5, bbox_to_anchor = (0,-0.075,1.03,1.0), bbox_transform = plt.gcf().transFigure)

    plt.savefig('figures/{}.pdf'.format(save_name), dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()