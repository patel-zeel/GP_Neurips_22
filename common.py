import os

os.environ["LATEXIFY"] = ""
os.environ["FIG_DIR"] = "."
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from time import time

from tinygp import GaussianProcess, kernels
import jaxopt

from mpl_toolkits import mplot3d

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import optax

import pandas as pd

from scipy.io import loadmat

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from gpax import ExactGP, GibbsKernel, HeteroscedasticNoise, HomoscedasticNoise
from gpax.utils import constrain, unconstrain, randomize

import lab.jax as B

import regdata as rd

# from utils import get_folds, get_gibbs_gp, initialize_params, train_fn, get_loss_fn

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib as mpl

from probml_utils import latexify, savefig

# from jax.config import config
# config.update("jax_debug_nans", True)


def get_folds(X, y, seed):
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    return [
        (X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        for train_idx, test_idx in splitter.split(X, y)
    ]


def get_inducing(X, num_inducing, type, key=jax.random.PRNGKey(0)):
    if type == "random":
        return jax.random.uniform(
            key, shape=(num_inducing, X.shape[1]), minval=X.min(), maxval=X.max()
        )
    elif type == "unispace":
        return jnp.linspace(X.min(), X.max(), num_inducing)
    else:
        raise ValueError("Inducing type not recognized")


def get_gibbs_gp(
    X, flex_scale, flex_var, flex_noise, num_inducing, inducing_type="random"
):
    X_inducing = get_inducing(X, num_inducing, inducing_type)
    kernel = GibbsKernel(
        flex_scale=flex_scale, flex_variance=flex_var, X_inducing=X_inducing
    )
    if flex_noise:
        noise = HeteroscedasticNoise()
    else:
        noise = HomoscedasticNoise()

    gp = ExactGP(kernel=kernel, noise=noise)
    return gp


def initialize_params(gp, X, key, scale_prior=None, var_prior=None, std_prior=None):
    params = gp.initialise_params(key, X=X, X_inducing=gp.kernel.X_inducing)

    #     print(jtu.tree_structure(params))
    #     print(jtu.tree_structure(gp.get_bijectors(), is_leaf=lambda x:isinstance(x, tfb.Bijector)))

    raw_params = unconstrain(params, gp.get_bijectors())
    raw_params = randomize(raw_params, key)

    if gp.kernel.flex_scale:
        raw_params["kernel"]["X_inducing"] = gp.kernel.X_inducing
        if std_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["kernel"]["inducing_std_scale"] = std_prior.sample(
                seed=key, sample_shape=raw_params["kernel"]["inducing_std_scale"].shape
            )
        if scale_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["kernel"]["scale_gp"]["kernel"][
                "lengthscale"
            ] = scale_prior.sample(
                seed=key,
                sample_shape=raw_params["kernel"]["scale_gp"]["kernel"][
                    "lengthscale"
                ].shape,
            )
        if var_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["kernel"]["scale_gp"]["kernel"]["variance"] = var_prior.sample(
                seed=key,
                sample_shape=raw_params["kernel"]["scale_gp"]["kernel"][
                    "variance"
                ].shape,
            )
        raw_params["kernel"]["scale_gp"]["noise"]["variance"] = jnp.sign(
            jnp.abs(raw_params["kernel"]["scale_gp"]["noise"]["variance"])
        ) * jnp.log(B.epsilon)

    if gp.kernel.flex_variance:
        raw_params["kernel"]["X_inducing"] = gp.kernel.X_inducing
        if std_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["kernel"]["inducing_std_variance"] = std_prior.sample(
                seed=key,
                sample_shape=raw_params["kernel"]["inducing_std_variance"].shape,
            )
        if scale_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["kernel"]["variance_gp"]["kernel"][
                "lengthscale"
            ] = scale_prior.sample(
                seed=key,
                sample_shape=raw_params["kernel"]["variance_gp"]["kernel"][
                    "lengthscale"
                ].shape,
            )
        if var_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["kernel"]["variance_gp"]["kernel"][
                "variance"
            ] = var_prior.sample(
                seed=key,
                sample_shape=raw_params["kernel"]["variance_gp"]["kernel"][
                    "variance"
                ].shape,
            )
        raw_params["kernel"]["variance_gp"]["noise"]["variance"] = jnp.sign(
            jnp.abs(raw_params["kernel"]["variance_gp"]["noise"]["variance"])
        ) * jnp.log(B.epsilon)
    if gp.noise.__class__.__name__ == "HeteroscedasticNoise":
        raw_params["kernel"]["X_inducing"] = gp.kernel.X_inducing
        if std_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["noise"]["inducing_std_noise"] = std_prior.sample(
                seed=key, sample_shape=raw_params["noise"]["inducing_std_noise"].shape
            )
        if scale_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["noise"]["noise_gp"]["kernel"][
                "lengthscale"
            ] = scale_prior.sample(
                seed=key,
                sample_shape=raw_params["noise"]["noise_gp"]["kernel"][
                    "lengthscale"
                ].shape,
            )
        if var_prior is not None:
            key = jax.random.split(key, num=1)[0]
            raw_params["noise"]["noise_gp"]["kernel"]["variance"] = var_prior.sample(
                seed=key,
                sample_shape=raw_params["noise"]["noise_gp"]["kernel"][
                    "variance"
                ].shape,
            )
        raw_params["noise"]["noise_gp"]["noise"]["variance"] = jnp.sign(
            jnp.abs(raw_params["noise"]["noise_gp"]["noise"]["variance"])
        ) * jnp.log(B.epsilon)
    return raw_params


def get_loss_fn(gp, X, y, scale_prior=None, var_prior=None, std_prior=None):
    bijectors = gp.get_bijectors()

    def loss_fn(raw_params):
        loss = 0.0
        if gp.kernel.flex_scale:
            if std_prior is not None:
                loss -= std_prior.log_prob(
                    raw_params["kernel"]["inducing_std_scale"]
                ).sum()
            if scale_prior is not None:
                loss -= scale_prior.log_prob(
                    raw_params["kernel"]["scale_gp"]["kernel"]["lengthscale"]
                ).sum()
            if var_prior is not None:
                loss -= var_prior.log_prob(
                    raw_params["kernel"]["scale_gp"]["kernel"]["variance"]
                ).sum()
            raw_params["kernel"]["scale_gp"]["noise"][
                "variance"
            ] = jax.lax.stop_gradient(
                raw_params["kernel"]["scale_gp"]["noise"]["variance"]
            )
        if gp.kernel.flex_variance:
            if std_prior is not None:
                loss -= std_prior.log_prob(
                    raw_params["kernel"]["inducing_std_variance"]
                ).sum()
            if scale_prior is not None:
                loss -= scale_prior.log_prob(
                    raw_params["kernel"]["variance_gp"]["kernel"]["lengthscale"]
                ).sum()
            if var_prior is not None:
                loss -= var_prior.log_prob(
                    raw_params["kernel"]["variance_gp"]["kernel"]["variance"]
                ).sum()
            raw_params["kernel"]["variance_gp"]["noise"][
                "variance"
            ] = jax.lax.stop_gradient(
                raw_params["kernel"]["variance_gp"]["noise"]["variance"]
            )
        if gp.noise.__class__.__name__ == "HeteroscedasticNoise":
            if std_prior is not None:
                loss -= std_prior.log_prob(
                    raw_params["noise"]["inducing_std_noise"]
                ).sum()
            if scale_prior is not None:
                loss -= scale_prior.log_prob(
                    raw_params["noise"]["noise_gp"]["kernel"]["lengthscale"]
                ).sum()
            if var_prior is not None:
                loss -= var_prior.log_prob(
                    raw_params["noise"]["noise_gp"]["kernel"]["variance"]
                ).sum()
            raw_params["noise"]["noise_gp"]["noise"][
                "variance"
            ] = jax.lax.stop_gradient(
                raw_params["noise"]["noise_gp"]["noise"]["variance"]
            )
        #         if flex_scale or flex_var or (gp.noise.__class__.__name__ == "HeteroscedasticNoise"):
        #             loss -= std_prior.log_prob(raw_params["kernel"]["X_inducing"]).sum()

        params = constrain(raw_params, bijectors)
        loss -= gp.log_probability(params, X, y)
        return loss

    return loss_fn


def train_fn(loss_fn, raw_params, optimizer, num_epochs):
    state = optimizer.init(raw_params)

    @jax.jit
    def step(params_and_state, xs):
        params, state = params_and_state
        value, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), (value, params)

    (params, state), (loss_history, params_history) = jax.lax.scan(
        step, (raw_params, state), xs=None, length=num_epochs
    )

    return {
        "params": params,
        "loss_history": loss_history,
        "params_history": params_history,
    }


def extract_results(result):
    min_idx = jnp.nanargmin(result["loss_history"], axis=1)
    print("min_indices", min_idx)
    losses = jax.vmap(lambda result, idx: result[idx])(result["loss_history"], min_idx)
    print("individual restart losses", losses)
    best_restart = jnp.nanargmin(losses)
    print("best_restart", best_restart, "best_loss", losses[best_restart])
    best_result = jax.tree_util.tree_map(lambda x: x[best_restart], result)
    best_params = jax.tree_util.tree_map(
        lambda x: x[min_idx[best_restart] - 2], best_result["params_history"]
    )
    return best_result, best_params


def nlpd_fn(pred_mean, pred_cov, y_test):
    return -tfd.MultivariateNormalFullCovariance(pred_mean.ravel(), pred_cov).log_prob(
        y_test.ravel()
    )


def msll_fn(pred_mean, pred_cov, y_test):
    return (
        -tfd.Normal(pred_mean.ravel(), jnp.diag(pred_cov) ** 0.5)
        .log_prob(y_test.ravel())
        .sum()
    )


def rmse_fn(pred_mean, y_test):
    return jnp.sqrt(jnp.mean((pred_mean.ravel() - y_test.ravel()) ** 2))


def get_step_data():
    end_points = [(-1.25, -0.74, 0), (-0.76, -0.2, 1), (0.5, 0.74, 2), (0.76, 1.5, 3)]
    n = 100
    x = jnp.concatenate(
        [
            jnp.sort(
                jax.random.uniform(
                    jax.random.PRNGKey(seed),
                    (int(n * abs((end - start))),),
                    minval=start,
                    maxval=end,
                )
            )
            for start, end, seed in end_points
        ]
    ).reshape(-1, 1)
    y = jnp.concatenate(
        [
            jnp.ones(int(n * abs((end - start)))) * jnp.sign((i % 2) - 0.5)
            for i, (start, end, seed) in enumerate(end_points)
        ]
    )
    # y += jax.random.normal(jax.random.PRNGKey(0), y.shape)*0.001

    return x, y


def get_aw_step():
    chunks = [
        (10, 0.2),
        (10, 0.4),
        (10, 0.6),
        (3, 0.4),
        (10, 0.6),
        (10, 0.8),
        (10, 1.0),
        (1, 0.4),
        (10, 0.6),
        (10, 0.8),
        (2, 0.6),
        (10, 0.8),
        (2, 0.6),
        (10, 0.0),
        (10, 0.2),
        (5, 0.0),
        (10, 0.2),
        (5, 0.0),
        (10, -0.2),
        (10, 0.0),
        (10, -0.2),
    ]
    y = jnp.concatenate([jnp.ones(i) * j for i, j in chunks])
    x = jnp.linspace(-5, 5, y.shape[0]).reshape(-1, 1)

    #     x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y[:, None])[:, 0]

    return x, y


def get_simulated_data1(seed=1221, n_samples=300):
    lengthscale_trend = lambda x: (0.5 * jnp.sin(5 * x / 8)) + 1.0
    variance_trend = lambda x: jnp.exp(jnp.sin(0.2 * x))  # (0.3 * x**2) + 0.4
    noise_var_trend = lambda x: jnp.exp(
        jnp.sin(0.2 * -x)
    )  # 0.4*jnp.exp(0.75 * jnp.sin(0.5 * x + 1))
    # 0.1 * jax.scipy.stats.norm.cdf(5 * x) + 0.1

    #     plt.plot(x, lengthscale_trend(x))
    #     plt.plot(x, variance_trend(x))
    #     plt.plot(x, noise_var_trend(x));

    def kernel_fn(x1, ls1, var1, x2, ls2, var2):
        l_sqr_avg = (ls1**2 + ls2**2) / 2
        prefix = jnp.sqrt(ls1 * ls2 / l_sqr_avg)
        exp_part = jnp.exp(-0.5 * ((x1 - x2) ** 2) / l_sqr_avg)
        return var1 * var2 * prefix * exp_part

    kernel_fn = jax.vmap(kernel_fn, in_axes=(None, None, None, 0, 0, 0))
    kernel_fn = jax.vmap(kernel_fn, in_axes=(0, 0, 0, None, None, None))

    def get_y(x):
        covariance = kernel_fn(
            x,
            lengthscale_trend(x),
            variance_trend(x),
            x,
            lengthscale_trend(x),
            variance_trend(x),
        )

        rows, columns = jnp.diag_indices_from(covariance)
        covariance = covariance.at[rows, columns].set(covariance[rows, columns] + 1e-6)

        key = jax.random.PRNGKey(seed)
        true_f = jnp.linalg.cholesky(covariance) @ jax.random.normal(
            key, shape=(x.shape[0],)
        )
        key = jax.random.split(key, 1)[0]
        y = true_f + jax.random.normal(key, true_f.shape) * (noise_var_trend(x) ** 0.5)
        return true_f, y

    key = jax.random.PRNGKey(seed + 1)
    x = jnp.sort(jax.random.uniform(key, (n_samples,), minval=-8, maxval=8))
    true_f, y = get_y(x)
    x = x.reshape(-1, 1)

    #     x_test = jnp.linspace(-2, 2, 300)
    #     y_test = get_y(x)

    return x, y, true_f, lengthscale_trend, variance_trend, noise_var_trend


def Tdata(data_seed=0, n_points=100):
    ## TODO: Generate data from a GP
    var = lambda x: jnp.exp(2 * jnp.sin(0.2 * x))
    noise_var = lambda x: jnp.exp(0.75 * jnp.sin(0.5 * x + 1)) + 0.1

    x = jax.random.uniform(jax.random.PRNGKey(data_seed), (n_points, 1))
    x = 16 * x - 8

    seed = jax.random.split(jax.random.PRNGKey(data_seed), 1)[0]
    y = var(x) * f(x) + tfd.Normal(0, noise_var(x)).sample(seed=seed)

    # x = StandardScaler().fit_transform(x)
    # y = StandardScaler().fit_transform(y)[:, 0]
    return x, y.ravel(), f, var, noise_var


def get_nonstat_2d(seed=999, n_train=121):
    key = jax.random.PRNGKey(seed)
    key2 = jax.random.PRNGKey(seed + 1)

    def noise(x):
        noise_val = 0.025
        maxval, minval = 1.0, -0.6
        x0 = (x[0] - minval) / (maxval - minval)
        x1 = (x[1] - minval) / (maxval - minval)
        return x0 * noise_val + x1 / 2 * noise_val

    def f(x):
        b = jnp.pi * (2 * x[0] + 0.5 * x[1] + 1)
        return 0.1 * (jnp.sin(b * x[0]) + jnp.sin(b * x[1]))

    x = jax.random.uniform(key, shape=(int(n_train / 0.8), 2), minval=-0.5, maxval=1.0)
    y = jax.vmap(f)(x) + jax.vmap(noise)(x)

    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y[:, None])[:, 0]

    return x, y


def get_heinonen(name):
    data = loadmat("datasets.mat")
    return data[name]["x"][0][0].astype(float), data[name]["y"][0][0].astype(float)


def get_heinonen_2d(n, seed=0, test=False):
    key = jax.random.PRNGKey(seed)

    def f(x1, x2):
        s_ = 0.025
        m = jnp.array([0.6, 0.2])
        s = jnp.array([s_, s_])
        h1 = -scipy.stats.norm.pdf(jnp.array([x1, x2]), loc=m, scale=s).prod() / 10

        m = jnp.array([0.6, 0.8])
        s = jnp.array([s_, s_])
        h2 = scipy.stats.norm.pdf(jnp.array([x1, x2]), loc=m, scale=s).prod() / 10

        f = 1.9 * (
            1.35
            + jnp.exp(x1)
            * jnp.sin(13 * ((x1 - 0.6) ** 2))
            * jnp.exp(-x2)
            * jnp.sin(7 * x2)
        )
        return f + h1 + h2

    if test:
        x = jnp.linspace(0, 1, n)
        X1, X2 = jnp.meshgrid(x, x)
        y = jax.vmap(jax.vmap(f))(X1, X2)
        x = jnp.array([(x1, x2) for x1, x2 in zip(X1.ravel(), X2.ravel())])
        key = jax.random.PRNGKey(seed + 1)
        y = y.ravel()  # + jax.random.normal(key, y.ravel().shape)*0.25
        return x, y.ravel()
    else:
        x = jax.random.uniform(key, (n * n, 2))
        y = jax.vmap(lambda x: f(x[0], x[1]))(x)
        y = y.ravel() + jax.random.normal(key, y.ravel().shape) * 0.25
        return x, y


def get_synth1d(key, flex_scale=True, flex_var=True, flex_noise=True, n_points=0):
    fn_dict = {}

    def kernel_fn(x1, ls1, var1, x2, ls2, var2):
        l_sqr_avg = (ls1**2 + ls2**2) / 2
        prefix = jnp.sqrt(ls1 * ls2 / l_sqr_avg)
        exp_part = jnp.exp(-0.5 * ((x1 - x2) ** 2) / l_sqr_avg)
        return (var1 * var2 * prefix * exp_part).squeeze()

    def add_noise(K, noise):
        rows, columns = jnp.diag_indices_from(K)
        return K.at[rows, columns].set(K[rows, columns] + noise.ravel() + 10e-6)

    kernel_fn = jax.vmap(kernel_fn, in_axes=(None, None, None, 0, 0, 0))
    kernel_fn = jax.vmap(kernel_fn, in_axes=(0, 0, 0, None, None, None))

    keys = jax.random.split(key, 3)
    if flex_scale:
        scale_fn = (
            lambda x: (0.5 * jnp.sin(x / 8)) + 1.5
        )  # jax.nn.softplus(gp.sample(keys[0]))
    else:
        scale_fn = lambda x: jnp.array(1.0).repeat(x.size).reshape(x.shape)

    if flex_var:
        var_fn = lambda x: 1.5 * jnp.exp(jnp.sin(0.2 * x))  # jnp.exp(jnp.sin(x))
    #         var_fn = lambda x:  jax.nn.softplus(gp.sample(keys[1])) #(1.1 + jnp.cos(x - jnp.pi / 2)) / 2
    else:
        var_fn = lambda x: jnp.array(1.0).repeat(x.size).reshape(x.shape)
    if flex_noise:
        noise_fn = lambda x: 2.5 * jax.nn.softplus(
            jnp.sin(0.2 * -x)
        )  # jnp.exp(jnp.sin(-x))
    #         noise_fn = lambda x: jax.nn.softplus(gp.sample(keys[2]))/2 # (1.1 + jnp.sin(x + jnp.pi / 2)) / 4
    else:
        noise_fn = lambda x: jnp.array(0.1).repeat(x.size).reshape(x.shape)

    fn_dict["scale"] = scale_fn
    fn_dict["var"] = var_fn
    fn_dict["noise"] = noise_fn
    #     lengthscale_trend = lambda x: (0.5 * jnp.sin(5 * x / 8)) + 1.0
    #     variance_trend = lambda x: jnp.exp(jnp.sin(0.2 * x))  # (0.3 * x**2) + 0.4
    #     noise_var_trend = lambda x: jnp.exp(jnp.sin(0.2 * -x))

    #     n_points = 125
    x = jnp.linspace(-30, 30, n_points).reshape(-1, 1)
    #     print(x.shape, scale_fn(x).shape, var_fn(x).shape)
    gp = GaussianProcess(kernel=1.0 * kernels.ExpSquared(scale=0.9), X=x)
    covar = kernel_fn(x, scale_fn(x), var_fn(x), x, scale_fn(x), var_fn(x))
    covar = add_noise(covar, jnp.array(0.0))

    true_fn = jnp.linalg.cholesky(covar) @ jax.random.normal(key, (n_points,))
    #     print(true_fn.shape, jax.random.normal(key, true_f.shape).shape)

    key = jax.random.split(key, 1)[0]
    y = true_fn + jax.random.normal(key, true_fn.shape).ravel() * (
        noise_fn(x).ravel() ** 0.5
    )

    return x, y, fn_dict, true_fn
