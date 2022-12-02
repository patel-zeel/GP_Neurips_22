import jax
import jaxopt
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from functools import partial

import optax

from gpax.kernels import RBF
import regdata as rd

import matplotlib.pyplot as plt

from gpax.plotting import plot_posterior
from gpax.utils import (
    add_to_diagonal,
    squared_distance,
    get_a_inv_b,
    repeat_to_size,
    train_fn,
)
import gpax.distributions as gd
import gpax.bijectors as gb
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

jax.config.update("jax_enable_x64", True)

jitter = 1e-6
cs = jsp.linalg.cho_solve
st = jsp.linalg.solve_triangular
dist_f = jax.vmap(squared_distance, in_axes=(None, 0))
dist_f = jax.vmap(dist_f, in_axes=(0, None))


def get_log_normal(desired_mode):
    log_mode = jnp.log(desired_mode)
    mu = 0.0
    scale = jnp.sqrt(mu - log_mode)
    return tfd.LogNormal(loc=mu, scale=scale)


def get_latent_chol(X, ell, sigma):
    kernel_fn = RBF(input_dim=X.shape[1], lengthscale=ell, scale=sigma)
    cov = kernel_fn(X, X)
    noisy_cov = add_to_diagonal(cov, 0.0, jitter)
    chol = jnp.linalg.cholesky(noisy_cov)
    return chol, kernel_fn


def get_white(h, X, ell, sigma, scalar=False):
    log_h = jnp.log(repeat_to_size(h, X.shape[0]))
    if scalar:
        return log_h[0]
    else:
        chol, _ = get_latent_chol(X, ell, sigma)
        return st(chol, log_h, lower=True)


def get_log_h(white_h, X, ell, sigma):
    chol, kernel_fn = get_latent_chol(X, ell, sigma)
    return chol @ white_h, chol, kernel_fn


def predict_h(white_h, X, X_new, ell, sigma, scalar=False):
    if scalar:
        chol, _ = get_latent_chol(X, ell, sigma)
        return (
            jnp.exp(repeat_to_size(white_h, X.shape[0])),
            jnp.exp(repeat_to_size(white_h, X_new.shape[0])),
            chol,
        )
    else:
        log_h, chol, kernel_fn = get_log_h(white_h, X, ell, sigma)
        K_star = kernel_fn(X_new, X)
        return (
            jnp.exp(log_h),
            jnp.exp(log_h.mean() + K_star @ cs((chol, True), log_h - log_h.mean())),
            chol,
        )


def gibbs_k(X1, X2, ell1, ell2, s1, s2):
    ell1, ell2 = ell1.reshape(-1, 1), ell2.reshape(-1, 1)  # 1D only
    l_avg_square = (ell1**2 + ell2.T**2) / 2.0
    prefix_part = jnp.sqrt(ell1 * ell2.T / l_avg_square)
    squared_dist = dist_f(X1, X2)
    exp_part = jnp.exp(-squared_dist / (2.0 * l_avg_square))
    s1, s2 = s1.reshape(-1, 1), s2.reshape(-1, 1)  # 1D only
    variance = s1 * s2.T
    return variance * prefix_part * exp_part, locals()


def generate_heinonen_gp_data(X, latent_key, data_key, flex_dict):
    white_ell, white_sigma, white_omega = jax.random.normal(
        latent_key, shape=(3, X.shape[0])
    )

    ell_chol, _ = get_latent_chol(X, ell=0.2, sigma=1.0)
    if flex_dict["ell"]:
        log_ell = ell_chol @ white_ell
    else:
        log_ell = repeat_to_size(white_ell[0], X.shape[0])

    sigma_chol, _ = get_latent_chol(X, ell=0.2, sigma=1.0)

    if flex_dict["sigma"]:
        log_sigma = sigma_chol @ white_sigma
    else:
        log_sigma = repeat_to_size(white_sigma[0], X.shape[0])

    omega_chol, _ = get_latent_chol(X, ell=0.2, sigma=1.0)
    if flex_dict["omega"]:
        log_omega = omega_chol @ white_omega
    else:
        log_omega = repeat_to_size(white_omega[0], X.shape[0])

    cov, _ = gibbs_k(
        X, X, jnp.exp(log_ell), jnp.exp(log_ell), jnp.exp(log_sigma), jnp.exp(log_sigma)
    )
    noisy_cov = add_to_diagonal(cov, jnp.exp(log_omega) ** 2, 0.0)

    # generate a sample with zero mean
    y = jax.random.multivariate_normal(
        data_key, mean=jnp.zeros(X.shape[0]), cov=noisy_cov
    )

    return y, jnp.exp(log_ell), jnp.exp(log_sigma), jnp.exp(log_omega)


def get_simulated_data(flex_scale=False, flex_var=False, flex_noise=False):
    key = jax.random.PRNGKey(1221)  # was 1221
    n_points = 200
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
        scale_fn = lambda x: (0.5 * jnp.sin(x / 8)) + 1.5
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

    fn_dict["lengthscale"] = scale_fn
    fn_dict["scale"] = var_fn
    fn_dict["noise"] = noise_fn
    #     lengthscale_trend = lambda x: (0.5 * jnp.sin(5 * x / 8)) + 1.0
    #     variance_trend = lambda x: jnp.exp(jnp.sin(0.2 * x))  # (0.3 * x**2) + 0.4
    #     noise_var_trend = lambda x: jnp.exp(jnp.sin(0.2 * -x))

    #     n_points = 125
    x = jnp.linspace(-30, 30, n_points).reshape(-1, 1)
    #     print(x.shape, scale_fn(x).shape, var_fn(x).shape)
    # gp = GaussianProcess(kernel=1.0 * kernels.ExpSquared(scale=0.9), X=x)
    covar = kernel_fn(x, scale_fn(x), var_fn(x), x, scale_fn(x), var_fn(x))
    covar = add_noise(covar, jnp.array(0.0))

    true_fn = jnp.linalg.cholesky(covar) @ jax.random.normal(key, (n_points,))
    #     print(true_fn.shape, jax.random.normal(key, true_f.shape).shape)

    key = jax.random.split(key, 1)[0]
    y = true_fn + jax.random.normal(key, true_fn.shape).ravel() * (
        noise_fn(x).ravel() ** 0.5
    )

    return x, y, fn_dict, true_fn
