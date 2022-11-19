import jax

import regdata as rd

from gpax.models import ExactGPRegression
from gpax.core import Parameter
from gpax.defaults import set_default_jitter
import gpax.means as gm
import gpax.likelihoods as gl
import gpax.kernels as gk
import gpax.distributions as gd
import gpax.bijectors as gb
from gpax.utils import train_fn

import optax


def get_model(cfg, X, inducing_key):
    # Config
    set_default_jitter(cfg.jitter)
    gb.set_positive_bijector(cfg.positive_bijector)

    # Setup
    if cfg.flex_noise:
        X_inducing = jax.random.uniform(
            inducing_key,
            (cfg.n_inducing, X.shape[1]),
            minval=X.min() - X.std(),
            maxval=X.max() + X.std(),
        )
        X_inducing = Parameter(X_inducing, fixed_init=True)
        latent_kernel = cfg.latent_kernel
        latent_ls = Parameter(1.0, gb.get_positive_bijector(), cfg.ls_prior)
        latent_scale = Parameter(1.0, gb.get_positive_bijector(), cfg.scale_prior)
        latent_gp = ExactGPRegression(
            kernel=latent_kernel(
                input_dim=X.shape[1], lengthscale=latent_ls, scale=latent_scale
            ),
            likelihood=gl.Gaussian(),
            mean=gm.Scalar(),
        )
        likelihood = gl.HeteroscedasticGaussian(
            latent_gp=latent_gp, X_inducing=X_inducing
        )
    else:
        likelihood = gl.Gaussian()

    kernel = gk.RBF(input_dim=X.shape[1])
    mean = gm.Scalar()
    return ExactGPRegression(
        kernel=kernel, likelihood=likelihood, mean=mean, X_inducing=X_inducing
    )


def get_result(init_key, inducing_key, cfg, X, y, X_test):
    gp = get_model(cfg, X, inducing_key)
    optimizer = cfg.optimizer
    gp.initialize(init_key)
    raw_params = gp.get_params()

    def loss_fn(raw_params):
        gp.set_params(raw_params)
        log_prior = gp.log_prior()
        log_prob = gp.log_probability(X, y)
        return -log_prob - log_prior

    result = train_fn(
        loss_fn, raw_params, optimizer, cfg.n_iters, lax_scan=cfg.lax_scan
    )
    gp.set_params(result["raw_params"])
    pred_mean, pred_cov = gp.predict(X, y, X_test)
    test_noise = gp.likelihood(X_test)
    best_constrained_params = gp.get_constrained_params()
    return pred_mean, pred_cov, result, best_constrained_params, test_noise


def get_data(name):
    if name == "MotorcycleHelmet":
        X, y, X_test = rd.MotorcycleHelmet().get_data()
        X_test = X_test * 1.5
        return X, y, X_test
    elif name == "Step":
        X, y, X_test = rd.Step().get_data()
        X_test = X_test * 1.5
        return X, y, X_test
    elif name == "Jump1D":
        X, y, X_test = rd.Jump1D().get_data()
        X_test = X_test * 1.5
        return X, y, X_test
    elif name == "Smooth1D":
        X, y, X_test = rd.Smooth1D().get_data()
        X_test = X_test * 1.5
        return X, y, X_test
    elif name == "Olympic":
        X, y, X_test = rd.Olympic().get_data()
        X_test = X_test * 1.5
        return X, y, X_test
    elif name == "SineNoisy":
        X, y, X_test = rd.SineNoisy().get_data()
        X_test = X_test * 1.5
        return X, y, X_test
    else:
        raise ValueError(f"Unknown dataset: {name}")
