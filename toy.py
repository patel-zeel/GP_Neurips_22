#################### Imports
from time import time

checkpoint = time()


def get_time_gap():
    global checkpoint
    gap = time() - checkpoint
    checkpoint = time()
    return gap


import inspect
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from functools import partial
from copy import deepcopy

import jax
import jax.tree_util as jtu

import matplotlib.pyplot as plt

from helper import get_model, get_result, get_data
import gpax.distributions as gd
import gpax.bijectors as gb
import gpax.kernels as gk
from gpax.plotting import plot_posterior

import optax

jax.config.update("jax_enable_x64", True)

print(f"init time: {get_time_gap():.2f}s")

#################### Config
Config = type("Config", (object,), {})


def get_config():
    cfg = Config()
    cfg.jitter = 1e-6
    cfg.positive_bijector = gb.Exp
    cfg.flex_noise = True
    cfg.flex_scale = True
    cfg.flex_lengthscale = True
    cfg.n_inducing = 10
    cfg.n_iters = 2000
    cfg.n_restarts = 9
    cfg.lax_scan = True
    cfg.optimizer = optax.adam(learning_rate=0.01)
    cfg.ls_prior = gd.Gamma(concentration=5.0, rate=1.0)
    cfg.scale_prior = gd.Gamma(concentration=0.5, rate=1.0)
    cfg.latent_kernel = gk.RBF  # Other kernels then RBF resulted in all NaNs
    return cfg


cfg = get_config()
# MotorcycleHelmet, Step, Jump1D, Smooth1D, Olympic, SineNoisy
data_name = "SineNoisy"
path = "toy"


#################### Data
X, y, X_test = get_data(data_name)

#################### Setup
init_key = jax.random.PRNGKey(0)
init_keys = jax.random.split(init_key, cfg.n_restarts)
inducing_key = jax.random.PRNGKey(1)
inducing_keys = jax.random.split(inducing_key, cfg.n_restarts)

##### Mini-testing: Start
# tmp_cfg = deepcopy(cfg)
# tmp_cfg.lax_scan = False
# get_result(init_key, inducing_key, tmp_cfg, X, y, X_test)
##### Mini-testing: End

get_result = partial(get_result, cfg=cfg, X=X, y=y, X_test=X_test)
get_result = jax.jit(jax.vmap(get_result))

print(f"setup time: {get_time_gap():.2f}s")

#################### Run
pred_means, pred_covs, results, best_constrained_params, test_noises = get_result(
    init_keys, inducing_keys
)

print(f"run time: {get_time_gap():.2f}s")


#################### Plot
fig1, ax1 = plt.subplots(3, 3, figsize=(12, 12))
fig2, ax2 = plt.subplots(3, 3, figsize=(12, 12))
fig3, ax3 = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
ax1 = ax1.flatten()
ax2 = ax2.flatten()
ax3 = ax3.flatten()

for i in range(init_keys.shape[0]):
    result = jtu.tree_map(lambda x: x[i], results)
    pred_mean, pred_cov = pred_means[i], pred_covs[i]
    best_param = jtu.tree_map(lambda x: x[i], best_constrained_params)
    test_noise = test_noises[i]

    ax1[i].plot(result["loss_history"])
    ax1[i].set_title(f"Run {i}")
    plot_posterior(X, y, X_test, pred_mean, pred_cov, ax=ax2[i])
    ax2[i].set_title(f"Loss: {result['loss_history'][-1]:.3f}")

    ax3[i].plot(
        best_param["X_inducing"], best_param["likelihood"]["scale_inducing"], "o"
    )
    ax3[i].plot(X_test, test_noise)

fig1.savefig(f"{path}/{data_name}_loss.png")
fig2.savefig(f"{path}/{data_name}_posterior.png")
fig3.savefig(f"{path}/{data_name}_inducing.png")

with open(f"{path}/{data_name}_config.txt", "w") as f:
    print(inspect.getsource(get_config), file=f)

print(f"plot time: {get_time_gap():.2f}s")
