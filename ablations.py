from time import time

init = time()
checkpoint = time()


def time_it(label):
    global checkpoint
    new = time()
    print(f"{label}: Time: {new - checkpoint:.2f} seconds")
    checkpoint = new


import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = f"{sys.argv[4]}"

import pandas as pd
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from scipy.io import loadmat

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from functools import partial
from itertools import product

import optax
from base import loss_fn, predict_fn, get_params

from gpax.utils import train_fn

jax.config.update("jax_enable_x64", True)

time_it("Imports")


def get_result(X, y, X_test, flex_dict, method):
    partial_loss_fn = partial(loss_fn, X=X, y=y, flex_dict=flex_dict, method=method)
    partial_get_params = partial(
        get_params, X=X, flex_dict=flex_dict, method=method, default=False
    )
    params = jax.vmap(partial_get_params)(
        jax.random.split(jax.random.PRNGKey(1000), 10)
    )
    partial_train_fn = partial(
        train_fn,
        loss_fn=partial_loss_fn,
        optimizer=optax.adam(0.01),
        n_iters=2500,
    )

    results = jax.vmap(partial_train_fn)(init_raw_params=params)
    best_idx = jnp.nanargmin(results["loss_history"][:, -1])
    result = jtu.tree_map(lambda x: x[best_idx], results)

    pred_mean, pred_var, pred_ell, pred_sigma, pred_omega = predict_fn(
        result["raw_params"], X, y, X_test, flex_dict, method
    )

    return {
        "pred_mean": pred_mean,
        "pred_var": pred_var,
        "pred_ell": pred_ell,
        "pred_sigma": pred_sigma,
        "pred_omega": pred_omega,
    }


def get_data(latent_seed, data_seed, ell, sigma, omega):
    data_name = f"gen_{latent_seed}_{data_seed}_{ell}_{sigma}_{omega}"
    data_path = f"data/{data_name}.mat"
    data = loadmat(data_path)[data_name]
    X = jnp.array(data["X"][0][0])
    y_clean = jnp.array(data["y_clean"][0][0]).squeeze()
    y = jnp.array(data["y_noisy"][0][0]).squeeze()
    ell_true = jnp.array(data["ell"][0][0]).squeeze()
    sigma_true = jnp.array(data["sigma"][0][0]).squeeze()
    omega_true = jnp.array(data["omega"][0][0]).squeeze()
    return X, y, y_clean, ell_true, sigma_true, omega_true


latent_seed = int(sys.argv[1])
data_seed = int(sys.argv[2])
method = sys.argv[3]

idx = 0
results = {}
for gen_ell, gen_sigma, gen_omega in product([1, 0], repeat=3):
    X_all, y_all, y_clean, ell_true, sigma_true, omega_true = get_data(
        latent_seed, data_seed, gen_ell, gen_sigma, gen_omega
    )

    ## Normalize
    x_scaler = MinMaxScaler()
    X_all = x_scaler.fit_transform(X_all)
    xscale = x_scaler.data_max_ - x_scaler.data_min_
    yscale = jnp.max(jnp.abs(y_all - jnp.mean(y_all)))
    ymean = jnp.mean(y_all)
    y_all = (y_all - ymean) / yscale

    X_A, X_B, y_A, y_B, y_clean_A, y_clean_B = train_test_split(
        X_all, y_all, y_clean, test_size=0.5, random_state=idx
    )

    XX = jnp.concatenate([X_A[:, None], X_B[:, None]], axis=2)
    YY = jnp.concatenate([y_A[:, None], y_B[:, None]], axis=1)
    YY_clean = jnp.concatenate([y_clean_A[:, None], y_clean_B[:, None]], axis=1)
    XX_test = jnp.concatenate([X_B[:, None], X_A[:, None]], axis=2)
    YY_test = jnp.concatenate([y_B[:, None], y_A[:, None]], axis=1)
    YY_clean_test = jnp.concatenate([y_clean_B[:, None], y_clean_A[:, None]], axis=1)

    results[f"gen_{gen_ell}_{gen_sigma}_{gen_omega}"] = {
        "XX": XX,
        "YY": YY,
        "YY_clean": YY_clean,
        "XX_test": XX_test,
        "YY_test": YY_test,
        "YY_clean_test": YY_clean_test,
    }

    for ell, sigma, omega in product([1, 0], repeat=3):
        column = ".".join(map(str, [gen_ell, gen_sigma, gen_omega]))
        row = ".".join(map(str, [ell, sigma, omega]))
        idx += 1

        flex_dict = {"ell": ell, "sigma": sigma, "omega": omega}

        partial_get_result = partial(
            get_result,
            flex_dict=flex_dict,
            method=method,
        )
        result = jax.vmap(partial_get_result, in_axes=(2, 1, 2))(XX, YY, XX_test)

        results[f"gen_{gen_ell}_{gen_sigma}_{gen_omega}"][
            f"{ell}_{sigma}_{omega}"
        ] = result

        time_it(f"{column} -> {row}: {idx} of 63")

print(f"Saving results")
pd.to_pickle(
    results,
    f"results/{method}_{latent_seed}_{data_seed}.pkl",
)

print("Total time taken to run the script: ", (time() - init) / 60, "minutes")
print()
