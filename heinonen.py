from time import time

import jaxopt

checkpoint = time()


def time_it(label):
    global checkpoint
    new = time()
    print(f"{label}: Time: {new - checkpoint:.2f} seconds")
    checkpoint = new


import jax
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

from common import (
    get_white,
    get_latent_chol,
    predict_h,
    gibbs_k,
    get_log_h,
    jitter,
    get_log_normal,
    get_simulated_data,
    generate_heinonen_gp_data,
)

jax.config.update("jax_enable_x64", True)

time_it("Imports")


cs = jsp.linalg.cho_solve
st = jsp.linalg.solve_triangular


def value_and_grad_fn(params, X, y):
    for name in ["ell", "sigma", "omega"]:
        params[f"{name}_gp_log_ell"] = jax.lax.stop_gradient(
            params[f"{name}_gp_log_ell"]
        )
        params[f"{name}_gp_log_sigma"] = jax.lax.stop_gradient(
            params[f"{name}_gp_log_sigma"]
        )

    grads = {}
    log_ell, chol_ell, _ = get_log_h(
        params["white_ell"],
        X,
        ell=jnp.exp(params["ell_gp_log_ell"]),
        sigma=jnp.exp(params["ell_gp_log_sigma"]),
    )
    log_omega, chol_omega, _ = get_log_h(
        params["white_omega"],
        X,
        ell=jnp.exp(params["omega_gp_log_ell"]),
        sigma=jnp.exp(params["omega_gp_log_sigma"]),
    )
    log_sigma, chol_sigma, _ = get_log_h(
        params["white_sigma"],
        X,
        ell=jnp.exp(params["sigma_gp_log_ell"]),
        sigma=jnp.exp(params["sigma_gp_log_sigma"]),
    )

    (ell, omega, sigma) = jnp.exp(log_ell), jnp.exp(log_omega), jnp.exp(log_sigma)

    K_f, aux = gibbs_k(X, X, ell, ell, sigma, sigma)
    K_y = add_to_diagonal(K_f, omega**2, 0.0)

    # ### Manual Grads
    # a, chol_y = get_a_inv_b(K_y, y, return_cholesky=True)
    # aat = a.reshape(-1, 1) @ a.reshape(1, -1)

    # ## omega
    # o2 = cs((chol_omega, True), log_omega - log_omega.mean())
    # o1 = aat @ jnp.diag(omega**2) - cs((chol_y, True), jnp.diag(omega**2))
    # grads["white_omega"] = chol_omega.T @ (jnp.diag(o1) - o2)

    # ## sigma
    # s2 = cs((chol_sigma, True), log_sigma - log_sigma.mean())
    # s1 = aat @ K_f - cs((chol_y, True), K_f)
    # grads["white_sigma"] = chol_sigma.T @ (jnp.diag(s1) - s2)

    # ## ell
    # dK = (
    #     (aux["variance"] / aux["prefix_part"] * aux["exp_part"] / aux["l_avg_square"] ** 3 / 8)
    #     * (ell.reshape(-1, 1) * ell.reshape(1, -1))
    #     * (4 * aux["squared_dist"] * ell.reshape(-1, 1) ** 2 - ell.reshape(-1, 1) ** 4 + ell.reshape(1, -1) ** 4)
    # )
    # l2 = cs((chol_ell, True), log_ell - log_ell.mean())
    # l1 = aat @ dK - cs((chol_y, True), dK)
    # grads["white_ell"] = chol_ell.T @ (jnp.diag(l1) - l2)

    mu_f = 0
    log_lik = tfd.MultivariateNormalFullCovariance(
        loc=mu_f, covariance_matrix=K_y
    ).log_prob(y)

    # Type - A - Prior on correlated parameters
    log_prior_ell = tfd.MultivariateNormalTriL(
        loc=log_ell.mean(), scale_tril=chol_ell
    ).log_prob(log_ell)
    log_prior_omega = tfd.MultivariateNormalTriL(
        loc=log_omega.mean(), scale_tril=chol_omega
    ).log_prob(log_omega)
    log_prior_sigma = tfd.MultivariateNormalTriL(
        loc=log_sigma.mean(), scale_tril=chol_sigma
    ).log_prob(log_sigma)

    # log_prior_ell += tfd.Normal(2.0, 2.0).log_prob(log_ell.mean())
    # log_prior_omega += tfd.Normal(2.0, 2.0).log_prob(log_omega.mean())
    # log_prior_sigma += tfd.Normal(2.0, 2.0).log_prob(log_sigma.mean())

    # Type -B - Prior on White parameters
    # log_prior_ell = (
    #     tfd.Normal(st(chol_ell, params["white_ell_mean"].repeat(X.shape[0]), lower=True), 1.0)
    #     .log_prob(params["white_ell"])
    #     .sum()
    # )
    # log_prior_omega = (
    #     tfd.Normal(st(chol_omega, params["white_omega_mean"].repeat(X.shape[0]), lower=True), 1.0)
    #     .log_prob(params["white_omega"])
    #     .sum()
    # )
    # log_prior_sigma = (
    #     tfd.Normal(st(chol_sigma, params["white_sigma_mean"].repeat(X.shape[0]), lower=True), 1.0)
    #     .log_prob(params["white_sigma"])
    #     .sum()
    # )

    # Type -C - Prior on Standard Normal parameters
    # log_prior_ell = tfd.Normal(0.0, 1.0).log_prob(params["white_ell"]).sum()
    # log_prior_omega = tfd.Normal(0.0, 1.0).log_prob(params["white_omega"]).sum()
    # log_prior_sigma = tfd.Normal(0.0, 1.0).log_prob(params["white_sigma"]).sum()

    lgp_ell_prior = 0.0
    lgp_sigma_prior = 0.0
    # lgp_ell_prior_d = gd.Frechet(rate=-jnp.log(0.1) * (0.2**0.5), dim=1)

    lgp_ell_prior_d = get_log_normal(desired_mode=0.2)
    lgp_sigma_prior_d = tfd.Exponential(
        rate=-jnp.log(0.05 / 5.0)
    )  # PC prior for U=5.0 with 0.05 probability

    lgp_ell_prior += lgp_ell_prior_d.log_prob(jnp.exp(params["ell_gp_log_ell"])).sum()
    lgp_ell_prior += lgp_ell_prior_d.log_prob(jnp.exp(params["omega_gp_log_ell"])).sum()
    lgp_ell_prior += lgp_ell_prior_d.log_prob(jnp.exp(params["sigma_gp_log_ell"])).sum()

    lgp_sigma_prior += lgp_sigma_prior_d.log_prob(
        jnp.exp(params["ell_gp_log_sigma"])
    ).sum()
    lgp_sigma_prior += lgp_sigma_prior_d.log_prob(
        jnp.exp(params["omega_gp_log_sigma"])
    ).sum()
    lgp_sigma_prior += lgp_sigma_prior_d.log_prob(
        jnp.exp(params["sigma_gp_log_sigma"])
    ).sum()

    print(
        "log_lik",
        log_lik,
        "log_prior_ell",
        log_prior_ell,
        "log_prior_omega",
        log_prior_omega,
        "log_prior_sigma",
        log_prior_sigma,
        "lgp_ell_prior",
        lgp_ell_prior,
        "lgp_sigma_prior",
        lgp_sigma_prior,
    )

    return -(
        log_lik
        + log_prior_ell
        + log_prior_omega
        + log_prior_sigma
        + lgp_ell_prior
        + lgp_sigma_prior
    )  # , grads

    # Use this for debugging
    # auto_grad, manual_grad = jax.grad(value_and_grad_fn, has_aux=True)(params, X, y)


def predict_fn(params, X, y, X_new):
    ell, ell_new = predict_h(
        params["white_ell"],
        X,
        X_new,
        ell=jnp.exp(params["ell_gp_log_ell"]),
        sigma=jnp.exp(params["ell_gp_log_sigma"]),
    )
    omega, omega_new = predict_h(
        params["white_omega"],
        X,
        X_new,
        ell=jnp.exp(params["omega_gp_log_ell"]),
        sigma=jnp.exp(params["omega_gp_log_sigma"]),
    )
    sigma, sigma_new = predict_h(
        params["white_sigma"],
        X,
        X_new,
        ell=jnp.exp(params["sigma_gp_log_ell"]),
        sigma=jnp.exp(params["sigma_gp_log_sigma"]),
    )

    K, _ = gibbs_k(X, X, ell, ell, sigma, sigma)
    K_noisy = add_to_diagonal(K, omega**2, jitter)
    chol_y = jnp.linalg.cholesky(K_noisy)

    K_star, _ = gibbs_k(X_new, X, ell_new, ell, sigma_new, sigma)
    K_star_star, _ = gibbs_k(X_new, X_new, ell_new, ell_new, sigma_new, sigma_new)

    pred_mean = K_star @ cs((chol_y, True), y)
    pred_cov = K_star_star - K_star @ cs((chol_y, True), K_star.T)
    return pred_mean, pred_cov, omega_new**2


#### data
# data = rd.MotorcycleHelmet
# X, y, _ = data().get_data()
# ## pre-scale
# X = MinMaxScaler().fit_transform(X)
# tmp_yscale = jnp.max(jnp.abs(y - jnp.mean(y)))
# y = (y - y.mean()) / tmp_yscale
# save_path = f"figures/{data.__name__}"
######################################

# X, y, fn_dict, _ = get_simulated_data(flex_scale=1, flex_noise=1, flex_var=1)
# save_path = "figures/simulated"

######################################
save_path = "figures/heinonen_gen"
X_key = jax.random.PRNGKey(0)
X_test_key = jax.random.PRNGKey(100)
X = jax.random.uniform(X_key, shape=(100, 1)).sort(axis=0)
X_test = jnp.linspace(-2, 3, 250).reshape(-1, 1)

latent_key = jax.random.PRNGKey(200)
data_key = jax.random.PRNGKey(301)
y, ell_true, sigma_true, omega_true = generate_heinonen_gp_data(X, latent_key, data_key)

## Normalize
# x_scaler = MinMaxScaler()
# X = x_scaler.fit_transform(X)
# X_test = x_scaler.transform(X_test)
# xscale = x_scaler.data_max_ - x_scaler.data_min_
# yscale = jnp.max(jnp.abs(y - jnp.mean(y)))
# ymean = jnp.mean(y)
# y = (y - ymean) / yscale

time_it("Data loaded")


def get_params(key):
    values = jax.random.uniform(key, shape=(3,), minval=-3.0, maxval=-1.0)
    params = {
        "white_ell": get_white(
            jnp.exp(values[0]),
            X,
            ell=0.2,
            sigma=1.0,
        ),
        "white_sigma": get_white(jnp.exp(values[1]), X, ell=0.2, sigma=1.0),
        "white_omega": get_white(jnp.exp(values[2]), X, ell=0.3, sigma=1.0),
        "ell_gp_log_ell": jnp.log(jnp.array(0.2)),
        "sigma_gp_log_ell": jnp.log(jnp.array(0.2)),
        "omega_gp_log_ell": jnp.log(jnp.array(0.3)),
        "ell_gp_log_sigma": jnp.log(jnp.array(1.0)),
        "sigma_gp_log_sigma": jnp.log(jnp.array(1.0)),
        "omega_gp_log_sigma": jnp.log(jnp.array(1.0)),
    }
    return params


value_and_grad_fn = partial(value_and_grad_fn, X=X, y=y)
# print("Initial loss", value_and_grad_fn(params))
# sys.exit()

time_it("Setup done")

params = jax.vmap(get_params)(jax.random.split(jax.random.PRNGKey(1000), 10))
partial_train_fn = partial(
    train_fn, loss_fn=value_and_grad_fn, optimizer=optax.adam(0.01), n_iters=2500
)

results = jax.vmap(partial_train_fn)(init_raw_params=params)
print("Losses: ", results["loss_history"][:, -1])
best_idx = jnp.nanargmin(results["loss_history"][:, -1])
result = jtu.tree_map(lambda x: x[best_idx], results)
# res = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=value_and_grad_fn).run(params)
# result = {"raw_params": res.params}

time_it("Training done")

plt.figure(figsize=(10, 3))
plt.plot(result["loss_history"])
plt.savefig(f"{save_path}_loss.png")


fig, ax = plt.subplots(1, 1, figsize=(15, 3))
time_it("Plotting loss done")

print(value_and_grad_fn(result["raw_params"]))

pred_mean, pred_cov, pred_noise = predict_fn(result["raw_params"], X, y, X_test)

pred_ell = predict_h(
    result["raw_params"]["white_ell"],
    X,
    X_test,
    ell=jnp.exp(result["raw_params"]["ell_gp_log_ell"]),
    sigma=jnp.exp(result["raw_params"]["ell_gp_log_sigma"]),
)[1]
pred_sigma = predict_h(
    result["raw_params"]["white_sigma"],
    X,
    X_test,
    ell=jnp.exp(result["raw_params"]["sigma_gp_log_ell"]),
    sigma=jnp.exp(result["raw_params"]["sigma_gp_log_sigma"]),
)[1]
pred_omega = predict_h(
    result["raw_params"]["white_omega"],
    X,
    X_test,
    ell=jnp.exp(result["raw_params"]["omega_gp_log_ell"]),
    sigma=jnp.exp(result["raw_params"]["omega_gp_log_sigma"]),
)[1]

time_it("Prediction done")

# Denormalize
# X = x_scaler.inverse_transform(X)
# X_test = x_scaler.inverse_transform(X_test)
# y = y * yscale + ymean
# pred_mean = pred_mean * yscale + ymean
# pred_cov = pred_cov * yscale**2
# pred_noise = pred_noise * yscale**2

# pred_ell = pred_ell * xscale
# pred_sigma = pred_sigma * yscale
# pred_omega = pred_omega * yscale
#########################################

ax.scatter(X, y, label="data")
ax.plot(X_test, pred_mean, label="mean")
ax.fill_between(
    X_test[:, 0],  # x
    pred_mean - 2 * jnp.sqrt(pred_cov.diagonal()),  # y1
    pred_mean + 2 * jnp.sqrt(pred_cov.diagonal()),  # y2
    alpha=0.5,
    label="2 std",
)
ax.fill_between(
    X_test[:, 0],  # x
    pred_mean - 2 * jnp.sqrt(pred_cov.diagonal() + pred_noise),  # y1
    pred_mean + 2 * jnp.sqrt(pred_cov.diagonal() + pred_noise),  # y2
    alpha=0.5,
    label="2 std + noise",
)
ax.legend()
fig.savefig(f"{save_path}_posterior.png")

print("ell lgp", jnp.exp(result["raw_params"]["ell_gp_log_ell"]))
print("ell sgp", jnp.exp(result["raw_params"]["sigma_gp_log_ell"]))
print("ell ogp", jnp.exp(result["raw_params"]["omega_gp_log_ell"]))
print("sigma lgp", jnp.exp(result["raw_params"]["ell_gp_log_sigma"]))
print("sigma sgp", jnp.exp(result["raw_params"]["sigma_gp_log_sigma"]))
print("sigma ogp", jnp.exp(result["raw_params"]["omega_gp_log_sigma"]))

fig, ax = plt.subplots(1, 1, figsize=(15, 3))


ax.plot(X_test, pred_ell, label="ell", color="r")
ax.plot(X_test, pred_sigma, label="sigma", color="g")
ax.plot(X_test, pred_omega, label="omega", color="b")

ax.plot(
    X, ell_true, label="ell_true", color="r", linestyle="--", alpha=0.8, linewidth=4
)
ax.plot(
    X, sigma_true, label="sigma_true", color="g", linestyle="--", alpha=0.8, linewidth=4
)
ax.plot(
    X, omega_true, label="omega_true", color="b", linestyle="--", alpha=0.8, linewidth=4
)

# ax.set_ylim(0, 2)
ax.legend()

fig.savefig(f"{save_path}_latent_fn.png")

time_it("Plotting done")

print()
