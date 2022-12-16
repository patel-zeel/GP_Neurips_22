from time import time

init = time()

checkpoint = time()


def time_it(label):
    global checkpoint
    new = time()
    print(f"{label}: Time: {new - checkpoint:.2f} seconds")
    checkpoint = new


import timeit
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from scipy.io import loadmat
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.io import loadmat

from sklearn.preprocessing import MinMaxScaler
from functools import partial

import optax


import matplotlib.pyplot as plt

from gpax.utils import train_fn

from base import loss_fn, predict_fn, get_params

jax.config.update("jax_enable_x64", True)

time_it("Imports")

# config
flex_dict = {"ell": 1, "omega": 1, "sigma": 1}
method = "delta_inducing"  # "heinonen" or "delta_inducing"
data_name = f"gen_200_300_1_1_1"
data_path = f"data/{data_name}.mat"
default_params = False

#### data
# data = rd.MotorcycleHelmet
# save_path = f"figures/{data.__name__}"
# X, y, _ = data().get_data()
## pre-scale
# X = MinMaxScaler().fit_transform(X)
# tmp_yscale = jnp.max(jnp.abs(y - jnp.mean(y)))
# y = (y - y.mean()) / tmp_yscale
######################################


######################################

data = loadmat(data_path)[data_name]
X = jnp.array(data["X"][0][0])
y = jnp.array(data["y_noisy"][0][0]).squeeze()
ell_true = jnp.array(data["ell"][0][0]).squeeze()
sigma_true = jnp.array(data["sigma"][0][0]).squeeze()
omega_true = jnp.array(data["omega"][0][0]).squeeze()

save_path = f"figures/{data_name}_{method}_{flex_dict['ell']}_{flex_dict['sigma']}_{flex_dict['omega']}"

## Normalize
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
# X_test = x_scaler.transform(X_test)
xscale = x_scaler.data_max_ - x_scaler.data_min_
yscale = jnp.max(jnp.abs(y - jnp.mean(y)))
ymean = jnp.mean(y)
y = (y - ymean) / yscale

X_test = jnp.linspace(-2, 3, 250).reshape(-1, 1)
time_it("Data loaded")

experiment_scale = 1

value_and_grad_fn = partial(loss_fn, X=X, y=y, flex_dict=flex_dict, method=method)


time_it("Setup done")

partial_get_params = partial(
    get_params, X=X, flex_dict=flex_dict, method=method, default=default_params
)
params = jax.vmap(partial_get_params)(jax.random.split(jax.random.PRNGKey(1000), 10))
partial_train_fn = partial(
    train_fn, loss_fn=value_and_grad_fn, optimizer=optax.adam(0.01), n_iters=2500
)

# print("Initial loss", value_and_grad_fn(jtu.tree_map(lambda x: x[0], params)))
# sys.exit()

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

time_it("Plotting loss done")

# value_and_grad_fn(result["raw_params"])  # To check final loss breakdown

pred_mean, pred_var, pred_ell, pred_sigma, pred_omega = predict_fn(
    result["raw_params"],
    X,
    y,
    X_test,
    flex_dict,
    method,
)

time_it("Prediction done")

# Denormalize
X = x_scaler.inverse_transform(X)
X_test = x_scaler.inverse_transform(X_test)
if method == "delta_inducing":
    X_inducing = x_scaler.inverse_transform(result["raw_params"]["X_inducing"])

y = y * yscale + ymean
pred_mean = pred_mean * yscale + ymean
pred_var = pred_var * yscale**2

pred_ell = pred_ell * xscale
pred_sigma = pred_sigma * yscale
pred_omega = pred_omega * yscale
#########################################

fig, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.scatter(X, y, label="data")
ax.plot(X_test, pred_mean, label="mean")
ax.fill_between(
    X_test[:, 0],  # x
    pred_mean - 2 * jnp.sqrt(pred_var),  # y1
    pred_mean + 2 * jnp.sqrt(pred_var),  # y2
    alpha=0.5,
    label="2 std",
)
ax.fill_between(
    X_test[:, 0],  # x
    pred_mean - 2 * jnp.sqrt(pred_var + pred_omega**2),  # y1
    pred_mean + 2 * jnp.sqrt(pred_var + pred_omega**2),  # y2
    alpha=0.5,
    label="2 std + noise",
)

if method == "delta_inducing":
    for each in X_inducing.ravel():
        ax.axvline(each, color="k", linestyle="--", alpha=0.5)

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

if method == "delta_inducing":
    for each in X_inducing.ravel():
        ax.axvline(each, color="k", linestyle="--", alpha=0.5)

if "ell_true" in locals():
    ax.plot(
        X, ell_true, label="ell_true", color="r", linestyle="--", alpha=0.8, linewidth=4
    )
    ax.plot(
        X,
        sigma_true,
        label="sigma_true",
        color="g",
        linestyle="--",
        alpha=0.8,
        linewidth=4,
    )
    ax.plot(
        X,
        omega_true,
        label="omega_true",
        color="b",
        linestyle="--",
        alpha=0.8,
        linewidth=4,
    )

# ax.set_ylim(0, 2)
ax.legend()

fig.savefig(f"{save_path}_latent_fn.png")

time_it("Plotting done")

print(f"Total time: {(time() - init)/60:.2f} min")
