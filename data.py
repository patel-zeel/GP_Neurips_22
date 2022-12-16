# JAX
import jax
import jax.numpy as jnp

# Other imports
import sys
from scipy.io import savemat
import regdata as rd
from functools import partial
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from base import JITTER, get_log_h, sqr_distance_fn, gibbs_kernel, get_log_h
from utils import add_to_diagonal


def generate_heinonen_gp_data(X, latent_key, data_key, flex_dict):
    latent_keys = jax.random.split(latent_key, 2)
    white_sigma, white_omega = jax.random.normal(latent_keys[0], shape=(2, X.shape[0]))
    white_ell = jax.random.normal(latent_keys[1], shape=X.shape)

    param_names = ["ell", "sigma", "omega"]
    aux = {
        "ell_gp_ell": jnp.array(0.2),
        "ell_gp_sigma": jnp.array(1.0),
        "sigma_gp_ell": jnp.array(0.2),
        "sigma_gp_sigma": jnp.array(1.0),
        "omega_gp_ell": jnp.array(0.3),
        "omega_gp_sigma": jnp.array(1.0),
        "white_ell": white_ell,
        "white_sigma": white_sigma,
        "white_omega": white_omega,
    }

    for name in param_names:
        if bool(flex_dict[name]) is True:
            fn = partial(
                get_log_h,
                ell=aux[f"{name}_gp_ell"],
                sigma=aux[f"{name}_gp_sigma"],
            )
            if name == "ell":
                aux[name] = jnp.exp(
                    jax.vmap(fn, in_axes=(1, 1))(aux[f"white_{name}"], X)[0].T
                )
            else:
                aux[name] = jnp.exp(fn(white_h=aux[f"white_{name}"], X=X)[0].ravel())
        else:
            if name == "ell":
                aux[name] = jnp.exp(aux[f"white_{name}"][0:1]).repeat(
                    X.shape[0], axis=0
                )
                assert aux[name].shape == X.shape
            else:
                aux[name] = jnp.exp(aux[f"white_{name}"][0]).repeat(X.shape[0])
                assert aux[name].size == X.shape[0]

    ell, sigma, omega = aux["ell"], aux["sigma"], aux["omega"]

    cov = gibbs_kernel(X, X, ell, ell, sigma, sigma)
    stable_cov = add_to_diagonal(cov, 0.0, JITTER)

    data_keys = jax.random.split(data_key, 2)
    # generate a sample with zero mean
    y_clean = jax.random.multivariate_normal(
        data_keys[0], mean=jnp.zeros(X.shape[0]), cov=stable_cov
    )

    y_noisy = y_clean + jax.random.normal(data_keys[1], shape=(X.shape[0],)) * omega

    return y_noisy, y_clean, ell, sigma, omega


if __name__ == "__main__":
    # Enable X64
    jax.config.update("jax_enable_x64", True)

    names = ["ell", "sigma", "omega"]
    try:
        latent_seed = int(sys.argv[1])
    except IndexError:
        latent_seed = 0
    try:
        data_seed = int(sys.argv[2])
    except IndexError:
        data_seed = 1
    try:
        flex_code = sys.argv[3]
    except IndexError:
        flex_code = "1_1_1"

    latent_key = jax.random.PRNGKey(latent_seed)
    data_key = jax.random.PRNGKey(data_seed)

    flex_dict = {
        name: True if x == "1" else False
        for name, x in zip(names, flex_code.split("_"))
    }
    X_key = jax.random.PRNGKey(latent_seed + 1000)
    X = jax.random.uniform(X_key, shape=(100, 1))
    print(
        f"Generating data with latent seed {latent_seed}, data seed {data_seed} and flex_dict {flex_dict}"
    )
    y_noisy, y_clean, ell, sigma, omega = generate_heinonen_gp_data(
        X, latent_key, data_key, flex_dict=flex_dict
    )

    prefix = f"gen_{latent_seed}_{data_seed}_{flex_code}"

    data = {
        prefix: {
            "X": X,
            "y_clean": y_clean,
            "y_noisy": y_noisy,
            "ell": ell,
            "sigma": sigma,
            "omega": omega,
        }
    }
    savemat(f"data/{prefix}.mat", data)


def get_mcycle():
    X, y, _ = rd.MotorcycleHelmet().get_data()
    return X, y


def get_jump1d():
    X, y, _ = rd.Jump1D().get_data()
    return X, y


def get_nonstat2d():
    X, y, _ = rd.NonStat2D(samples=20**2, noise_variance=0.0).get_data()
    noise = jax.random.normal(jax.random.PRNGKey(121), shape=y.shape) * 0.5
    y = y + noise * jnp.abs(X[:, 0] + X[:, 1])
    return X, y


def get_simulated():
    X_key = jax.random.PRNGKey(0)
    X = jax.random.uniform(X_key, shape=(300, 1), minval=-3, maxval=3).sort(axis=0)
    latent_key = jax.random.PRNGKey(203)
    data_key = jax.random.PRNGKey(303)
    y_noisy, y_clean, ell, sigma, omega = generate_heinonen_gp_data(
        X, latent_key, data_key, flex_dict={"ell": True, "sigma": True, "omega": True}
    )
    return X, y_noisy


# def get_nonstat2d(seed=999, n_train=121):
#     key = jax.random.PRNGKey(seed)
#     key2 = jax.random.PRNGKey(seed + 1)

#     def noise(x):
#         noise_val = 5.0
#         maxval, minval = 1.0, -0.6
#         x0 = (x[0] - minval) / (maxval - minval)
#         x1 = (x[1] - minval) / (maxval - minval)
#         return x0 * noise_val + x1 / 2 * noise_val

#     def f(x):
#         b = jnp.pi * (2 * x[0] + 0.5 * x[1] + 1)
#         return 0.1 * (jnp.sin(b * x[0]) + jnp.sin(b * x[1]))

#     x = jax.random.uniform(key, shape=(int(n_train / 0.8), 2), minval=-0.5, maxval=1.0)
#     y = jax.vmap(f)(x) + jax.vmap(noise)(x)

#     x = MinMaxScaler().fit_transform(x)

#     return x, y
