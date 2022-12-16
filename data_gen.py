from itertools import product
from scipy.io import savemat
import jax

import matplotlib.pyplot as plt

from data import generate_heinonen_gp_data

jax.config.update("jax_enable_x64", True)

names = ["ell", "sigma", "omega"]
idx = 0
for flex_code_raw in product(["1", "0"], repeat=3):
    flex_code = "_".join(flex_code_raw)
    flex_dict = {
        name: True if x == "1" else False
        for name, x in zip(names, flex_code.split("_"))
    }
    for latent_seed, data_seed in zip(range(200, 220), range(300, 320)):

        latent_key = jax.random.PRNGKey(latent_seed)
        data_key = jax.random.PRNGKey(data_seed)

        X_key = jax.random.PRNGKey(idx + 10000)
        idx += 1
        X = jax.random.uniform(X_key, shape=(160, 1)).sort(axis=0)

        y_noisy, y_clean, ell, sigma, omega = generate_heinonen_gp_data(
            X, latent_key=latent_key, data_key=data_key, flex_dict=flex_dict
        )

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

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(X, y_clean, "o", label="y")
        ax[0].plot(X, y_noisy, "o", label="y_noisy")
        ax[0].fill_between(
            X.ravel(),
            y_clean - 2 * omega,
            y_clean + 2 * omega,
            alpha=0.5,
            label="95% noise",
        )
        ax[1].plot(X, ell, label="ell")
        ax[1].plot(X, sigma, label="sigma")
        ax[1].plot(X, omega, label="omega")
        ax[0].legend()
        ax[1].legend()
        fig.savefig(f"dump/{prefix}.png")

        savemat(f"data/{prefix}.mat", data)


print("Done")
