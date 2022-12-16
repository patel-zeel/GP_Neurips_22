import jax

import matplotlib.pyplot as plt
from data import generate_heinonen_gp_data

jax.config.update("jax_enable_x64", True)

names = ["ell", "sigma", "omega"]
flex_code = "1_1_1"
flex_dict = {
    name: True if x == "1" else False for name, x in zip(names, flex_code.split("_"))
}
idx = 0
for latent_seed, data_seed in zip(range(200, 210), range(300, 310)):

    latent_key = jax.random.PRNGKey(latent_seed)

    data_key = jax.random.PRNGKey(data_seed)

    X_key = jax.random.PRNGKey(latent_seed + 1000)
    X = jax.random.uniform(X_key, shape=(160, 1)).sort(axis=0)

    print(
        f"Generating data with latent seed {latent_seed}, data seed {data_seed} and flex_dict {flex_dict}"
    )
    y_noisy, y_clean, ell, sigma, omega = generate_heinonen_gp_data(
        X, latent_key=latent_key, data_key=data_key, flex_dict=flex_dict
    )

print("Done")
