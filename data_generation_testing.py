import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from common import generate_heinonen_gp_data

X_key = jax.random.PRNGKey(0)
X_test_key = jax.random.PRNGKey(1)
X = jax.random.uniform(X_key, shape=(100, 1)).sort(axis=0)
X_test = jax.random.uniform(X_test_key, shape=(200, 1)).sort(axis=0)

data_key = jax.random.PRNGKey(2)
y, ell, sigma, omega = generate_heinonen_gp_data(X, key=data_key)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(X, y, "o")
ax[1].plot(X, ell, label="ell")
ax[1].plot(X, sigma, label="sigma")
ax[1].plot(X, omega, label="omega")
ax[1].legend()
fig.savefig("figures/heinonen_gen_data.png")

print("Done")
