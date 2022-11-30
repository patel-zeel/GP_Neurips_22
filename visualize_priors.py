import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

desired_mode = 1.0
log_mode = jnp.log(desired_mode)
mu = 0.1
scale = jnp.sqrt(mu - log_mode)


x = jnp.linspace(0.01, 3.99, 100)
log_x = jnp.log(x)


# prior = tfd.Exponential(rate=-jnp.log(0.05))
prior = tfd.LogNormal(loc=mu, scale=scale)
log_prior = tfb.Log()(prior)

ax[0].plot(x, prior.prob(x))
ax[0].axvline(1.0, color="red")
ax[0].axvline(0.8, color="red")
ax[0].set_title("Prior")

ax[1].plot(log_x, log_prior.prob(log_x))
ax[1].axvline(jnp.log(1.0), color="red")
ax[1].axvline(jnp.log(0.8), color="red")
ax[1].set_title("Log Prior")


plt.savefig(f"figures/priors.png")
