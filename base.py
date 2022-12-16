# JAX
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp

# TFD
import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

# Utils
from utils import add_to_diagonal, repeat_to_size, squared_distance

# Enable X64
jax.config.update("jax_enable_x64", True)

# Some constants and functions
JITTER = 1e-6
cho_solve = jsp.linalg.cho_solve
solve_tri = jsp.linalg.solve_triangular
sqr_distance_fn = jax.vmap(squared_distance, in_axes=(None, 0))
sqr_distance_fn = jax.vmap(sqr_distance_fn, in_axes=(0, None))


def get_rbf(lengthscale, scale):
    def rbf(X1, X2):
        X1 = X1 / lengthscale
        X2 = X2 / lengthscale
        return scale * jnp.exp(-0.5 * sqr_distance_fn(X1, X2))

    return rbf


def get_log_normal(desired_mode):
    log_mode = jnp.log(desired_mode)
    mu = 0.0
    scale = jnp.sqrt(mu - log_mode)
    return tfb.Log()(tfd.Normal(loc=mu, scale=scale))


def get_latent_chol(X, ell, sigma, return_kernel_fn=False):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    kernel_fn = get_rbf(lengthscale=ell, scale=sigma)
    cov = kernel_fn(X, X)
    noisy_cov = add_to_diagonal(cov, 0.0, JITTER)
    chol = jnp.linalg.cholesky(noisy_cov)
    if return_kernel_fn:
        return chol, kernel_fn
    else:
        return chol


def get_white(X, h, ell, sigma, scalar):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if bool(scalar) is True:
        assert h.size == 1
        return jnp.log(h)

    log_h = jnp.log(repeat_to_size(h, X.shape[0]))
    chol = get_latent_chol(X, ell, sigma)
    return solve_tri(chol, log_h, lower=True)


def get_log_h(white_h, X, ell, sigma, return_kernel_fn=False):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if return_kernel_fn:
        chol, kernel_fn = get_latent_chol(X, ell, sigma, return_kernel_fn=True)
        return (chol @ white_h).squeeze(), chol.squeeze(), kernel_fn
    else:
        chol = get_latent_chol(X, ell, sigma, return_kernel_fn=False)
        return (chol @ white_h).squeeze(), chol.squeeze()


def predict_log_h(white_h, X, X_new, ell, sigma, scalar, return_chol=False):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        # check if iterable
        if isinstance(X_new, list):
            X_new = jtu.tree_map(lambda x: x.reshape(-1, 1), X_new)
        else:
            X_new = X_new.reshape(-1, 1)

    if bool(scalar) is True:
        white_h = white_h.reshape(())
        out = [white_h * jnp.ones(X.shape[0])]
        if isinstance(X_new, list):
            for x_new in X_new:
                out.append(white_h * jnp.ones(x_new.shape[0]))
        else:
            out.append(white_h * jnp.ones(X_new.shape[0]))

        if return_chol:
            chol = get_latent_chol(X, ell, sigma)
            out.append(chol)
        return out
    else:
        white_h = white_h.reshape(-1)
        log_h, chol, kernel_fn = get_log_h(
            white_h, X, ell, sigma, return_kernel_fn=True
        )

        out = [log_h]
        alpha = cho_solve((chol, True), log_h - log_h.mean())
        if isinstance(X_new, list):
            for x_new in X_new:
                out.append(log_h.mean() + kernel_fn(x_new, X) @ alpha)
        else:
            out.append(log_h.mean() + kernel_fn(X_new, X) @ alpha)
        if return_chol:
            out.append(chol)
        return out


def _gibbs_kernel_common(X1, X2, ell1, ell2):
    X1, X2 = X1.reshape(-1, 1), X2.reshape(-1, 1)  # 1D only
    ell1, ell2 = ell1.reshape(-1, 1), ell2.reshape(-1, 1)  # 1D only

    l_avg_square = (ell1**2 + ell2.T**2) / 2.0
    prefix_part = jnp.sqrt(ell1 * ell2.T / l_avg_square)
    squared_dist = sqr_distance_fn(X1, X2)
    exp_part = jnp.exp(-squared_dist / (2.0 * l_avg_square))

    return (prefix_part * exp_part).squeeze()


def gibbs_kernel(X1, X2, ell1, ell2, s1, s2):
    s1, s2 = s1.reshape(-1, 1), s2.reshape(-1, 1)  # 1D only
    variance = s1 * s2.T

    f = jax.vmap(_gibbs_kernel_common, in_axes=(1, 1, 1, 1), out_axes=2)
    return variance * f(X1, X2, ell1, ell2).prod(axis=2)


def stop_gradient(params, name):
    params[name] = jax.lax.stop_gradient(params[name])
    return params


def loss_fn(params, X, y, flex_dict, method, train_latent_gp_hparams=False):
    if method == "delta_inducing":
        X_base = params["X_inducing"]
    else:
        X_base = X
    param_names = ["ell", "sigma", "omega"]
    aux = {}

    lgp_ell_prior = get_log_normal(desired_mode=0.2)
    lgp_sigma_prior = tfd.Exponential(
        rate=-jnp.log(0.05 / 5.0)
    )  # PC prior for U=5.0 with 0.05 probability

    log_prior = 0.0
    for name in param_names:
        if train_latent_gp_hparams is False:
            params = stop_gradient(params, f"{name}_gp_log_ell")
            params = stop_gradient(params, f"{name}_gp_log_sigma")

        # Get the latent hyperparameters
        l_ell = jnp.exp(params[f"{name}_gp_log_ell"])
        l_sigma = jnp.exp(params[f"{name}_gp_log_sigma"])
        if flex_dict[name]:
            if method == "heinonen":
                get_train_h = lambda white_h, X: get_log_h(white_h, X, l_ell, l_sigma)
                if name == "ell":
                    get_train_h = jax.vmap(get_train_h, in_axes=(1, 1), out_axes=(1, 2))
                aux[f"log_{name}"], aux[f"chol_{name}"] = get_train_h(
                    params[f"white_{name}"], X_base
                )
            elif method == "delta_inducing":
                get_train_h = lambda x, x_new, white_h: predict_log_h(
                    white_h,
                    x,
                    x_new,
                    l_ell,
                    l_sigma,
                    scalar=not flex_dict[name],
                    return_chol=True,
                )
                if name == "ell":
                    get_train_h = jax.vmap(
                        get_train_h, in_axes=(1, 1, 1), out_axes=[1, 1, 2]
                    )
                (
                    aux[f"log_{name}_inducing"],
                    aux[f"log_{name}"],
                    aux[f"chol_{name}"],
                ) = get_train_h(X_base, X, params[f"white_{name}"])
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            get_chol = lambda x: get_latent_chol(x, l_ell, l_sigma)
            if name == "ell":
                get_chol = jax.vmap(get_chol, in_axes=1, out_axes=2)
                assert params[f"white_{name}"].size == X.shape[1]
                aux[f"log_{name}"] = jax.vmap(lambda x: x.repeat(X.shape[0]))(
                    params[f"white_{name}"].reshape(-1)
                ).T
                if method == "delta_inducing":
                    aux[f"log_{name}_inducing"] = jax.vmap(
                        lambda x: x.repeat(X_base.shape[0])
                    )(params[f"white_{name}"].reshape(-1)).T
            else:
                assert params[f"white_{name}"].size == 1
                aux[f"log_{name}"] = params[f"white_{name}"].repeat(X.shape[0])
                if method == "delta_inducing":
                    aux[f"log_{name}_inducing"] = params[f"white_{name}"].repeat(
                        X_base.shape[0]
                    )

            aux[f"chol_{name}"] = get_chol(X_base)

        # Compute latent prior

        ######## Snelson & Ghahramani FITC (To try out later)
        # kernel_fn = get_rbf(lengthscale=l_ell, variance=l_sigma)
        # K_star = kernel_fn(X, X_base)
        # subspace = K_star @ cho_solve((aux[f"chol_{name}"], True), K_star.T)
        # cov = subspace + jnp.diag(subspace) + jnp.diag(JITTER * jnp.ones(X.shape[0]))
        # aux[f"chol_{name}"] = jnp.linalg.cholesky(cov)
        # suffix = ""

        ######## our method
        suffix = "_inducing" if method == "delta_inducing" else ""
        fn = lambda x, chol: tfd.MultivariateNormalTriL(
            loc=x.mean().repeat(x.shape[0]), scale_tril=chol
        ).log_prob(x)
        if name == "ell":
            fn = jax.vmap(fn, in_axes=(1, 2))
        aux[f"log_prior_{name}"] = fn(aux[f"log_{name}{suffix}"], aux[f"chol_{name}"])
        log_prior += aux[f"log_prior_{name}"].sum()
        # print(f"Log prior {name}:", aux[f"log_prior_{name}"].sum())

        # Compute latent gp hyperparameter prior
        if train_latent_gp_hparams:
            log_prior += lgp_ell_prior.log_prob(params[f"{name}_gp_log_ell"])
            log_prior += lgp_sigma_prior.log_prob(params[f"{name}_gp_log_sigma"])

    (ell, sigma, omega) = map(lambda name: jnp.exp(aux[f"log_{name}"]), param_names)

    K_f = gibbs_kernel(X, X, ell, ell, sigma, sigma)
    K_y = add_to_diagonal(K_f, omega**2, JITTER)

    mu_f = params["global_mean"].repeat(y.shape[0])
    log_lik = tfd.MultivariateNormalFullCovariance(
        loc=mu_f, covariance_matrix=K_y
    ).log_prob(y)

    # print("Log lik: ", log_lik)
    return -(log_lik + log_prior)


def predict_fn(params, X, y, X_new, flex_dict, method):
    if method == "delta_inducing":
        X_base = params["X_inducing"]
    else:
        X_base = X

    param_names = ["ell", "sigma", "omega"]
    aux = {}

    for name in param_names:
        l_ell = jnp.exp(params[f"{name}_gp_log_ell"])
        l_sigma = jnp.exp(params[f"{name}_gp_log_sigma"])

        fn = lambda x, x_new, white_h: predict_log_h(
            X=x,
            X_new=x_new,
            white_h=white_h,
            ell=l_ell,
            sigma=l_sigma,
            scalar=not flex_dict[name],
            return_chol=False,
        )
        if name == "ell":
            if method == "delta_inducing":
                fn = jax.vmap(fn, in_axes=(1, 1, 1), out_axes=[1, 1, 1])
            elif method == "heinonen":
                fn = jax.vmap(fn, in_axes=(1, 1, 1), out_axes=[1, 1])
            else:
                raise ValueError(f"Unknown method: {method}")

        if method == "delta_inducing":
            _, aux[f"log_{name}"], aux[f"log_{name}_new"] = fn(
                X_base, [X, X_new], params[f"white_{name}"]
            )
        elif method == "heinonen":
            aux[f"log_{name}"], aux[f"log_{name}_new"] = fn(
                X_base, X_new, params[f"white_{name}"]
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    ell, sigma, omega = map(lambda name: jnp.exp(aux[f"log_{name}"]), param_names)
    ell_new, sigma_new, omega_new = map(
        lambda name: jnp.exp(aux[f"log_{name}_new"]), param_names
    )

    K = gibbs_kernel(X, X, ell, ell, sigma, sigma)
    K_noisy = add_to_diagonal(K, omega**2, JITTER)
    chol_y = jnp.linalg.cholesky(K_noisy)

    K_star = gibbs_kernel(X_new, X, ell_new, ell, sigma_new, sigma)
    K_star_star_diag = gibbs_kernel(
        X_new, X_new, ell_new, ell_new, sigma_new, sigma_new
    ).diagonal()

    pred_mean = params["global_mean"] + K_star @ cho_solve(
        (chol_y, True), y - params["global_mean"]
    )
    pred_var = (
        K_star_star_diag - (K_star @ cho_solve((chol_y, True), K_star.T)).diagonal()
    )
    return pred_mean, pred_var, ell_new, sigma_new, omega_new


experiment_scale = 1.0


def get_params(key, X, flex_dict, method, default=False, n_inducing=None):
    keys = jax.random.split(key, 3)
    if n_inducing is None:
        n_inducing = int(jnp.log(X.shape[0])) + 1
    else:
        n_inducing = int(n_inducing)

    if method == "delta_inducing":
        inducing_key = jax.random.split(keys[0], 1)[0]
        X_inducing = jax.random.choice(inducing_key, X, shape=(n_inducing,))
        X_base = X_inducing
    elif method == "heinonen":
        X_base = X
    else:
        raise ValueError(f"Invalid method: {method}")

    if default is False:
        h_ell = jax.random.uniform(
            keys[0], minval=0.03 * experiment_scale, maxval=0.3 * experiment_scale
        )
        h_sigma = jax.random.uniform(
            keys[1], minval=0.1 * experiment_scale, maxval=0.5 * experiment_scale
        )
        h_omega = jax.random.uniform(
            keys[2], minval=0.01 * experiment_scale, maxval=0.10 * experiment_scale
        )
    else:
        h_ell = jnp.array(0.05)
        h_sigma = jnp.array(0.3)
        h_omega = jnp.array(0.05)

    ell_whites = []
    for idx in range(X.shape[1]):
        x = X_base[:, idx : idx + 1]
        ell_whites.append(
            get_white(
                x, h_ell, ell=0.2, sigma=1.0, scalar=not flex_dict["ell"]
            ).reshape(-1, 1)
        )

    ell_whites = jnp.concatenate(ell_whites, axis=1)

    params = {
        "white_ell": ell_whites,
        "white_sigma": get_white(
            X_base,
            h_sigma,
            ell=0.2,
            sigma=1.0,
            scalar=not flex_dict["sigma"],
        ),
        "white_omega": get_white(
            X_base,
            h_omega,
            ell=0.3,
            sigma=1.0,
            scalar=not flex_dict["omega"],
        ),
        "ell_gp_log_ell": jnp.log(jnp.array(0.2)),
        "sigma_gp_log_ell": jnp.log(jnp.array(0.2)),
        "omega_gp_log_ell": jnp.log(jnp.array(0.3)),
        "ell_gp_log_sigma": jnp.log(jnp.array(1.0)),
        "sigma_gp_log_sigma": jnp.log(jnp.array(1.0)),
        "omega_gp_log_sigma": jnp.log(jnp.array(1.0)),
        "global_mean": jnp.array(0.0),
    }
    if method == "delta_inducing":
        params["X_inducing"] = X_inducing

    return params
