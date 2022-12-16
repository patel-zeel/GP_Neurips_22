import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import optax

distance_jitter = 0.0
SIZE_SMALL = 9
DEFAULT_WIDTH = 6.0
DEFAULT_HEIGHT = 1.5


def squared_distance(X1, X2):
    return jnp.square(X1 - X2).sum() + distance_jitter


def add_to_diagonal(K, value, jitter):
    diag_indices = jnp.diag_indices_from(K)
    return K.at[diag_indices].set(K[diag_indices] + value + jitter)


def repeat_to_size(value, size):
    if value.size == 1:
        return jnp.repeat(value, size)
    elif value.size == size:
        return value
    else:
        raise ValueError("value.size must be 1 or size")


def latexify(
    width_scale_factor=1,
    height_scale_factor=1,
    fig_width=None,
    fig_height=None,
    font_size=SIZE_SMALL,
):
    f"""
    width_scale_factor: float, DEFAULT_WIDTH will be divided by this number, DEFAULT_WIDTH is page width: {DEFAULT_WIDTH} inches.
    height_scale_factor: float, DEFAULT_HEIGHT will be divided by this number, DEFAULT_HEIGHT is {DEFAULT_HEIGHT} inches.
    fig_width: float, width of the figure in inches (if this is specified, width_scale_factor is ignored)
    fig_height: float, height of the figure in inches (if this is specified, height_scale_factor is ignored)
    font_size: float, font size
    """
    if fig_width is None:
        fig_width = DEFAULT_WIDTH / width_scale_factor
    if fig_height is None:
        fig_height = DEFAULT_HEIGHT / height_scale_factor

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams["pdf.fonttype"] = 42

    # Font sizes
    # SIZE_MEDIUM = 14
    # SIZE_LARGE = 24
    # https://stackoverflow.com/a/39566040
    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size - 2)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title

    # latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    plt.rcParams["backend"] = "ps"
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(fig_width, fig_height))


def train_fn(loss_fn, init_raw_params, optimizer, n_iters=1, lax_scan=True):
    state = optimizer.init(init_raw_params)

    # dry run
    # loss_fn(init_raw_params)

    if lax_scan:
        value_and_grad_fn = jax.value_and_grad(loss_fn)

        def step(raw_params_and_state, aux):
            raw_params, state = raw_params_and_state
            loss, grads = value_and_grad_fn(raw_params)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            return (raw_params, state), (raw_params, loss)

        (raw_params, state), (raw_params_history, loss_history) = jax.lax.scan(
            f=step, init=(init_raw_params, state), xs=None, length=n_iters
        )
    else:
        raw_params_history = []
        loss_history = []
        raw_params = init_raw_params
        grad_fn = jax.grad(loss_fn)
        for _ in range(n_iters):
            loss = loss_fn(raw_params)
            grads = grad_fn(raw_params)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            raw_params_history.append(raw_params)
            loss_history.append(loss)
        loss_history = jnp.array(loss_history)
    return {
        "raw_params": raw_params,
        "raw_params_history": raw_params_history,
        "loss_history": loss_history,
    }
