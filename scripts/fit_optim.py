import jax
import jax.numpy as jnp
import numpy as np
import optax


def _is_gpu():
    """True when JAX's default backend is a GPU or TPU."""
    return jax.default_backend() in ("gpu", "tpu")


def _make_optimizer(memory_size=10):
    return optax.chain(
        optax.clip_by_global_norm(max_norm=1e6),
        optax.scale_by_lbfgs(memory_size=memory_size, scale_init_precond=True),
        optax.scale(-1.0),
        optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30,
            slope_rtol=1e-4,
            decrease_factor=0.5,
            increase_factor=1.5,
        ),
    )


def fit_optax(
    fitter,
    theta0=None,
    maxiter=2000,
    tol_grad=1e-5,
    tol_nll=1e-8,
    memory_size=10,
):
    """L-BFGS + Armijo backtracking via Python loop (CPU path).

    Returns (theta, logl, converged, n_iter, trace_logl, trace_gnorm).
    Bounds enforced by projected gradient (clip after apply_updates).
    """
    if theta0 is None:
        theta0 = fitter.init.copy()
    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    bounds_lo = jnp.array([b[0] for b in fitter.bounds], dtype=jnp.float64)
    bounds_hi = jnp.array([b[1] for b in fitter.bounds], dtype=jnp.float64)

    def neg_logl_fn(t):
        return -fitter._logl_from_theta(t)

    neg_logl_fn_jit = jax.jit(neg_logl_fn)
    optimizer = _make_optimizer(memory_size)

    theta = theta0_j
    opt_state = optimizer.init(theta)
    vg_fn = jax.jit(jax.value_and_grad(neg_logl_fn))
    value, grad = vg_fn(theta)

    trace_logl = []
    trace_gnorm = []
    consec = 0
    converged = False
    n_iter = 0

    for i in range(maxiter):
        logl_i = -float(value)
        gnorm_i = float(jnp.linalg.norm(grad))
        trace_logl.append(logl_i)
        trace_gnorm.append(gnorm_i)

        if not np.isfinite(logl_i) or not np.isfinite(gnorm_i):
            print(
                f"[OPTAX] step {i} non-finite  logl={logl_i}  |g|={gnorm_i}",
                flush=True,
            )
            break

        satisfied = gnorm_i < tol_grad or (
            len(trace_logl) > 1 and abs(logl_i - trace_logl[-2]) < tol_nll
        )
        consec = consec + 1 if satisfied else 0
        if consec >= 3:
            converged = True
            n_iter = i + 1
            break

        try:
            updates, opt_state = optimizer.update(
                grad, opt_state, theta,
                value=value, grad=grad, value_fn=neg_logl_fn_jit,
            )
        except Exception as exc:
            print(f"[OPTAX] step {i} optimizer.update failed: {exc}", flush=True)
            break

        theta = jnp.clip(optax.apply_updates(theta, updates), bounds_lo, bounds_hi)
        value, grad = vg_fn(theta)
        n_iter = i + 1

        if -float(value) < trace_logl[-1] - 1e-6:
            print(
                f"[OPTAX] warning: logl decreased at step {i}"
                f" ({-float(value):.6g} < {trace_logl[-1]:.6g})",
                flush=True,
            )

    return (
        np.asarray(theta, dtype=np.float64),
        -float(value),
        converged,
        n_iter,
        np.array(trace_logl),
        np.array(trace_gnorm),
    )


def fit_optax_lax(
    fitter,
    theta0=None,
    maxiter=2000,
    tol_grad=1e-5,
    tol_nll=1e-8,
    memory_size=10,
):
    """L-BFGS + Armijo via jax.lax.while_loop (GPU path, single sync at exit).

    The entire loop (including backtracking linesearch's inner while_loop) is
    compiled into one XLA program.  No Python round-trips inside the loop.
    trace_logl and trace_gnorm are returned as empty arrays.
    """
    if theta0 is None:
        theta0 = fitter.init.copy()
    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    bounds_lo = jnp.array([b[0] for b in fitter.bounds], dtype=jnp.float64)
    bounds_hi = jnp.array([b[1] for b in fitter.bounds], dtype=jnp.float64)
    _maxiter = int(maxiter)
    _tol_grad = float(tol_grad)
    _tol_nll = float(tol_nll)

    def neg_logl_fn(t):
        return -fitter._logl_from_theta(t)

    optimizer = _make_optimizer(memory_size)

    value, grad = jax.value_and_grad(neg_logl_fn)(theta0_j)
    opt_state = optimizer.init(theta0_j)

    # carry: (theta, opt_state, value, grad, n_iter, consec, converged, prev_value)
    # prev_value = +inf so the first logl-change check doesn't trigger spuriously.
    init_carry = (
        theta0_j, opt_state, value, grad,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(jnp.inf, dtype=jnp.float64),
    )

    def cond_fn(carry):
        _, _, value, grad, n_iter, _, converged, _ = carry
        is_finite = jnp.isfinite(value) & jnp.isfinite(jnp.linalg.norm(grad))
        return is_finite & ~converged & (n_iter < _maxiter)

    def body_fn(carry):
        theta, opt_state, value, grad, n_iter, consec, converged, prev_value = carry

        gnorm = jnp.linalg.norm(grad)
        satisfied = (gnorm < _tol_grad) | (jnp.abs(-value - (-prev_value)) < _tol_nll)
        new_consec = jnp.where(satisfied, consec + 1, jnp.zeros_like(consec))
        new_converged = new_consec >= 3

        def do_step(_):
            updates, new_opt_state = optimizer.update(
                grad, opt_state, theta,
                value=value, grad=grad, value_fn=neg_logl_fn,
            )
            new_theta = jnp.clip(
                optax.apply_updates(theta, updates), bounds_lo, bounds_hi
            )
            new_value, new_grad = jax.value_and_grad(neg_logl_fn)(new_theta)
            return new_theta, new_opt_state, new_value, new_grad

        def stay_put(_):
            return theta, opt_state, value, grad

        new_theta, new_opt_state, new_value, new_grad = jax.lax.cond(
            ~new_converged, do_step, stay_put, None
        )

        return (
            new_theta, new_opt_state, new_value, new_grad,
            n_iter + 1, new_consec, new_converged, value,
        )

    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    theta_f, _, value_f, _, n_iter_f, _, converged_f, _ = final_carry

    theta_f, value_f, n_iter_f, converged_f = jax.block_until_ready(
        (theta_f, value_f, n_iter_f, converged_f)
    )

    return (
        np.asarray(theta_f, dtype=np.float64),
        float(-value_f),
        bool(converged_f),
        int(n_iter_f),
        np.array([]),
        np.array([]),
    )
