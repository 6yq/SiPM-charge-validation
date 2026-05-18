import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _is_gpu():
    """True when JAX's default backend is a GPU or TPU."""
    return jax.default_backend() in ("gpu", "tpu")


def _make_optimizer(memory_size=10):
    return optax.chain(
        optax.clip_by_global_norm(max_norm=1e6),
        optax.scale_by_lbfgs(memory_size=memory_size, scale_init_precond=True),
        optax.scale(-1.0),
        optax.scale_by_zoom_linesearch(
            max_linesearch_steps=30,
            slope_rtol=1e-4,
            curv_rtol=0.9,
            approx_dec_rtol=1e-6,
            increase_factor=1.5,
            initial_guess_strategy="one",
        ),
    )


def _progress_bar_kwargs(maxiter):
    return {
        "total": maxiter,
        "unit": "step",
        "ncols": 90,
        "miniters": 50,
        "mininterval": 0,
        "maxinterval": float("inf"),
        "leave": False,
        "file": sys.stderr,
    }


def _seed_progress_bar_kwargs(n_seeds):
    return {
        "total": n_seeds,
        "unit": "seed",
        "ncols": 90,
        "miniters": 1,
        "mininterval": 0,
        "maxinterval": float("inf"),
        "leave": False,
        "file": sys.stderr,
    }


def _compile_vg(fitter):
    """JIT-compile value_and_grad once; share across seeds to avoid LLVM OOM."""

    def neg_logl(t):
        return -fitter._logl_from_theta(t)

    return jax.jit(jax.value_and_grad(neg_logl)), jax.jit(neg_logl)


def _make_step_fn(optimizer, vg_fn, value_fn_jit, bounds_lo, bounds_hi):
    """JIT one optimizer step so Optax's line-search loop is compiled once."""

    def step(theta, opt_state, value, grad):
        updates, new_opt_state = optimizer.update(
            grad,
            opt_state,
            theta,
            value=value,
            grad=grad,
            value_fn=value_fn_jit,
        )
        new_theta = jnp.clip(optax.apply_updates(theta, updates), bounds_lo, bounds_hi)
        new_value, new_grad = vg_fn(new_theta)
        return new_theta, new_opt_state, new_value, new_grad

    return jax.jit(step)


def fit_optax(
    fitter,
    theta0=None,
    maxiter=2000,
    tol_grad=1e-2,
    tol_nll=1e-6,
    memory_size=10,
    desc="L-BFGS",
    vg_fn=None,
    value_fn_jit=None,
    progress=True,
):
    """L-BFGS + Wolfe zoom line search via Python loop (CPU path).

    Returns (theta, logl, converged, n_iter, trace_logl, trace_gnorm).
    Bounds enforced by projected gradient (clip after apply_updates).
    Pass pre-compiled vg_fn/value_fn_jit from fit_multistart to avoid
    recompilation across seeds.
    """
    if theta0 is None:
        theta0 = fitter.init.copy()
    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    bounds_lo = jnp.array([b[0] for b in fitter.bounds], dtype=jnp.float64)
    bounds_hi = jnp.array([b[1] for b in fitter.bounds], dtype=jnp.float64)

    if vg_fn is None or value_fn_jit is None:
        vg_fn, value_fn_jit = _compile_vg(fitter)

    optimizer = _make_optimizer(memory_size)
    step_fn = _make_step_fn(optimizer, vg_fn, value_fn_jit, bounds_lo, bounds_hi)

    theta = theta0_j
    opt_state = optimizer.init(theta)
    value, grad = vg_fn(theta)

    trace_logl = []
    trace_gnorm = []
    consec = 0
    converged = False
    n_iter = 0

    pbar = (
        _tqdm(
            desc=desc,
            **_progress_bar_kwargs(maxiter),
        )
        if _HAS_TQDM and progress
        else None
    )

    for i in range(maxiter):
        logl_i = -float(value)
        gnorm_i = float(jnp.linalg.norm(grad))
        trace_logl.append(logl_i)
        trace_gnorm.append(gnorm_i)

        if pbar is not None:
            pbar.set_postfix(logl=f"{logl_i:.3f}", g=f"{gnorm_i:.1e}", refresh=False)
            pbar.update(1)

        if not np.isfinite(logl_i) or not np.isfinite(gnorm_i):
            if pbar is not None:
                pbar.write(
                    f"[OPTAX] step {i} non-finite  logl={logl_i}  |g|={gnorm_i}",
                    file=sys.stderr,
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
            theta, opt_state, value, grad = step_fn(theta, opt_state, value, grad)
        except Exception as exc:
            if pbar is not None:
                pbar.write(f"[OPTAX] step {i} update failed: {exc}", file=sys.stderr)
            break

        n_iter = i + 1

    if pbar is not None:
        pbar.close()

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
    tol_grad=1e-2,
    tol_nll=1e-6,
    memory_size=10,
    desc="L-BFGS (lax)",
    **kwargs,  # absorb vg_fn/value_fn_jit passed by fit_multistart
):
    """L-BFGS + Wolfe zoom line search via lax loop (GPU path, single sync).

    The entire loop (including zoom linesearch's inner while_loop) is
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

    init_carry = (
        theta0_j,
        opt_state,
        value,
        grad,
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
                grad,
                opt_state,
                theta,
                value=value,
                grad=grad,
                value_fn=neg_logl_fn,
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
            new_theta,
            new_opt_state,
            new_value,
            new_grad,
            n_iter + 1,
            new_consec,
            new_converged,
            value,
        )

    print(f"[OPTAX] {desc}: compiling + running (maxiter={_maxiter})...", flush=True)
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    theta_f, _, value_f, _, n_iter_f, _, converged_f, _ = final_carry

    theta_f, value_f, n_iter_f, converged_f = jax.block_until_ready(
        (theta_f, value_f, n_iter_f, converged_f)
    )

    print(
        f"[OPTAX] {desc}: done  n_iter={int(n_iter_f)}  logl={float(-value_f):.4f}",
        flush=True,
    )

    return (
        np.asarray(theta_f, dtype=np.float64),
        float(-value_f),
        bool(converged_f),
        int(n_iter_f),
        np.array([]),
        np.array([]),
    )


# ─── multi-start ─────────────────────────────────────────────────────────────

_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17)


def _halton(index, base):
    f = 1.0
    r = 0.0
    i = int(index)
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _seed_quantile(index, dim):
    # Avoid exact bounds, but still span nearly the full allowed interval.
    return 0.02 + 0.96 * _halton(index, _HALTON_BASES[dim])


def _make_seeds(fitter, theta0, n_seeds=3):
    """Generate physically diverse initial vectors.

    Seed 0: default theta0.
    Subsequent seeds use a deterministic low-discrepancy design over the
    strongest local-minimum degeneracies: occupancy, AP count, AP charge,
    Gen-Poisson dispersion, and optional dark-count occupancy.
    """
    theta0 = np.asarray(theta0, dtype=float)
    names = list(fitter.param_names)
    log_rho_idx = names.index("log_rho") if "log_rho" in names else None
    logit_beta_idx = names.index("logit_beta") if "logit_beta" in names else None
    log_xi_idx = names.index("log_xi") if "log_xi" in names else None
    log_mu_dark_idx = names.index("log_mu_dark") if "log_mu_dark" in names else None

    n_seeds = max(1, int(n_seeds))
    bounds_lo = [b[0] for b in fitter.bounds]
    bounds_hi = [b[1] for b in fitter.bounds]
    seeds = [np.clip(theta0.copy(), bounds_lo, bounds_hi)]

    dims = []
    for idx in (log_rho_idx, logit_beta_idx, log_xi_idx, log_mu_dark_idx):
        if idx is not None:
            lo, hi = fitter.bounds[idx]
            dims.append((idx, float(lo), float(hi)))

    for seed_i in range(1, n_seeds):
        seed = theta0.copy()
        for dim_i, (idx, lo, hi) in enumerate(dims):
            q = _seed_quantile(seed_i, dim_i)
            seed[idx] = float(lo + q * (hi - lo))
        seeds.append(np.clip(seed, bounds_lo, bounds_hi))
    return seeds


def fit_multistart(
    fitter,
    theta0=None,
    n_seeds=3,
    use_lax=False,
    maxiter=2000,
    tol_grad=1e-2,
    tol_nll=1e-6,
    memory_size=10,
):
    """Run optimizer from n_seeds starting points; return best (highest logl) result.

    Returns (theta, logl, converged, n_iter, trace_logl, trace_gnorm)
    where trace_* comes from the winning seed.
    Pre-compiles value_and_grad once and shares it across seeds (CPU path only)
    to avoid repeated LLVM JIT compilation and OOM.
    """
    if theta0 is None:
        theta0 = fitter.init.copy()
    theta0 = np.asarray(theta0, dtype=float)

    fit_fn = fit_optax_lax if use_lax else fit_optax
    seeds = _make_seeds(fitter, theta0, n_seeds)

    # Compile once per multistart run; lax path ignores these via **kwargs.
    vg_fn, value_fn_jit = (None, None) if use_lax else _compile_vg(fitter)

    best_logl = -np.inf
    best_result = None
    seed_pbar = (
        _tqdm(desc="multi-start", **_seed_progress_bar_kwargs(len(seeds)))
        if _HAS_TQDM and len(seeds) > 1
        else None
    )

    for i, seed in enumerate(seeds):
        result = fit_fn(
            fitter,
            theta0=seed,
            maxiter=maxiter,
            tol_grad=tol_grad,
            tol_nll=tol_nll,
            memory_size=memory_size,
            desc=f"seed {i+1}/{len(seeds)}",
            vg_fn=vg_fn,
            value_fn_jit=value_fn_jit,
            progress=(seed_pbar is None),
        )
        theta_i, logl_i = result[0], result[1]
        if logl_i > best_logl:
            best_logl = logl_i
            best_result = result
        if seed_pbar is not None:
            seed_pbar.set_postfix(best=f"{best_logl:.3f}", refresh=False)
            seed_pbar.update(1)

    if seed_pbar is not None:
        seed_pbar.close()
    return best_result
