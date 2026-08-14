from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.optimize


def _alg(alg: Callable, z0: np.ndarray, *args, **kwargs):
    def wrapper(z_):
        return alg(z_, *args, **kwargs)

    return scipy.optimize.root(wrapper, z0)


def _ode_explicit(
    ode: Callable,
    dt: np.ndarray,
    x0: np.ndarray,
    *args,
    **kwargs,
):

    return x0 + dt * ode(x0, *args, **kwargs)


def _ode_implicit(
    ode: Callable,
    dt: np.ndarray,
    x0: np.ndarray,
    *args,
    **kwargs,
):

    def wrapper(x_):
        x_ = x_.reshape(x0.shape)
        return (x_ - x0 - dt * ode(x_, *args, **kwargs)).ravel()

    return _alg(wrapper, x0.ravel())


def _ode_crank_nicolson(
    ode: Callable,
    dt: np.ndarray,
    x0: np.ndarray,
    *args,
    **kwargs,
):

    def wrapper(x_):
        x_ = x_.reshape(x0.shape)
        return (
            x_ - x0 - 0.5 * dt * (ode(x_, *args, **kwargs) + ode(x0, *args, **kwargs))
        ).ravel()

    return _alg(wrapper, x0.ravel())


def _dae_explicit(
    ode: Callable,
    alg: Callable,
    dt: np.ndarray,
    x0: np.ndarray,
    z0: np.ndarray,
    *args,
    **kwargs,
):

    x = x0 + dt * ode(x0, z0, *args, **kwargs)

    def wrapper(z_):
        return alg(x, z_, *args, **kwargs)

    z = _alg(wrapper, z0.ravel()).reshape(z0.shape)
    return x, z


def _dae_implicit(
    ode: Callable,
    alg: Callable,
    dt: np.ndarray,
    x0: np.ndarray,
    z0: np.ndarray,
    *args,
    **kwargs,
):

    def wrapper(xz_):
        x_ = xz_[: x0.size].reshape(x0.shape)
        z_ = xz_[x0.size :].reshape(z0.shape)
        return np.hstack(
            [
                x_ - x0 - dt * ode(x_, z_, *args, **kwargs),
                alg(x_, z_, *args, **kwargs),
            ]
        )

    xz0 = np.hstack([x0.ravel(), z0.ravel()])
    xz = _alg(wrapper, xz0)
    x = xz[: x0.size].reshape(x0.shape)
    z = xz[x0.size :].reshape(z0.shape)
    return x, z


def _dae_crank_nicolson(
    ode: Callable,
    alg: Callable,
    dt: np.ndarray,
    x0: np.ndarray,
    z0: np.ndarray,
    *args,
    **kwargs,
):
    def wrapper(xz_):
        x_ = xz_[: x0.size].reshape(x0.shape)
        z_ = xz_[x0.size :].reshape(z0.shape)
        return np.hstack(
            [
                x_
                - x0
                - 0.5
                * dt
                * (ode(x_, z_, *args, **kwargs) + ode(x0, z0, *args, **kwargs)),
                alg(x_, z_, *args, **kwargs),
            ]
        )

    xz0 = np.hstack([x0.ravel(), z0.ravel()])
    xz = _alg(wrapper, xz0)
    return xz[: x0.size].reshape(x0.shape), xz[x0.size :].reshape(z0.shape)


def _dae_icsolve(
    alg: Callable,
    x0: np.ndarray,
    z0: np.ndarray,
    *args,
    **kwargs,
):
    def wrapper(z_):
        return alg(x0, z_, *args, **kwargs)

    return _alg(wrapper, z0.ravel())


_ODE_DISPATCH = {
    "explicit": _ode_explicit,
    "implicit": _ode_implicit,
    "crank-nicolson": _ode_crank_nicolson,
}

_DAE_DISPATCH = {
    "explicit": _dae_explicit,
    "implicit": _dae_implicit,
    "crank-nicolson": _dae_crank_nicolson,
}


def integrate_ode(
    ode: Callable,
    t: np.ndarray,
    x0: np.ndarray,
    u: np.ndarray | None = None,
    *args,
    **kwargs,
):
    method = kwargs.pop("method", "explicit")
    x = [x0]
    for k, dt in enumerate(np.diff(t), start=1):
        if u is None:
            x0 = _ODE_DISPATCH[method](ode, dt, x0, *args, **kwargs)
        else:
            x0 = _ODE_DISPATCH[method](ode, dt, x0, u[:, k - 1 : k], *args, **kwargs)

        x.append(x0)
    return np.hstack(x)


def integrate_dae(
    ode: Callable,
    alg: Callable,
    t: np.ndarray,
    x0: np.ndarray,
    z0: np.ndarray,
    u: np.ndarray | None = None,
    *args,
    **kwargs,
):
    method = kwargs.pop("method", "explicit")
    x = [x0]
    z0 = _dae_icsolve(alg, x0, z0, *args, **kwargs)
    z = [z0]
    for k, dt in enumerate(np.diff(t), start=1):
        if u is None:
            x0, z0 = _DAE_DISPATCH[method](ode, alg, dt, x0, z0, *args, **kwargs)
        else:
            x0, z0 = _DAE_DISPATCH[method](
                ode, alg, dt, x0, z0, u[:, k - 1 : k], *args, **kwargs
            )
    return np.hstack(x), np.hstack(z)
