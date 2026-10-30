from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import tqdm.auto as tqdm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, root
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class Solution(NamedTuple):
    x: np.ndarray
    z: np.ndarray


def jacobian(
    fun: Callable[..., np.ndarray], arr: np.ndarray, sparse: bool = False
) -> np.ndarray | csc_matrix:
    """
    Estimates jacobian of
        arr: (n, 1)
        fun: (m, 1)
    Returns the Jacobian matrix of shape (m, n).
    """
    n = arr.shape[0]
    m = fun(arr).shape[0]
    jac = np.zeros((m, n))

    epsilon = np.sqrt(np.finfo(float).eps)
    h = np.zeros_like(arr)
    h[0, :] = epsilon

    for i in range(n):
        jac[:, i : i + 1] = (fun(arr + h) - fun(arr - h)) / (2 * epsilon)
        h = np.roll(h, 1, axis=0)
    if sparse:
        return csc_matrix(jac)
    return jac


class DAESolver:
    """
    Specific DAESolver for index-1 systems
    """

    def __init__(
        self,
        *,
        f: Callable[..., np.ndarray],
        g: Callable[..., np.ndarray],
        linear: bool = False,
        sparse: bool = False,
    ):
        """
        Initialize the DAE solver with the given functions and options.

        Parameters
        ----------
        f : Callable[..., np.ndarray]
            The differential equation function.
        g : Callable[..., np.ndarray]
            The algebraic equation function.
        linear : bool, optional
            Whether the problem 0=g(...,z,...) is linear with respect to z and will converge in a single newton iteration, by default False.
        sparse : bool, optional
            Whether to use sparse matrices, by default False.
        """
        self.f = f
        self.g = g
        self.linear = linear
        self.sparse = sparse
        return

    def dfdx(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray
    ) -> np.ndarray | csc_matrix:
        """
        Compute the Jacobian of f with respect to x.

        Returns a sparse matrix if self.sparse is True.

        Parameters
        ----------
        x : np.ndarray
            The state vector.
            shape (dim_x, 1)
        z : np.ndarray
            The algebraic variable vector.
            shape (dim_z, 1)
        u : np.ndarray
            The control input vector.
            shape (dim_u, 1)

        Returns
        -------
        np.ndarray | csc_matrix
            The Jacobian matrix df/dx.
        """
        dfdx = jacobian(lambda x_: self.f(x_, z, u), x, self.sparse)
        if self.sparse:
            return csc_matrix(dfdx)
        return dfdx

    def dfdz(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray
    ) -> np.ndarray | csc_matrix:
        """
        Compute the Jacobian of f with respect to z.

        Returns a sparse matrix if self.sparse is True.

        Parameters
        ----------
        x : np.ndarray
            The state vector.
            shape (dim_x, 1)
        z : np.ndarray
            The algebraic variable vector.
            shape (dim_z, 1)
        u : np.ndarray
            The control input vector.
            shape (dim_u, 1)

        Returns
        -------
        np.ndarray | csc_matrix
            The Jacobian matrix df/dz.
        """
        return jacobian(lambda z_: self.f(x, z_, u), z, sparse=self.sparse)

    def dfdu(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray
    ) -> np.ndarray | csc_matrix:
        """
        Compute the Jacobian of f with respect to u.

        Returns a sparse matrix if self.sparse is True.

        Parameters
        ----------
        x : np.ndarray
            The state vector.
            shape (dim_x, 1)
        z : np.ndarray
            The algebraic variable vector.
            shape (dim_z, 1)
        u : np.ndarray
            The control input vector.
            shape (dim_u, 1)

        Returns
        -------
        np.ndarray | csc_matrix
            The Jacobian matrix df/du.
        """
        return jacobian(lambda u_: self.f(x, z, u_), u, sparse=self.sparse)

    def dgdx(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray
    ) -> np.ndarray | csc_matrix:
        """
        Compute the Jacobian of g with respect to x.

        Returns a sparse matrix if self.sparse is True.

        Parameters
        ----------
        x : np.ndarray
            The state vector.
            shape (dim_x, 1)
        z : np.ndarray
            The algebraic variable vector.
            shape (dim_z, 1)
        u : np.ndarray
            The control input vector.
            shape (dim_u, 1)

        Returns
        -------
        np.ndarray | csc_matrix
            The Jacobian matrix dg/dx.
        """
        return jacobian(lambda x_: self.g(x_, z, u), x, sparse=self.sparse)

    def dgdz(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray
    ) -> np.ndarray | csc_matrix:
        """
        Compute the Jacobian of g with respect to z.

        Returns a sparse matrix if self.sparse is True.

        Parameters
        ----------
        x : np.ndarray
            The state vector.
            shape (dim_x, 1)
        z : np.ndarray
            The algebraic variable vector.
            shape (dim_z, 1)
        u : np.ndarray
            The control input vector.
            shape (dim_u, 1)

        Returns
        -------
        np.ndarray | csc_matrix
            The Jacobian matrix dg/dz.
        """
        return jacobian(lambda z_: self.g(x, z_, u), z, sparse=self.sparse)

    def dgdu(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray
    ) -> np.ndarray | csc_matrix:
        """
        Compute the Jacobian of g with respect to u.

        Returns a sparse matrix if self.sparse is True.

        Parameters
        ----------
        x : np.ndarray
            The state vector.
            shape (dim_x, 1)
        z : np.ndarray
            The algebraic variable vector.
            shape (dim_z, 1)
        u : np.ndarray
            The control input vector.
            shape (dim_u, 1)

        Returns
        -------
        np.ndarray | csc_matrix
            The Jacobian matrix dg/du.
        """
        return jacobian(lambda u_: self.g(x, z, u_), u, sparse=self.sparse)

    def dxdt(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.f(x, z, u)

    def dzdt(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray, dudt: np.ndarray
    ) -> np.ndarray:
        dgdz = self.dgdz(x, z, u)
        dgdx = self.dgdx(x, z, u)
        dxdt = self.dxdt(x, z, u)
        return self.linsolve(dgdz, -dgdx @ dxdt - self.dgdu(x, z, u) @ dudt)

    def linsolve(self, A: np.ndarray | csc_matrix, b: np.ndarray) -> np.ndarray:
        solution = spsolve(A, b) if self.sparse else np.linalg.solve(A, b)
        return np.asarray(solution).reshape(A.shape[0], -1)

    def calc_ic(self, x0: np.ndarray, z0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculate the consistent initial condition for the algebraic variable z0
        given the initial state x0 and control input u0.
        Returns a consistent initial condition for z0 that satisfies g(x0, z0, u0) = 0.

        """
        if self.linear:
            A = self.dgdz(x0, z0, u)
            b = -self.g(x0, z0, u)
            return z0 + self.linsolve(A, b)

        dim_z = z0.shape[0]

        sol = root(lambda z: self.g(x0, z.reshape((dim_z, 1)), u).ravel(), z0)
        if not sol.success:
            raise RuntimeError("Failed to find consistent initial condition for z0")
        return sol.x.reshape((dim_z, 1))

    def integrate_reduced(
        self, t: np.ndarray, x0: np.ndarray, z0: np.ndarray, u: np.ndarray
    ) -> Solution:
        """
        Integrate the reduced DAE system over the given time array.
        Returns the state trajectory only, assuming the algebraic variables are eliminated.
        Applies a reduction by eliminating the algebraic variables using the consistent initial
        """
        dim_x = x0.shape[0]
        dim_z = z0.shape[0]
        lut_u = interp1d(
            t,
            u,
            axis=1,
            kind="linear",
            fill_value=(u[:, 0], u[:, -1]),  # type: ignore
            bounds_error=False,
        )

        def as_ode(T, Y):
            x = Y[:dim_x][:, np.newaxis]
            z = Y[dim_x : dim_x + dim_z][:, np.newaxis]
            u = lut_u(T)[:, np.newaxis]

            dudt = ((lut_u(T + 1e-6) - lut_u(T - 1e-6)) / (2e-6))[:, np.newaxis]
            dxdt = self.dxdt(x, z, u)
            dzdt = self.dzdt(x, z, u, dudt)
            return np.vstack([dxdt, dzdt]).ravel()

        z0 = self.calc_ic(x0, z0, u[:, 0:1])
        Y0 = np.vstack([x0, z0]).ravel()
        solution = solve_ivp(
            lambda t, Y: as_ode(t, Y),
            t_span=(t[0], t[-1]),
            y0=Y0,
            t_eval=t,
            first_step=t[1] - t[0],
            max_step=t[1] - t[0],
        ).y
        x = solution[:dim_x, :]
        z = solution[dim_x : dim_x + dim_z, :]

        return Solution(x=x, z=z)

    def integrate_imex(
        self, t: np.ndarray, x0: np.ndarray, z0: np.ndarray, u: np.ndarray
    ) -> Solution:
        """
        Integrate the DAE system over the given time array.
        Returns the state and algebraic variable trajectories.
        needs u as a constant (dim_u,1)
        """
        k = t.shape[0]
        dim_x = x0.shape[0]
        dim_z = z0.shape[0]

        x = np.zeros((dim_x, k))
        z = np.zeros((dim_z, k))
        z0 = self.calc_ic(x0, z0, u[:, 0:1])

        x[:, 0:1] = x0
        z[:, 0:1] = z0
        for i, dt in tqdm.tqdm(
            enumerate(np.diff(t), start=1), total=len(np.diff(t)), initial=1
        ):
            x0 = x[:, i - 1 : i] + dt * self.dxdt(x0, z0, u[:, i - 1 : i])
            z0 = self.calc_ic(x0, z0, u[:, i : i + 1])
            x[:, i : i + 1] = x0
            z[:, i : i + 1] = z0
        return Solution(x=x, z=z)

    def integrate_implicit(
        self, t: np.ndarray, x0: np.ndarray, z0: np.ndarray, u: np.ndarray
    ) -> Solution:
        """
        Integrate the DAE system using an implicit method over the given time array.
        Returns the state and algebraic variable trajectories.
        Assumes the control input u is constant (dim_u, 1).
        """
        k = t.shape[0]
        dim_x = x0.shape[0]
        dim_z = z0.shape[0]

        def residuals(Y, x_old, z_old, u_old, u_new, dt):
            x_new = Y[:dim_x][:, np.newaxis]
            z_new = Y[dim_x : dim_x + dim_z][:, np.newaxis]

            f_new = self.f(x_new, z_new, u_new)
            f_old = self.f(x_old, z_old, u_old)
            g_new = self.g(x_new, z_new, u_new)
            res = np.vstack([(x_new - x_old) - dt * 0.5 * (f_new + f_old), g_new])
            return res.ravel()

        x = np.zeros((dim_x, k))
        z = np.zeros((dim_z, k))
        z0 = self.calc_ic(x0, z0, u[:, 0:1])

        x[:, 0:1] = x0
        z[:, 0:1] = z0
        for i, dt in tqdm.tqdm(
            enumerate(np.diff(t), start=1), total=len(np.diff(t)), initial=1
        ):
            z_ = z0
            x_ = x0 + dt * self.dxdt(x0, z0, u[:, i - 1 : i])
            Y0 = np.vstack([x_, z_])
            sol = root(
                lambda Y: residuals(
                    Y,
                    x0,
                    z0,
                    u[:, i - 1 : i],
                    u[:, i : i + 1],
                    dt,
                ),
                x0=Y0.ravel(),
                method="hybr",
            )
            if not sol.success:
                print("Root solver failed, trying least_squares...")
                sol = least_squares(
                    lambda Y: residuals(
                        Y,
                        x0,
                        z0,
                        u[:, i - 1 : i],
                        u[:, i : i + 1],
                        dt,
                    ),
                    x0=Y0.ravel(),
                )
                if not sol.success:
                    raise RuntimeError("Failed to solve implicit step")

            Y = sol.x
            x0 = Y[:dim_x][:, np.newaxis]
            z0 = Y[dim_x:][:, np.newaxis]
            x[:, i : i + 1] = x0
            z[:, i : i + 1] = z0
        return Solution(x=x, z=z)

    def integrate_dynamic(
        self, t: np.ndarray, x0: np.ndarray, z0: np.ndarray, u: np.ndarray
    ) -> Solution:
        """
        Integrate the DAE system using a dynamic method over the given time array.
        Returns the state and algebraic variable trajectories.
        Assumes the control input u is constant (dim_u, 1).
        """

        k = t.shape[0]
        dim_x = x0.shape[0]
        dim_z = z0.shape[0]

        z0 = self.calc_ic(x0, z0, u[:, 0:1])

        x = np.zeros((dim_x, k))
        z = np.zeros((dim_z, k))
        x[:, 0:1] = x0
        z[:, 0:1] = z0

        def residuals(Y, x_old, z_old, u_old, u_new, dt):
            x_new = Y[:dim_x][:, np.newaxis]
            z_new = Y[dim_x : dim_x + dim_z][:, np.newaxis]

            f_new = self.f(x_new, z_new, u_new)
            f_old = self.f(x_old, z_old, u_old)
            g_new = self.g(x_new, z_new, u_new)
            res = np.vstack([(x_new - x_old) - dt * 0.5 * (f_new + f_old), g_new])
            return res.ravel()

        for i, dt in tqdm.tqdm(
            enumerate(np.diff(t), start=1), total=len(np.diff(t)), initial=1
        ):
            x_ = x0 + dt * self.dxdt(x0, z0, u[:, i - 1 : i])
            z_ = self.calc_ic(x_, z0, u[:, i : i + 1])
            if np.linalg.norm(self.g(x_, z_, u[:, i : i + 1])) > 1e-6:
                print(
                    f"Warning: Large residual detected, {np.linalg.norm(self.g(x_, z_, u[:, i : i + 1]))}, attempting implicit solve..."
                )
                Y0 = np.vstack([x_, z_])
                sol = root(
                    lambda Y: residuals(
                        Y,
                        x0,
                        z0,
                        u[:, i - 1 : i],
                        u[:, i : i + 1],
                        dt,
                    ),
                    x0=Y0.ravel(),
                    method="hybr",
                )
                if not sol.success:
                    print("Root solver failed, trying least_squares...")
                    sol = least_squares(
                        lambda Y: residuals(
                            Y,
                            x0,
                            z0,
                            u[:, i - 1 : i],
                            u[:, i : i + 1],
                            dt,
                        ),
                        x0=Y0.ravel(),
                    )
                    if not sol.success:
                        raise RuntimeError("Failed to solve implicit step")

                Y = sol.x
                x_ = Y[:dim_x][:, np.newaxis]
                z_ = Y[dim_x:][:, np.newaxis]

            x0 = x_
            z0 = z_
            x[:, i : i + 1] = x0
            z[:, i : i + 1] = z0
        return Solution(x=x, z=z)
