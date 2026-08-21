# Upscale things
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.optimize
import tqdm.auto as tqdm
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve

from ecm_model import ECM


@dataclass
class ModuleSolution:
    cell_soc: np.ndarray
    cell_polarization: np.ndarray
    cell_hysteresis: np.ndarray
    cell_voltage: np.ndarray
    cell_current: np.ndarray
    sensor_voltage: np.ndarray
    module_voltage: np.ndarray
    module_current: np.ndarray


class Module:
    def __init__(
        self,
        module_parameters: dict | None = None,
        cell_parameters: dict | None = None,
        cells_in_series: int = 1,
        cells_in_parallel: int = 1,
        parameters_source: Literal["gitt", "ici"] = "gitt",
    ):
        self._cells_in_series = cells_in_series
        self._cells_in_parallel = cells_in_parallel

        self._ecm = ECM(
            parameters=cell_parameters,
            samples=cells_in_series * cells_in_parallel,
            parameters_source=parameters_source,
        )
        if module_parameters is None:
            module_parameters = {}
        module_parameters.setdefault(
            "Nominal Capacity / Ah",
            self._ecm.parameters["Nominal Capacity / Ah"] * cells_in_parallel,
        )
        module_parameters.setdefault("Negative terminal resistance / Ohm", 1e-3)
        module_parameters.setdefault("Positive terminal resistance / Ohm", 1e-3)
        module_parameters.setdefault("Parallel resistance / Ohm", 1e-3)
        module_parameters.setdefault("Series resistance / Ohm", 1e-3)
        self._module_parameters = module_parameters
        return

    @property
    def module_parameters(self):
        return self._module_parameters

    @property
    def cell_parameters(self):
        return self._ecm.parameters

    @property
    def cells_in_series(self):
        return self._cells_in_series

    @property
    def cells_in_parallel(self):
        return self._cells_in_parallel

    @property
    def ecm(self):
        return self._ecm

    @property
    def z0(self):
        return np.zeros(
            (
                2
                + (self.cells_in_series + 1) * self.cells_in_parallel
                + (self.cells_in_series + 2) * self.cells_in_parallel
                + (self.cells_in_series + 1) * (self.cells_in_parallel - 1),
            )
        )

    def _split_z(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """
        Split the module states into cell states
        z: States of the module
        """
        # Split the algebraic states into potentials and currents
        n_pot = 2 + (self.cells_in_series + 1) * (self.cells_in_parallel)
        pot = z[:n_pot]
        cur = z[n_pot:]

        # Split the currents into series and parallel currents
        n_ser = (self.cells_in_series + 2) * self.cells_in_parallel

        cur_ser = cur[:n_ser]
        cur_par = cur[n_ser:]
        cur_cell = cur_ser[self.cells_in_parallel : -self.cells_in_parallel]

        # Some states are reshaped to match the module topology
        return {
            "Negative terminal potential / V": pot[0],
            "Positive terminal potential / V": pot[-1],
            "Inner node potential / V": pot[1:-1],
            "Series current / A": cur_ser,
            "Parallel current / A": cur_par,
            "Cell current / A": cur_cell,
        }

    def _cell_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
        """
        Cell voltage of the module
        x: States SOC, POL, HYST
        z: States of the module
        u: Current
        """
        current = self._split_z(z)["Cell current / A"][:, np.newaxis]
        return self.ecm.voltage(x, current)

    def _sensor_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
        U_inn = self._split_z(z)["Inner node potential / V"].reshape(
            (self.cells_in_series + 1, self.cells_in_parallel)
        )
        return U_inn[1:, :] - U_inn[:-1, :]

    def _module_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
        U_neg = self._split_z(z)["Negative terminal potential / V"]
        U_pos = self._split_z(z)["Positive terminal potential / V"]
        return U_pos - U_neg

    def _alg(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Algebraic equations for the though-up module
        """

        R_neg = self.module_parameters["Negative terminal resistance / Ohm"]
        R_pos = self.module_parameters["Positive terminal resistance / Ohm"]
        R_par = self.module_parameters["Parallel resistance / Ohm"]
        R_ser = self.module_parameters["Series resistance / Ohm"]

        z_split = self._split_z(z)

        U_neg = z_split["Negative terminal potential / V"]
        U_pos = z_split["Positive terminal potential / V"]
        U_inn = z_split["Inner node potential / V"].reshape(
            (self.cells_in_series + 1, self.cells_in_parallel)
        )

        V_cell = self._cell_voltage(x, z, u).reshape(
            (self.cells_in_series, self.cells_in_parallel)
        )

        J_ser = z_split["Series current / A"].reshape(
            (self.cells_in_series + 2, self.cells_in_parallel)
        )
        J_par = z_split["Parallel current / A"].reshape(
            (self.cells_in_series + 1, self.cells_in_parallel - 1)
        )

        eqs = []
        eqs.append(U_neg)

        eqs.append((U_inn[0, :] - U_neg) - (J_ser[0, :] * R_neg))

        eqs.append((U_inn[1::, :] - U_inn[0:-1, :]) - (J_ser[1:-1, :] * R_ser + V_cell))

        eqs.append((U_pos - U_inn[-1, :]) - (J_ser[-1, :] * R_pos))

        eqs.append(J_ser[-1, :].sum() - u)

        eqs.append((U_inn[:, 1::] - U_inn[:, 0:-1]) - (J_par * R_par))

        eq = J_ser[0:-1, :] - J_ser[1::, :]
        eq[:, 0:-1] -= J_par
        eq[:, 1::] += J_par

        eqs.append(eq)

        return np.hstack([eq.ravel() for eq in eqs])

    def _ode(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        current = self._split_z(z)["Cell current / A"]
        return self.ecm.ode(x, current[:, np.newaxis])

    def _jac(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> csr_array:
        return csr_array(
            scipy.optimize.approx_fprime(
                z, lambda z_, x_=x, u_=u: self._alg(x_, z_, u_)
            )
        )

    def _solve(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        jac: np.ndarray | csr_array | None = None,
    ) -> np.ndarray:

        if jac is None:
            jac = self._jac(x, z, u).toarray()
        if isinstance(jac, np.ndarray):
            jac = csr_array(jac)
        return spsolve(jac, -self._alg(x, z, u)) + z

    def step(
        self,
        dt: float,
        x: np.ndarray,
        z: np.ndarray,
        u_prev: np.ndarray,
        u_next: np.ndarray,
        jac: np.ndarray | csr_array | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Module ECM step
        d_t: Time step
        x: States SOC, POL, HYST
        z: States of the module
        u: Current
        """

        x = x + dt * self._ode(x, z, u_prev)
        z = self._solve(x, z, u_next, jac=jac)
        return x, z

    def simulate(
        self,
        t: np.ndarray,
        u: np.ndarray,
        x0: np.ndarray | None = None,
        z0: np.ndarray | None = None,
    ):

        if x0 is None:
            x0 = self.ecm.x0
        if z0 is None:
            z0 = self.z0
        if x0.ndim == 1:
            x0 = x0[:, np.newaxis]

        if u.ndim == 1:
            u = u[np.newaxis, :]

        jac = self._jac(x0, z0, u[:, 0:1])
        z0 = self._solve(x0, z0, u[:, 0:1], jac=jac)  # type: ignore

        x = [x0]
        z = [z0]
        for k, dt in tqdm.tqdm(enumerate(np.diff(t), start=1), initial=1, total=t.size):
            x0, z0 = self.step(dt, x0, z0, u[:, k - 1 : k], u[:, k : k + 1], jac=jac)  # type: ignore
            x.append(x0)
            z.append(z0)

        vol_cell = [
            self._cell_voltage(x[k], z[k], u[:, k : k + 1]).reshape(
                (self.cells_in_series, self.cells_in_parallel)
            )
            for k in range(t.size)
        ]
        vol_sensor = [
            self._sensor_voltage(x[k], z[k], u[:, k : k + 1]) for k in range(t.size)
        ]
        vol_module = [
            self._module_voltage(x[k], z[k], u[:, k : k + 1]) for k in range(t.size)
        ]
        soc, pol, hyst = self._ecm._split_x(np.hstack(x))

        cur_cell = [
            self._split_z(z[k])["Cell current / A"].reshape(
                self.cells_in_series, self.cells_in_parallel
            )
            for k in range(t.size)
        ]
        return ModuleSolution(
            cell_soc=soc.reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            cell_polarization=pol.reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            cell_hysteresis=hyst.reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            cell_voltage=np.stack(vol_cell, axis=-1),
            sensor_voltage=np.stack(vol_sensor, axis=-1),
            module_voltage=np.stack(vol_module, axis=-1),
            cell_current=np.stack(cur_cell, axis=-1),
            module_current=u,
        )
