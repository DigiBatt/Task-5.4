# Upscale things
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ecm import ECM
from solver import DAESolver


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
        cells_in_series: int = 3,
        cells_in_parallel: int = 4,
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
    def ecm(self) -> ECM:
        return self._ecm

    @property
    def z0(self) -> np.ndarray:
        return np.zeros(
            (
                2
                + (self.cells_in_series + 1) * self.cells_in_parallel
                + (self.cells_in_series + 2) * self.cells_in_parallel
                + (self.cells_in_series + 1) * (self.cells_in_parallel - 1),
                1,
            )
        )

    def _split_z(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """
        Split the module states into cell states
        z: States of the module
        """
        # Split the algebraic states into potentials and currents
        n_pot = 2 + (self.cells_in_series + 1) * (self.cells_in_parallel)
        pot = z[:n_pot, :]
        cur = z[n_pot:, :]

        # Split the currents into series and parallel currents
        n_ser = (self.cells_in_series + 2) * self.cells_in_parallel

        cur_ser = cur[:n_ser, :]
        cur_par = cur[n_ser:, :]

        cur_cell = cur_ser[self.cells_in_parallel : -self.cells_in_parallel]

        # Some states are reshaped to match the module topology
        return {
            "Negative terminal potential / V": pot[0:1, :],
            "Positive terminal potential / V": pot[-1:, :],
            "Inner node potential / V": pot[1:-1, :],
            "Series current / A": cur_ser,
            "Parallel current / A": cur_par,
            "Cell current / A": cur_cell,
        }

    def cell_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Cell voltage of the module
        x: States SOC, POL, HYST
        z: States of the module
        u: Current
        """
        current = self._split_z(z)["Cell current / A"]
        return self.ecm.voltage(x, current)

    def sensor_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        U_inn = self._split_z(z)["Inner node potential / V"].reshape(
            (self.cells_in_series + 1, self.cells_in_parallel, -1)
        )
        return U_inn[1:] - U_inn[:-1]

    def module_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        U_neg = self._split_z(z)["Negative terminal potential / V"]
        U_pos = self._split_z(z)["Positive terminal potential / V"]
        return U_pos - U_neg

    def alg(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Algebraic equations for the though-up module
        """
        R_neg = self.module_parameters["Negative terminal resistance / Ohm"][
            ..., np.newaxis
        ]
        R_pos = self.module_parameters["Positive terminal resistance / Ohm"][
            ..., np.newaxis
        ]
        R_par = self.module_parameters["Parallel resistance / Ohm"][..., np.newaxis]
        R_ser = self.module_parameters["Series resistance / Ohm"][..., np.newaxis]

        z_split = self._split_z(z)

        U_neg = z_split["Negative terminal potential / V"]
        U_pos = z_split["Positive terminal potential / V"]
        U_inn = z_split["Inner node potential / V"].reshape(
            (self.cells_in_series + 1, self.cells_in_parallel, -1)
        )

        V_cell = self.cell_voltage(x, z, u).reshape(
            (self.cells_in_series, self.cells_in_parallel, -1)
        )

        J_ser = z_split["Series current / A"].reshape(
            (self.cells_in_series + 2, self.cells_in_parallel, -1)
        )
        J_par = z_split["Parallel current / A"].reshape(
            (self.cells_in_series + 1, self.cells_in_parallel - 1, -1)
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

        return np.vstack([eq.ravel()[:, np.newaxis] for eq in eqs])

    def ode(self, x: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        current = self._split_z(z)["Cell current / A"]
        return self.ecm.ode(x, current)

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

        if z0.ndim == 1:
            z0 = z0[:, np.newaxis]
        if u.ndim == 1:
            u = u[np.newaxis, :]
        if u.shape[1] == 1:
            u = np.repeat(u, t.size, axis=1)

        solver = DAESolver(f=self.ode, g=self.alg, linear=False, sparse=False)
        solution = solver.integrate_implicit(t, x0, z0, u)
        x, z = solution.x, solution.z

        soc, pol, hyst = self.ecm._split_x(x)
        solution = ModuleSolution(
            cell_soc=soc.reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            cell_polarization=pol.reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            cell_hysteresis=hyst.reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            cell_voltage=self.cell_voltage(x, z, u).reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            sensor_voltage=self.sensor_voltage(x, z, u).reshape(
                (self.cells_in_series, self.cells_in_parallel, t.size)
            ),
            module_voltage=self.module_voltage(x, z, u),
            module_current=u,
            cell_current=self._split_z(z)["Cell current / A"].reshape(
                self.cells_in_series, self.cells_in_parallel, -1
            ),
        )
        return solution

        # x = [x0]
        # z = [z0]
        # for k, dt in tqdm.tqdm(enumerate(np.diff(t), start=1), initial=1, total=t.size):
        #     x0, z0 = self.step(dt, x0, z0, u[:, k - 1 : k], u[:, k : k + 1])  # type: ignore
        #     x.append(x0)
        #     z.append(z0)

        # vol_cell = [
        #     self._cell_voltage(x[k], z[k], u[:, k : k + 1]).reshape(
        #         (self.cells_in_series, self.cells_in_parallel)
        #     )
        #     for k in range(t.size)
        # ]
        # vol_sensor = [
        #     self._sensor_voltage(x[k], z[k], u[:, k : k + 1]) for k in range(t.size)
        # ]
        # vol_module = [
        #     self._module_voltage(x[k], z[k], u[:, k : k + 1]) for k in range(t.size)
        # ]
        # soc, pol, hyst = self.ecm._split_x(np.hstack(x))

        # cur_cell = [
        #     self._split_z(z[k])["Cell current / A"].reshape(
        #         self.cells_in_series, self.cells_in_parallel
        #     )
        #     for k in range(t.size)
        # ]
        # return ModuleSolution(
        #     cell_soc=soc.reshape(
        #         (self.cells_in_series, self.cells_in_parallel, t.size)
        #     ),
        #     cell_polarization=pol.reshape(
        #         (self.cells_in_series, self.cells_in_parallel, t.size)
        #     ),
        #     cell_hysteresis=hyst.reshape(
        #         (self.cells_in_series, self.cells_in_parallel, t.size)
        #     ),
        #     cell_voltage=np.stack(vol_cell, axis=-1),
        #     sensor_voltage=np.stack(vol_sensor, axis=-1),
        #     module_voltage=np.stack(vol_module, axis=-1),
        #     cell_current=np.stack(cur_cell, axis=-1),
        #     module_current=u,
        # )


if __name__ == "__main__":
    model = Module(
        cells_in_series=4,
        cells_in_parallel=3,
        cell_parameters={"Initial State of Charge / 1": 0.1},
    )

    u = np.zeros(100)
    u = np.concatenate([u, np.ones(1000), np.zeros(100), -np.ones(1000)], axis=0)

    t = np.linspace(0, 1, u.size) * 3600
    u = u.reshape(1, -1) * model.module_parameters["Nominal Capacity / Ah"]
    solution = model.simulate(t, u)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 6, figsize=(15, 3), constrained_layout=True)
    for i, attr in enumerate(
        [
            "cell_voltage",
            "cell_soc",
            "cell_polarization",
            "cell_current",
            "module_voltage",
            "module_current",
        ]
    ):
        x = getattr(solution, attr)
        print(f"{attr}: shape = {x.shape}")
        axs[i].plot(t, x.reshape(-1, t.size).T)
        axs[i].set_title(attr.replace("_", " ").title())
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()
