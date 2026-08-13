# Upscale things
import pathlib
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.optimize
from scipy.interpolate import interp1d
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve


def _make_interpolator(x, y):
    return interp1d(
        x,
        y,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )


@dataclass
class ECMSolution:
    pass


class ECM:
    def __init__(
        self,
        parameters: dict | None = None,
        samples: int = 1,
    ):
        self._data = pl.read_csv(
            pathlib.Path(__file__).resolve().joinpath("data/gitt_ecm_parameters.csv")
        )
        if parameters is None:
            parameters = {}
        parameters.setdefault("Initial State of Charge / 1", 0.5)
        parameters.setdefault("Nominal Capacity / Ah", 5.0)
        parameters.setdefault("Mean State of Health / 1", 1.0)
        parameters.setdefault("Stdev State of Health / 1", 0.05)
        parameters.setdefault("Mean State of Resistance / 1", 1.0)
        parameters.setdefault("Stdev State of Resistance / 1", 0.05)
        parameters.setdefault("Timeconstant hysteresis / 1", np.full(samples, 0.1))

        parameters.setdefault(
            "State of Health / 1",
            np.random.normal(
                parameters["Mean State of Health / 1"],
                parameters["Stdev State of Health / 1"],
                samples,
            ),
        )
        parameters.setdefault(
            "State of Resistance / 1",
            np.random.normal(
                parameters["Mean State of Resistance / 1"],
                parameters["Stdev State of Resistance / 1"],
                samples,
            ),
        )

        for df in self._data.partition_by("Direction / 1"):
            for col in [
                "Open-circuit voltage / V",
                "R0 / Ohm",
                "R1 / Ohm",
                "Tau1 / s",
            ]:
                d = df.select("Direction / 1").to_series().unique().item()
                x = df.select("State of Charge / 1").to_series().to_numpy()
                y = df.select(col).to_series().to_numpy()
                parameters.setdefault(f"{d} {col}", _make_interpolator(x, y))
        self._parameters = parameters
        self._samples = samples
        return

    @property
    def data(self):
        return self._data

    @property
    def parameters(self):
        return self._parameters

    @property
    def samples(self):
        return self._samples

    def _hmix(self, hyst: np.ndarray, chg: np.ndarray, dhg: np.ndarray):
        alpha = (1 - hyst) / 2
        beta = (hyst - 1) / 2
        return alpha * chg + beta * dhg

    def ode(self, x: np.ndarray, u: np.ndarray):
        """
        Cell ECM ODE
        x: States SOC, POL, HYST
        u: Current
        """
        if u.ndim == 1:
            u = u[np.newaxis, :]
        if x.ndim == 1:
            x = x[:, np.newaxis]
        soc = x[0, :]
        pol = x[1, :]
        hyst = x[2, :]

        d_soc = (
            u
            / self._parameters["Nominal Capacity / Ah"]
            / self.parameters["State of Health / 1"]
            / 3600
        )
        R1_c = self._parameters["Charge R1 / Ohm"](soc)
        R1_d = self._parameters["Discharge R1 / Ohm"](soc)
        T1_c = self._parameters["Charge Tau1 / s"](soc)
        T1_d = self._parameters["Discharge Tau1 / s"](soc)
        R1 = self.parameters["State of Resistance / 1"] * self._hmix(hyst, R1_c, R1_d)
        T1 = self._hmix(hyst, T1_c, T1_d)
        d_pol = (u * R1 - pol) / T1
        d_hyst = (
            (np.sign(u) - hyst)
            / self._parameters["Timeconstant hysteresis / 1"]
            * d_soc
        )
        return np.vstack((d_soc, d_pol, d_hyst))

    def voltage(self, x: np.ndarray, u: np.ndarray):
        """
        Cell ECM voltage
        x: States SOC, POL, HYST
        u: Current
        """
        if u.ndim == 1:
            u = u[np.newaxis, :]
        if x.ndim == 1:
            x = x[:, np.newaxis]
        soc = x[0, :]
        pol = x[1, :]
        hyst = x[2, :]

        R0_c = self._parameters["Charge R0 / Ohm"](soc)
        R0_d = self._parameters["Discharge R0 / Ohm"](soc)
        R0 = self.parameters["State of Resistance / 1"] * self._hmix(hyst, R0_c, R0_d)
        OCV_c = self._parameters["Charge Open-circuit voltage / V"](soc)
        OCV_d = self._parameters["Discharge Open-circuit voltage / V"](soc)
        OCV = self._hmix(hyst, OCV_c, OCV_d)
        return OCV + pol + u * R0

    def step(self, dt: float, x: np.ndarray, u: np.ndarray):
        """
        Cell ECM step
        d_t: Time step
        x: States SOC, POL, HYST
        u: Current
        """
        return x + dt * self.ode(x, u)


@dataclass
class ModuleSolution:
    x: np.ndarray
    z: np.ndarray

    @property
    def cell_soc(self):
        return self.x[0, :, :]

    @property
    def cell_polarization(self):
        return self.x[1, :, :]

    @property
    def cell_hysteresis(self):
        return self.x[2, :, :]


class Module:
    def __init__(
        self,
        module_parameters: dict | None = None,
        cell_parameters: dict | None = None,
        cells_in_series: int = 1,
        cells_in_parallel: int = 1,
    ):
        self._cells_in_series = cells_in_series
        self._cells_in_parallel = cells_in_parallel

        self._ecm = ECM(
            parameters=cell_parameters, samples=cells_in_series * cells_in_parallel
        )
        if module_parameters is None:
            module_parameters = {}
        module_parameters.setdefault(
            "Nominal Capacity / Ah",
            self._ecm.parameters["Nominal Capacity / Ah"] * cells_in_parallel,
        )
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

    def _split_z(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """
        Split the module states into cell states
        z: States of the module
        """
        n_pot = 2 + self.cells_in_series * (self.cells_in_parallel + 1)
        n_cur_ser = (self.cells_in_series + 2) * self.cells_in_parallel
        pot = z[:n_pot]
        cur = z[n_pot::]
        cur_ser = cur[:n_cur_ser].reshape(
            self.cells_in_series + 2, self.cells_in_parallel
        )
        cur_par = cur[n_cur_ser::].reshape(
            self.cells_in_series, self.cells_in_parallel - 1
        )
        return {
            "Negative terminal potential / V": pot[0],
            "Positive terminal potential / V": pot[-1],
            "Inner node potential / V": pot[1:-1].reshape(
                self.cells_in_series, self.cells_in_parallel + 1
            ),
            "Series current / A": cur_ser,
            "Parallel current / A": cur_par,
            "Cell current / A": cur_ser[1:-1, :],
        }

    def _cell_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
        """
        Cell voltage of the module
        x: States SOC, POL, HYST
        z: States of the module
        u: Current
        """
        current = self._split_z(z)["Cell current / A"]
        return self.ecm.voltage(x, current)

    def _sensor_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
        U_inn = self._split_z(z)["Inner node potential / V"]
        return U_inn[1:, :] - U_inn[:-1, :]

    def _module_voltage(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
        U_neg = self._split_z(z)["Negative terminal potential / V"]
        U_pos = self._split_z(z)["Positive terminal potential / V"]
        return U_pos - U_neg

    def _alg(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
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
        U_inn = z_split["Inner node potential / V"]
        V_cell = self._cell_voltage(x, z, u)
        J_ser = z_split["Series current / A"]
        J_par = z_split["Parallel current / A"]

        eqs = []
        eqs.append(U_neg - U_inn[0, :] - J_ser[0, :] * R_neg)

        eqs.append((U_inn[1:, :] - U_inn[0:-1, :]) - (J_ser[1:-1, :] * R_ser + V_cell))
        eqs.append(J_ser[-1, :].sum() - u)

        eq = np.zeros_like(U_inn)
        eq[:, 0:-1] += J_par
        eq[:, 1::] -= J_par
        eqs.append(eq)

        return np.hstack([eq.ravel() for eq in eqs])

    def _ode(self, x: np.ndarray, z: np.ndarray, u: np.ndarray):
        current = self._split_z(z)["Cell current / A"]
        return self.ecm.ode(x, current)

    def _solve(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        jac: np.ndarray | csr_array | None = None,
    ):
        if jac is None:
            jac = np.asarray(
                scipy.optimize.approx_fprime(
                    lambda z_, x_=x, u_=u: self._alg(x_, z_, u_), z
                )
            )
        if isinstance(jac, np.ndarray):
            jac = csr_array(jac)
        return spsolve(jac, -self._alg(x, z, u))  # type: ignore

    def step(
        self,
        dt: float,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        J: np.ndarray | csr_array | None = None,
    ):
        """
        Module ECM step
        d_t: Time step
        x: States SOC, POL, HYST
        z: States of the module
        u: Current
        """
        x = x + dt * self._ode(x, z, u)
        z = self._solve(x, z, u, jac=J)  # type: ignore[unknown-argument]
        return x, z

    def simulate(
        self,
        t: np.ndarray,
        u: np.ndarray,
        x0: np.ndarray | None = None,
        z0: np.ndarray | None = None,
    ):
        if x0 is None:
            x0 = np.full((3, self.cells_in_series * self.cells_in_parallel), 0.5)
        if z0 is None:
            z0 = np.zeros(
                (
                    2
                    + self.cells_in_series * (self.cells_in_parallel + 1)
                    + (self.cells_in_series + 2) * self.cells_in_parallel
                    + self.cells_in_series * (self.cells_in_parallel - 1),
                    u.shape[1],
                )
            )
        z0 = self._solve(x0, z0, u[:, 0])  # type: ignore

        x = [x0]
        z = [z0]
        for k, dt in enumerate(np.diff(t), start=1):
            x0, z0 = self.step(dt, x[-1], z[-1], u[:, k])  # type: ignore
            x.append(x0)
            z.append(z0)
        return np.array(x), np.array(z)
