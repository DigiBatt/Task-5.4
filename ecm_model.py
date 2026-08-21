# Upscale things
import pathlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import tqdm.auto as tqdm
from scipy.interpolate import interp1d


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
    soc: np.ndarray
    polarization: np.ndarray
    hysteresis: np.ndarray
    voltage: np.ndarray


class ECM:
    def __init__(
        self,
        parameters: dict | None = None,
        samples: int = 1,
        parameters_source: Literal["gitt", "ici"] = "gitt",
    ):
        self._data = pl.read_csv(
            pathlib.Path(__file__)
            .resolve()
            .parent.joinpath(f"data/{parameters_source}_results.csv")
        )
        if parameters is None:
            parameters = {}

        parameters.setdefault("Initial State of Charge / 1", 0.5)
        parameters.setdefault("Initial Hysteresis / 1", 0.0)

        parameters.setdefault("Nominal Capacity / Ah", 5.0)

        parameters.setdefault("Mean State of Health / 1", 1.0)
        parameters.setdefault("Stdev State of Health / 1", 0.05)

        parameters.setdefault("Mean State of Resistance / 1", 1.0)
        parameters.setdefault("Stdev State of Resistance / 1", 0.05)

        parameters.setdefault("Timeconstant hysteresis / 1", 0.1)

        parameters.setdefault(
            "State of Health / 1",
            np.random.normal(
                parameters["Mean State of Health / 1"],
                parameters["Stdev State of Health / 1"],
                samples,
            )[:, np.newaxis],
        )
        parameters.setdefault(
            "State of Resistance / 1",
            np.random.normal(
                parameters["Mean State of Resistance / 1"],
                parameters["Stdev State of Resistance / 1"],
                samples,
            )[:, np.newaxis],
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

    @property
    def x0(self):
        soc = np.full((self.samples, 1), self.parameters["Initial State of Charge / 1"])
        pol = np.zeros((self.samples, 1))
        hyst = np.full((self.samples, 1), self.parameters["Initial Hysteresis / 1"])
        return np.vstack((soc, pol, hyst))

    def _split_x(self, x: np.ndarray) -> list[np.ndarray]:
        return np.vsplit(x, 3)

    def _hmix(self, hyst: np.ndarray, chg: np.ndarray, dhg: np.ndarray):
        alpha = (1 + hyst) / 2
        beta = (hyst - 1) / 2
        return alpha * chg - beta * dhg

    def _R0(self, x: np.ndarray):
        soc, _, hyst = self._split_x(x)
        R0_c = self._parameters["Charge R0 / Ohm"](soc)
        R0_d = self._parameters["Discharge R0 / Ohm"](soc)
        return self._hmix(hyst, R0_c, R0_d)

    def _R1(self, x: np.ndarray):
        soc, _, hyst = self._split_x(x)
        R1_c = self._parameters["Charge R1 / Ohm"](soc)
        R1_d = self._parameters["Discharge R1 / Ohm"](soc)
        return self._hmix(hyst, R1_c, R1_d)

    def _T1(self, x: np.ndarray):
        soc, _, hyst = self._split_x(x)
        T1_c = self._parameters["Charge Tau1 / s"](soc)
        T1_d = self._parameters["Discharge Tau1 / s"](soc)
        return self._hmix(hyst, T1_c, T1_d)

    def _OCV(self, x: np.ndarray):
        soc, _, hyst = self._split_x(x)
        OCV_c = self._parameters["Charge Open-circuit voltage / V"](soc)
        OCV_d = self._parameters["Discharge Open-circuit voltage / V"](soc)
        return self._hmix(hyst, OCV_c, OCV_d)

    def ode(self, x: np.ndarray, u: np.ndarray):
        """
        Cell ECM ODE
        x: States SOC, POL, HYST
        u: Current
        """
        soc, pol, hyst = self._split_x(x)
        current = u
        soh = self._parameters["State of Health / 1"]
        sor = self._parameters["State of Resistance / 1"]
        capacity = self._parameters["Nominal Capacity / Ah"]
        R1 = self._R1(x)
        T1 = self._T1(x)
        Th = self._parameters["Timeconstant hysteresis / 1"]

        d_soc = current / capacity / soh / 3600
        d_pol = (current * sor * R1 - pol) / T1

        d_hyst = np.sign(current) * (np.sign(current) - hyst) / Th * d_soc

        return np.vstack((d_soc, d_pol, d_hyst))

    def voltage(self, x: np.ndarray, u: np.ndarray):
        """
        Cell ECM voltage
        x: States SOC, POL, HYST
        u: Current
        """
        soc, pol, hyst = self._split_x(x)
        current = u
        sor = self._parameters["State of Resistance / 1"]
        R0 = self._R0(x)
        OCV = self._OCV(x)
        return OCV + sor * R0 * current + pol

    def step(self, dt: float, x: np.ndarray, u: np.ndarray):
        """
        Cell ECM step
        d_t: Time step
        x: States SOC, POL, HYST
        u: Current
        """
        return x + dt * self.ode(x, u)

    def simulate(self, t: np.ndarray, u: np.ndarray, x0: np.ndarray | None = None):
        if x0 is None:
            x0 = self.x0

        x = [x0]
        for k, dt in tqdm.tqdm(enumerate(np.diff(t), start=1), initial=1, total=t.size):
            x0 = self.step(dt, x0, u[:, k - 1 : k])
            x.append(x0)

        x = np.hstack(x)
        soc, pol, hyst = self._split_x(x)
        vol = self.voltage(x, u)

        return ECMSolution(
            soc=soc,
            polarization=pol,
            hysteresis=hyst,
            voltage=vol,
        )
