import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def decay(t, OCV: float, U0: float, log_tau: float):
    tau = 10**log_tau
    return OCV + U0 * np.exp(-t / tau)


def fit_decay(t, V, sigma=1e-3):

    if V[-1] > V[0]:
        lo, hi = [V[-1], -1, np.log10(2)], [4.2, 0, np.log10(36000)]
    elif V[-1] < V[0]:
        lo, hi = [2.5, 0, np.log10(2)], [V[-1], 1, np.log10(36000)]
    else:
        lo, hi = [2.5, -1, np.log10(2)], [4.2, 1, np.log10(36000)]
    p0 = [V[-1], 0.0, np.log10(100)]

    popt, pcov = curve_fit(
        decay, t, V, p0=p0, sigma=sigma, absolute_sigma=True, bounds=(lo, hi)
    )
    perr = np.sqrt(np.diag(pcov))
    results = {
        "Open-circuit voltage / V": popt[0],
        "Element-1 initial overpotential / V": popt[1],
        "Log Tau1 / 1": popt[2],
        "Tau1 / s": 10 ** popt[2],
        "Open-circuit voltage stdev / V": perr[0],
        "Element-1 initial overpotential stdev / V": perr[1],
        "Log Tau1 stdev / 1": perr[2],
        "Tau1 stdev / s": (10 ** popt[2]) * np.log(10) * perr[2],
    }
    return results


def fit_polarization(t, U, I, sigma=1e-3):
    def simulate(t: np.ndarray, U0: float, R0: float, R1: float, log_Tau1: float):
        pol = np.zeros_like(t)
        pol[0] = U0
        Tau1 = 10**log_Tau1
        for k, dt in enumerate(np.diff(t), start=1):
            pol[k] = pol[k - 1] + dt * (R1 * I[k - 1] - pol[k - 1]) / Tau1
        return pol + R0 * I

    popt, pcov = curve_fit(
        simulate,
        t,
        U,
        p0=[0.0, 0.01, 0.01, np.log10(100)],
        sigma=sigma,
        absolute_sigma=True,
        bounds=([-1, 1e-5, 1e-5, np.log10(2)], [1, 10, 10, np.log10(36000)]),
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcov))
    results = {
        "Element-1 initial overpotential / V": popt[0],
        "R0 / Ohm": popt[1],
        "R1 / Ohm": popt[2],
        "Log Tau1 / 1": popt[3],
        "Tau1 / s": 10 ** popt[3],
        "Element-1 initial overpotential stdev / V": perr[0],
        "R0 stdev / Ohm": perr[1],
        "R1 stdev / Ohm": perr[2],
        "Log Tau1 stdev / 1": perr[3],
        "Tau1 stdev / s": (10 ** popt[3]) * np.log(10) * perr[3],
    }
    return results


def determine_direction(mode: pl.Series) -> pl.Series:
    """
    Determines the charge/discharge current direction
    """
    mode_list = mode.to_list()
    dir_list = []

    for k, m in enumerate(mode_list):
        if k == 0:
            dir_list.append(None if m == "Rest" else m)
        else:
            dir_list.append(dir_list[-1] if m == "Rest" else m)
    return pl.Series(name="Direction / 1", values=dir_list)


def interpolate(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)(  # type: ignore
        z
    )
