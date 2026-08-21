import pathlib
import random

import numpy as np
import polars as pl
import scipy.interpolate
import scipy.optimize
import statespace

from ocv_balancing import load_ne, load_pe, soc_to_ocv


def _make_interpolator(x, y):
    return scipy.interpolate.interp1d(
        x,
        y,
        fill_value="extrapolate",
        bounds_error=False,  # type: ignore
    )


class ECM(statespace.core.model.StateModel):
    def __init__(self, samples: int = 1):
        super().__init__(
            f=self.ecm_f,
            h=self.ecm_h,
            dim_x=3 * samples,
            dim_z=1 * samples,
            dim_u=1,
            dim_p=6 * samples,
        )
        self.samples = samples
        data = pl.read_csv("data/gitt_ecm_parameters.csv")
        self._Th = 0.1
        self._Cnom = 5.0
        luts = {}
        for d in ["Charge", "Discharge"]:
            for p in ["R0 / Ohm", "R1 / Ohm", "Tau1 / s"]:
                mask = pl.col("Direction / 1") == d
                x, y = data.filter(mask).select(["State of Charge / 1", p]).to_numpy().T
                luts[f"{d} {p}"] = _make_interpolator(x, y)  # type: ignore

        mask_lith = pl.col("Mode") == "Lithiation"
        mask_dlith = pl.col("Mode") == "Delithiation"
        luts["Lithiation NE"] = _make_interpolator(
            *load_ne()
            .filter(mask_lith)
            .select(["Degree of Lithiation / 1", "Voltage / V"])
            .to_numpy()
            .T,
        )
        luts["Delithiation NE"] = _make_interpolator(
            *load_ne()
            .filter(mask_dlith)
            .select(["Degree of Lithiation / 1", "Voltage / V"])
            .to_numpy()
            .T,
        )
        luts["Lithiation PE"] = _make_interpolator(
            *load_pe()
            .filter(mask_lith)
            .select(["Degree of Lithiation / 1", "Voltage / V"])
            .to_numpy()
            .T,
        )
        luts["Delithiation PE"] = _make_interpolator(
            *load_pe()
            .filter(mask_dlith)
            .select(["Degree of Lithiation / 1", "Voltage / V"])
            .to_numpy()
            .T,
        )
        self.luts = luts
        return

    def _hmix(self, h, c, d):
        a = (1 + h) / 2
        b = (1 - h) / 2
        return a * c + b * d

    def ecm_f(self, dt, x, u, p):
        soc, pol, hst = np.vsplit(x, 3)
        soh, sor, sto_0_ne, sto_1_ne, sto_0_pe, sto_1_pe = np.vsplit(p, 6)
        R1 = sor * self._hmix(
            hst,
            self.luts["Charge R1 / Ohm"](soc),
            self.luts["Discharge R1 / Ohm"](soc),
        )
        Tau1 = self._hmix(
            hst,
            self.luts["Charge Tau1 / s"](soc),
            self.luts["Discharge Tau1 / s"](soc),
        )

        d_soc = 0 * soc + u / (self._Cnom * 3600 * soh)
        d_pol = (u * R1 - pol) / Tau1
        d_hst = np.sign(u) * (np.sign(u) - hst) * d_soc / self._Th
        return x + dt * np.vstack([d_soc, d_pol, d_hst])

    def ecm_h(self, x, u, p):
        soc, pol, hst = np.vsplit(x, 3)
        soh, sor, sto_0_ne, sto_1_ne, sto_0_pe, sto_1_pe = np.vsplit(p, 6)

        R0 = sor * self._hmix(
            hst,
            self.luts["Charge R0 / Ohm"](soc),
            self.luts["Discharge R0 / Ohm"](soc),
        )

        ocv_chg = soc_to_ocv(
            soc,
            sto_0_ne,
            sto_1_ne,
            sto_0_pe,
            sto_1_pe,
            self.luts["Lithiation NE"],
            self.luts["Delithiation PE"],
        )
        ocv_dhg = soc_to_ocv(
            soc,
            sto_0_ne,
            sto_1_ne,
            sto_0_pe,
            sto_1_pe,
            self.luts["Delithiation NE"],
            self.luts["Lithiation PE"],
        )
        Voc = self._hmix(
            hst,
            ocv_chg,  # self.luts["Charge Open-circuit voltage / V"](soc),
            ocv_dhg,  # self.luts["Discharge Open-circuit voltage / V"](soc),
        )

        return Voc + pol + u * R0


def run_ukf(
    t: np.ndarray,
    I: np.ndarray,
    V: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    p: np.ndarray,
):
    ecm = ECM(samples=1)
    ukf = statespace.FilterFactory().from_model(
        "UHINF",
        ecm,
        process_noise=Q,
        measurement_noise=R,
        initial_covariance=P,
        initial_state=x0[:, None],
    )
    return ukf.run(t, V[None, :], u=I[None, :], progress=False, p=p[:, None])


def objective(
    params: np.ndarray,
    t: np.ndarray,
    I: np.ndarray,
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    axs,
    itr,
):

    itr.append(itr[-1] + 1)
    for p, ax in zip(params, axs):
        ax.plot(itr[-1], p, color="k", ls="none", marker="o")
    soh, sor, sto_0_ne, sto_1_ne, sto_0_pe, sto_1_pe = params
    Q = np.diag([1e-4, 1e-6, 1e-8])
    p = np.array([soh, sor, sto_0_ne, sto_1_ne, sto_0_pe, sto_1_pe])
    try:
        est = run_ukf(t, I, V, P, Q, R, x0, p)
        nll = -np.mean(est["log_likelihood"])
    except Exception:
        nll = np.inf
    print(f"Iteration {itr[-1]}: NLL = {nll:.3f}, params = {params}")
    axs[-1].plot(itr[-1], nll, color="k", ls="none", marker="o")
    plt.pause(1e-3)
    return nll


if __name__ == "__main__":
    path = pathlib.Path(
        "C:\\Users\\KristianFredrikKlepp\\OneDrive - Corvus Energy\\Documents\\data\\digibatt\\WP4\\cycles"
    ).resolve()

    file = random.choice(list(path.glob("*.parquet")))
    data = pl.read_parquet(file).filter(pl.col("Cycle").is_between(4, 4))

    t, I, V = (
        data.select(["Program Duration#ms", "Current#A", "Voltage#V"]).to_numpy().T
    )
    t *= 1e-3
    t = t - t[0]
    _t = np.arange(0, t[-1], 1)
    _I = _make_interpolator(t, I)(_t)
    _V = _make_interpolator(t, V)(_t)

    R = np.diag([1e-8])
    x0 = np.array([0.5, 0.0, 0.0])
    P = np.diag([1e-6, 1e-8, 1e-12])

    A = np.zeros((2, 9))
    A[0, 2] = -1
    A[0, 3] = 1
    A[1, 4] = 1
    A[1, 5] = -1

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        1, 9, figsize=(3 * 9, 4), constrained_layout=True, sharex=True
    )
    itr = [0]
    opt = scipy.optimize.differential_evolution(
        objective,
        args=(
            _t,
            _I,
            _V,
            P,
            R,
            x0,
            axs,
            itr,
        ),
        # constraints=scipy.optimize.LinearConstraint(A, lb=0.01, ub=1),
        # maxiter=10,
        bounds=[
            (0.5, 1.0),  # soh
            (0.8, 1.5),  # sor
            (0.0, 1.0),  # sto_0_ne
            (0.0, 1.0),  # sto_1_ne
            (0.0, 1.0),  # sto_0_pe
            (0.0, 1.0),  # sto_1_pe
            # (-10, -2),  # log_q1
            # (-10, -2),  # log_q2
            # (-10, -2),  # log_q3
        ],
        disp=True,
        x0=np.array([0.9, 1.0, 0.05, 0.95, 0.95, 0.05]),
    )
    print(opt)
