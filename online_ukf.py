import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.interpolate
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


if __name__ == "__main__":
    path = pathlib.Path(
        "C:\\Users\\KristianFredrikKlepp\\OneDrive - Corvus Energy\\Documents\\data\\digibatt\\WP4\\cycles"
    ).resolve()

    file = random.choice(list(path.glob("*.parquet")))
    data = pl.read_parquet(file).filter(pl.col("Cycle").is_between(3, 4))

    t, I, V = (
        data.select(["Program Duration#ms", "Current#A", "Voltage#V"]).to_numpy().T
    )
    t *= 1e-3
    t = t - t[0]
    _t = np.arange(0, t[-1], 1)
    _I = _make_interpolator(t, I)(_t)
    _V = _make_interpolator(t, V)(_t)
    print(f"Running on {file.name} with {len(_t)} samples")
    ecm = ECM(samples=1)
    joint = statespace.ModelFactory().transform(ecm, "joint")

    x = np.vstack(
        [
            np.full(ecm.samples, 0.5),
            np.full(ecm.samples, 0.0),
            np.full(ecm.samples, 0.0),
            np.full(ecm.samples, 0.9),
            np.full(ecm.samples, 1.1),
            np.full(ecm.samples, 0.05),
            np.full(ecm.samples, 0.95),
            np.full(ecm.samples, 0.95),
            np.full(ecm.samples, 0.05),
        ]
    )
    P = np.diag([1e-2, 1e-3, 1e-6] + [1e-6, 1e-6] + [1e-9, 1e-9, 1e-9, 1e-9])
    Q = np.diag([1e-8, 1e-3, 1e-6] + [1e-6, 1e-6] + [1e-9, 1e-9, 1e-9, 1e-9])
    R = np.diag([1e-8])

    alpha = 0.25
    beta = 0.25
    k = _t.size // 2
    maxiter = 100
    colors = plt.cm.viridis(np.linspace(0, 1, maxiter))
    fig, axs = plt.subplots(
        2,
        2 + Q.shape[0],
        constrained_layout=True,
        sharex=False,
        figsize=(3 * (2 + Q.shape[0]), 4),
    )
    axs[0, 0].plot(_t, _V, color="k", label="Measured")
    axs[0, 1].axvline(_t[-k], color="k", ls="--")

    for iov, cov in enumerate([*np.diag(R), *np.diag(Q)]):
        axs[1, iov].semilogy(0, cov, color=colors[0], marker=".", ls="none")

    for itr in range(maxiter):
        ukf = statespace.FilterFactory().from_model(
            "UHINF",
            joint,
            process_noise=Q,
            measurement_noise=R,
            initial_covariance=P,
            initial_state=x,
        )

        est = ukf.run(_t, _V[None, :], u=_I[None, :], progress=True)
        residual = est["residual"][:, -k:]
        innovation = est["innovation"][:, -k:]
        gain = est["gain"][:, -k:]
        Q = (1 - alpha) * Q + alpha * (gain @ innovation.T @ innovation @ gain.T) / k  # type: ignore
        R = (1 - beta) * R + beta * (residual @ residual.T) / k  # type: ignore

        L = np.mean(est["log_likelihood"])
        axs[1, -1].plot(itr, -L, color=colors[itr], ls="none", marker="o")
        axs[0, 0].plot(_t, est["measurement"][0, :], color=colors[itr])
        axs[1, 0].semilogy(itr, R[0, 0], color=colors[itr], marker=".", ls="none")
        for i in range(Q.shape[0]):
            axs[0, i + 1].plot(_t, est["state"][i, :], color=colors[itr])
            axs[1, i + 1].semilogy(
                itr, Q[i, i], color=colors[itr], marker=".", ls="none"
            )
        plt.pause(0.1)

    plt.show()
