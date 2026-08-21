import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.interpolate
import scipy.optimize
import statespace


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
            dim_p=2 * samples,
        )
        self.samples = samples
        data = pl.read_csv("data/gitt_ecm_parameters.csv")
        self._Th = 0.1
        self._Cnom = 5.0
        luts = {}
        for d in ["Charge", "Discharge"]:
            for p in ["Open-circuit voltage / V", "R0 / Ohm", "R1 / Ohm", "Tau1 / s"]:
                mask = pl.col("Direction / 1") == d
                x, y = data.filter(mask).select(["State of Charge / 1", p]).to_numpy().T
                luts[f"{d} {p}"] = _make_interpolator(x, y)  # type: ignore

        # mask_lith = pl.col("Mode") == "Lithiation"
        # mask_dlith = pl.col("Mode") == "Delithiation"
        # luts["Lithiation NE"] = _make_interpolator(
        #     *load_ne()
        #     .filter(mask_lith)
        #     .select(["Degree of Lithiation / 1", "Voltage / V"])
        #     .to_numpy()
        #     .T,
        # )
        # luts["Delithiation NE"] = _make_interpolator(
        #     *load_ne()
        #     .filter(mask_dlith)
        #     .select(["Degree of Lithiation / 1", "Voltage / V"])
        #     .to_numpy()
        #     .T,
        # )
        # luts["Lithiation PE"] = _make_interpolator(
        #     *load_pe()
        #     .filter(mask_lith)
        #     .select(["Degree of Lithiation / 1", "Voltage / V"])
        #     .to_numpy()
        #     .T,
        # )
        # luts["Delithiation PE"] = _make_interpolator(
        #     *load_pe()
        #     .filter(mask_dlith)
        #     .select(["Degree of Lithiation / 1", "Voltage / V"])
        #     .to_numpy()
        #     .T,
        # )
        self.luts = luts
        return

    def _hmix(self, h, c, d):
        a = (1 + h) / 2
        b = (1 - h) / 2
        return a * c + b * d

    def ecm_f(self, dt, x, u, p):
        soc, pol, hst = np.vsplit(x, 3)
        soh, sor = np.vsplit(p, 2)
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
        soh, sor = np.vsplit(p, 2)

        R0 = sor * self._hmix(
            hst,
            self.luts["Charge R0 / Ohm"](soc),
            self.luts["Discharge R0 / Ohm"](soc),
        )

        # ocv_chg = soc_to_ocv(
        #     soc,
        #     sto_0_ne,
        #     sto_1_ne,
        #     sto_0_pe,
        #     sto_1_pe,
        #     self.luts["Lithiation NE"],
        #     self.luts["Delithiation PE"],
        # )
        # ocv_dhg = soc_to_ocv(
        #     soc,
        #     sto_0_ne,
        #     sto_1_ne,
        #     sto_0_pe,
        #     sto_1_pe,
        #     self.luts["Delithiation NE"],
        #     self.luts["Lithiation PE"],
        # )
        Voc = self._hmix(
            hst,
            self.luts["Charge Open-circuit voltage / V"](soc),
            self.luts["Discharge Open-circuit voltage / V"](soc),
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
        "UKF",
        ecm,
        process_noise=Q,
        measurement_noise=R,
        initial_covariance=P,
        initial_state=x0[:, None],
    )
    return ukf.run(t, V[None, :], u=I[None, :], progress=False, p=p[:, None])


class Objective:
    def __init__(
        self,
        t,
        I,
        V,
        P,
        R,
        x0,
        p,
        axs,
    ):
        self.t = t
        self.I = I
        self.V = V
        self.P = P
        self.R = R
        self.x0 = x0
        self.p = p
        self.axs = axs
        self.itr = [0]
        return

    def __hash__(self):
        return hash(
            (
                tuple(self.t),
                tuple(self.I),
                tuple(self.V),
                tuple(self.P.flatten()),
                tuple(self.R.flatten()),
                tuple(self.x0),
                tuple(self.p),
                tuple(self.axs),
            )
        )

    def __call__(self, params):
        params = np.array(params, dtype=float)
        # self.itr.append(self.itr[-1] + 1)
        # for q, ax in zip(params, axs):
        #    ax.plot(self.itr[-1], q, color="k", ls="none", marker="o")
        Q = np.diag(10**params)
        k = self.t.size // 3
        try:
            est = run_ukf(self.t, self.I, self.V, self.P, Q, self.R, self.x0, self.p)
            nll = -np.mean(est["log_likelihood"][-k:])
        except Exception:
            nll = 10
        # print(f"Iteration {self.itr[-1]}: NLL = {nll:.3f}, params = {params}")
        # axs[-1].plot(self.itr[-1], nll, color="k", ls="none", marker="o")
        # plt.pause(1e-3)
        return nll

    def batchrun(self, params):
        import concurrent.futures

        import tqdm.auto as tqdm

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.__call__, p): idx for idx, p in enumerate(params)
            }
            nll = [10] * len(futures)
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            idx = futures[future]
            nll[idx] = future.result()
        return nll


if __name__ == "__main__":
    from skopt import gp_minimize

    data = pl.read_parquet(
        "data/DigiBatt-BAK-5000-N21700CG-006-GITT-data.parquet"
    )  # .filter(pl.col("Cycle").is_between(4, 4))
    print(data.columns)
    t, I, V = data.select(["Time_h", "I_A", "U_V"]).to_numpy().T
    t = (t - t[0]) * 3600
    _t = np.arange(0, t[-1] / 100, 10)
    _I = _make_interpolator(t, I)(_t)
    _V = _make_interpolator(t, V)(_t)
    print(f"Resampled from {t.size} to {_t.size} samples")
    R = np.diag([1e-2])
    x0 = np.array([0.5, 0.0, 0.0])
    P = np.diag([1e-2, 1e-6, 1e-8])
    p = np.array([1, 1])

    fig, axs = plt.subplots()
    axs.plot(_t, _V)
    plt.show()
    fig, axs = plt.subplots(
        1, 4, figsize=(3 * 4, 4), constrained_layout=True, sharex=True
    )

    objective = Objective(
        t=_t,
        I=_I,
        V=_V,
        P=P,
        R=R,
        x0=x0,
        p=p,
        axs=axs,
    )
    _x0 = [np.random.uniform(-10, -1, size=3).tolist() for _ in range(50)]
    _y0 = objective.batchrun(_x0)
    opt = gp_minimize(
        objective,
        dimensions=[
            (-10.0, -1.0),
            (-10.0, -1.0),
            (-10.0, -1.0),
        ],
        n_calls=100,
        n_initial_points=20,
        n_jobs=-1,
        verbose=True,
        x0=_x0,
        y0=_y0,
    )
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(opt.func_vals, marker="o", mfc="w", color="k", ls="--")
    axs[1].plot(np.vstack(opt.x_iters)[:, 0], marker="o", mfc="w", color="k", ls="--")
    axs[2].plot(np.vstack(opt.x_iters)[:, 1], marker="o", mfc="w", color="k", ls="--")
    axs[3].plot(np.vstack(opt.x_iters)[:, 2], marker="o", mfc="w", color="k", ls="--")

    Q = np.diag(10 ** np.array(opt.x, dtype=float))
    est = run_ukf(_t, _I, _V, P, Q, R, x0, p)
    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(_t, _V, color="k", label="Measured")
    axs[0].plot(_t, est["measurement"][0, :], color="r", label="Estimated")
    axs[1].plot(_t, est["state"][0, :], color="k", label="Estimated SOC")
    axs[2].plot(_t, est["state"][1, :], color="k", label="Estimated Polarization")
    axs[3].plot(_t, est["state"][2, :], color="k", label="Estimated Hysteresis")

    # opt = scipy.optimize.differential_evolution(
    #     objective,
    #     args=(
    #         _t,
    #         _I,
    #         _V,
    #         P,
    #         R,
    #         x0,
    #         p,
    #         axs,
    #         itr,
    #     ),
    #     # constraints=scipy.optimize.LinearConstraint(A, lb=0.01, ub=1),
    #     # maxiter=10,
    #     bounds=[
    #         # (0.5, 1.0),  # soh
    #         # (0.8, 1.5),  # sor
    #         # (0.0, 1.0),  # sto_0_ne
    #         # (0.0, 1.0),  # sto_1_ne
    #         # (0.0, 1.0),  # sto_0_pe
    #         # (0.0, 1.0),  # sto_1_pe
    #         (-10, -2),  # log_q1
    #         (-10, -2),  # log_q2
    #         (-10, -2),  # log_q3
    #     ],
    #     disp=True,
    #     x0=np.array([-8, -8, -8]),
    #     workers=-1,
    # )
    print(opt)
    plt.show()
