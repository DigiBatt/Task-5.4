import concurrent.futures
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.interpolate
import scipy.optimize
import statespace
import tqdm.auto as tqdm


def _make_interpolator(x, y):
    return scipy.interpolate.interp1d(
        x[::10],
        y[::10],
        kind="quadratic",
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
        self._Th = 0.4
        self._Cnom = 5.0
        luts = {}
        for d in ["Charge", "Discharge"]:
            for p in ["Open-circuit voltage / V", "R0 / Ohm", "R1 / Ohm", "Tau1 / s"]:
                mask = pl.col("Direction / 1") == d
                x, y = data.filter(mask).select(["State of Charge / 1", p]).to_numpy().T
                luts[f"{d} {p}"] = _make_interpolator(x, y)  # type: ignore

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

        Voc = self._hmix(
            hst,
            self.luts["Charge Open-circuit voltage / V"](soc),
            self.luts["Discharge Open-circuit voltage / V"](soc),
        )

        return Voc + pol + u * R0


def simulate(params, t, I, V):
    soc_0, pol_0, hyst_0, soh, sor = params
    ecm = ECM(samples=1)
    sim = ecm.run(
        t,
        u=I[None, :],
        x0=np.array([soc_0, pol_0, hyst_0])[:, None],
        p=np.array([soh, sor])[:, None],
    )["z"][0, :]
    return sim


def objective(params, t, I, V):
    sim = simulate(params, t, I, V)
    return np.mean(np.sqrt((V - sim) ** 2))


def fit_cycle(t, I, V, x0: np.ndarray | None = None):
    def wrapper(_t, soc_0, pol_0, hyst_0, soh, sor):
        params = np.array([soc_0, pol_0, hyst_0, soh, sor])
        return simulate(params, _t, I, V)

    popt, pcov = scipy.optimize.curve_fit(
        wrapper,
        t,
        V,
        p0=x0 if x0 is not None else np.array([0.5, 0.0, 0.0, 1.0, 1.0]),
        bounds=([0, -1, -1, 0.1, 0.1], [1, 1, 1, 2, 5]),
        maxfev=10,
    )
    return popt
    # if x0 is None:
    #     x0 = np.array([0.5, 0.0, 0.0, 1.0, 1.0])
    # opt = scipy.optimize.minimize(
    #     objective,
    #     x0=x0,
    #     args=(t, I, V),
    #     bounds=[(0, 1), (-1, 1), (-1, 1), (0.1, 2), (0.1, 5)],
    # )
    # return opt.x


def fit_file(file: pathlib.Path):
    out_path = (
        pathlib.Path(__file__).resolve().parent.joinpath(f"data/{file.stem}.parquet")
    )
    if out_path.exists():
        tqdm.tqdm.write(f"Skipping {file.stem} because {out_path} exists")
        return pl.read_parquet(out_path)
    results = []
    x0 = None
    tqdm.tqdm.write(f"Fitting {file.stem}")
    for cycle in pl.read_parquet(file).partition_by("Cycle", maintain_order=True):
        t = 1e-3 * cycle.select("Program Duration#ms").to_series().to_numpy()
        V = cycle.select("Voltage#V").to_series().to_numpy()
        I = cycle.select("Current#A").to_series().to_numpy()
        opt = fit_cycle(t, I, V, x0=x0)
        tqdm.tqdm.write(
            f"Fitted cycle {cycle.select('Cycle').to_series().mean()} with {opt}"
        )
        x0 = opt
        param = {
            "State of Health / 1": opt[3],
            "State of Resistance / 1": opt[4],
            "Cycle Count / 1": cycle.select("Cycle").to_series().mean(),
            "Timestamp / datetime": cycle.select("Timestamp").to_series().mean(),
            "Test Time / s": np.mean(t),
            "File name / 1": file.stem,
            "Ambient Temperature / degC": cycle.select("Tenv#degC").to_series().mean(),
            "Surface Temperature / degC": cycle.select("T1#degC").to_series().mean(),
        }
        results.append(param)
    tqdm.tqdm.write(f"Fitted {len(results)} cycles in {file.stem}")
    results = pl.DataFrame(results)
    results.write_parquet(out_path)
    return results


if __name__ == "__main__":
    import concurrent.futures
    import pathlib

    import matplotlib.pyplot as plt
    import tqdm

    path = pathlib.Path(
        "C:\\Users\\KristianFredrikKlepp\\OneDrive - Corvus Energy\\Documents\\data\\digibatt\\WP4\\cycles"
    ).resolve()

    fig, axs = plt.subplots()
    plt.pause(1e0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fit_file, file) for file in path.glob("*.parquet")]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            df = future.result()
            axs.plot(
                df.select("State of Health / 1").to_series().to_numpy(),
                df.select("State of Resistance / 1").to_series().to_numpy(),
                color="k",
                ls="none",
                marker="o",
            )
            plt.pause(1e0)
