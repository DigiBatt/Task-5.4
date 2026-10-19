import pathlib

import numpy as np
import polars as pl
import scipy.optimize as optimize
from scipy.integrate import cumulative_trapezoid

from ecm_model import ECM


def load_file(file: pathlib.Path):
    data = pl.read_parquet(file)
    tol_current = 1e-3
    tol_voltage = 1e-3
    diff_current = pl.col("Current / A").diff().abs()
    diff_voltage = pl.col("Voltage / V").diff().abs()
    expr_step = (
        pl.when((diff_current > tol_current) & (diff_voltage > tol_voltage))
        .then(1)
        .otherwise(0)
    ).cum_sum()
    return data.with_columns(
        (pl.col("Program Duration#ms") / 1000 / 3600).alias("Test Time / h"),
        (pl.col("Current#A")).alias("Current / A"),
        (pl.col("Voltage#V")).alias("Voltage / V"),
    ).with_columns(expr_step.alias("Step Count / 1"))


def agg_steps(data: pl.DataFrame):
    agg = data.group_by("Step Count / 1").agg(
        pl.col("Voltage / V").max().alias("Max Voltage / V"),
        pl.col("Voltage / V").min().alias("Min Voltage / V"),
        pl.col("Current / A").max().alias("Max Current / A"),
        pl.col("Current / A").min().alias("Min Current / A"),
        pl.col("Current / A").mean().alias("Mean Current / A"),
        pl.col("Test Time / h").first().alias("First Test Time / h"),
        pl.col("Test Time / h").last().alias("Last Test Time / h"),
        (pl.col("Test Time / h").last() - pl.col("Test Time / h").first()).alias(
            "Step Duration / h"
        ),
    )
    return agg


def run_profile(params, t, I):
    soc_0, pol_0, hyst_0, soh, sor = params
    ecm = ECM(
        parameters={
            "Initial State of Charge / 1": soc_0,
            "Initial Polarization / V": pol_0,
            "Initial Hysteresis / 1": hyst_0,
            "Mean State of Health / 1": soh,
            "Mean State of Resistance / 1": sor,
            "Stdev State of Health / 1": 0,
            "Stdev State of Resistance / 1": 0,
        },
        parameters_source="ici",
    )

    return ecm.simulate(t * 3600, I[np.newaxis, :], progress=False)


def objective(params, t, I, V):
    sim = run_profile(params, t, I)
    rmse = np.sqrt(((sim.voltage.ravel() - V) ** 2).mean())
    # print(f"RMSE: {rmse:.5f}, log10(RMSE): {np.log10(rmse):.5f}")
    return rmse


def fit_file(file: pathlib.Path):
    data = load_file(file)

    t, I, V = data.select("Test Time / h", "Current / A", "Voltage / V").to_numpy().T
    Q = cumulative_trapezoid(I, t, initial=0)
    q = Q.min()
    print()
    SOC = (Q - q) / (Q.max() - q)
    result = optimize.differential_evolution(
        objective,
        bounds=[(0, 1), (-1, 1), (-1, 1), (0.5, 1.2), (0.4, 3)],
        args=(t[:1000], I[:1000], V[:1000]),
        disp=True,
        x0=[SOC[0], 0.0, 0.0, 1.0, 1.0],
    )
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = pathlib.Path(
        "C:\\Users\\KristianFredrikKlepp\\OneDrive - Corvus Energy\\Documents\\data\\digibatt\\WP4\\checkups"
    ).resolve()
    for file in path.glob("*.parquet"):
        data = load_file(file)
        steps = agg_steps(data)
        fig, axs = plt.subplots(2, 1, sharex=True)
        for step in data.partition_by("Step Count / 1"):
            axs[0].plot(
                step["Test Time / h"],
                step["Current / A"],
                label=f"Step {step['Step Count / 1'][0]}",
            )
            axs[1].plot(
                step["Test Time / h"],
                step["Voltage / V"],
                label=f"Step {step['Step Count / 1'][0]}",
            )

        chg = steps.filter(pl.col("Mean Current / A") > 0)
        chg = steps.filter(pl.col("Mean Current / A") < 0)
        plt.show()

        # result = fit_file(file)
        # print(file, result)
