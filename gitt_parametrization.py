import pathlib

import numpy as np
import polars as pl
import tqdm.auto as tqdm

from utils import determine_direction, fit_decay, fit_polarization, interpolate


def load_data():
    path = (
        pathlib.Path(__file__)
        .resolve()
        .parent.joinpath("data/DigiBatt-BAK-5000-N21700CG-006-GITT-data.parquet")
    )
    rename = {
        "Time_h": "Test Time / h",
        "U_V": "Voltage / V",
        "I_A": "Current / A",
        "DataSet": "Record Count / 1",
    }
    tol_current = 1e-1
    tol_voltage = 0  # 5e-3
    expr_step = (
        1
        + pl.when(
            (pl.col("Current / A").round(1).diff().abs() > tol_current)
            & (pl.col("Voltage / V").diff().abs() > tol_voltage)
        )
        .then(1)
        .otherwise(0)
        .cum_sum()
    )

    expr_cap = (
        (pl.col("Test Time / h").diff() * pl.col("Current / A").shift(1))
        .fill_null(0)
        .cum_sum()
    ).over("Direction / 1")

    expr_soc = (pl.col("Capacity / Ah") - pl.col("Capacity / Ah").min()) / (
        pl.col("Capacity / Ah").max() - pl.col("Capacity / Ah").min()
    )

    expr_mode = (
        pl.when(pl.col("Current / A").round(1).abs() < tol_current)
        .then(pl.lit("Rest"))
        .otherwise(
            pl.when(pl.col("Current / A") > 0)
            .then(pl.lit("Charge"))
            .otherwise(pl.lit("Discharge"))
        )
    )
    data = (
        pl.read_parquet(path)
        .rename(rename)
        .with_columns(expr_step.alias("Step Count / 1"), expr_mode.alias("Mode / 1"))
        .with_columns(
            pl.when(pl.col("Step Count / 1") == 3)
            .then(pl.lit("Charge"))
            .when(pl.col("Step Count / 1") < 3)
            .then(None)
            .otherwise(pl.col("Mode / 1").map_batches(determine_direction))
            .alias("Direction / 1")
        )
        .with_columns(expr_cap.alias("Capacity / Ah"))
        .with_columns(expr_soc.over("Direction / 1").alias("State of Charge / 1"))
        .with_columns((pl.col("Test Time / h") * 3600).alias("Test Time / s"))
        .with_columns(pl.len().over("Step Count / 1").alias("_step_length"))
    )
    return data.drop_nulls()


def fit_ocv(data: pl.DataFrame):
    results = []
    for step in tqdm.tqdm(
        data.filter(pl.col("Mode / 1") == "Rest").partition_by(
            "Step Count / 1", maintain_order=True
        )
    ):
        n = step.select("Step Count / 1").to_series().first()
        direction = step.select("Direction / 1").to_series().first()

        t = step.select("Test Time / h").to_series().to_numpy()
        t = 3600 * (t - t[0])
        V = step.select("Voltage / V").to_series().to_numpy()
        soc = step.select("State of Charge / 1").to_series().to_numpy()
        results.append(fit_decay(t, V))
        results[-1].update(
            {
                "Step Count / 1": n,
                "Direction / 1": direction,
                "State of Charge / 1": soc.mean(),
            }
        )
    return pl.DataFrame(results)


def fit_ecm(data: pl.DataFrame):
    results = []
    for step in tqdm.tqdm(
        data.filter(pl.col("Mode / 1") != "Rest").partition_by(
            "Step Count / 1", maintain_order=True
        )
    ):
        n = step.select("Step Count / 1").to_series().first()
        direction = step.select("Direction / 1").to_series().first()
        chunk = data.filter(pl.col("Step Count / 1").is_between(n - 1, n + 1))  # type:ignore

        t = chunk.select("Test Time / h").to_series().to_numpy()
        t = 3600 * (t - t[0])
        I = chunk.select("Current / A").to_series().to_numpy()
        V = chunk.select("Voltage / V").to_series().to_numpy()
        soc = chunk.select("State of Charge / 1").to_series().to_numpy()
        ocv = chunk.select("Open-circuit voltage / V").to_series().to_numpy()
        results.append(fit_polarization(t, V - ocv, I))
        results[-1].update(
            {
                "Direction / 1": direction,
                "State of Charge / 1": soc.mean(),
            }
        )
    return pl.DataFrame(results)


def combine_results(ocv: pl.DataFrame, ecm: pl.DataFrame):
    soc = np.linspace(0, 1, 101)
    direction = ["Charge", "Discharge"]
    results = []
    for d in direction:
        results.append(
            {
                "State of Charge / 1": soc,
                "Direction / 1": [d] * soc.size,
            }
        )
        for df in [ocv, ecm]:
            for col in [
                "Open-circuit voltage / V",
                "R0 / Ohm",
                "R1 / Ohm",
                "Tau1 / s",
                "Log Tau1 / 1",
            ]:
                if col in df:
                    results[-1][col] = interpolate(
                        *df.filter(pl.col("Direction / 1") == d)
                        .select("State of Charge / 1", col)
                        .to_numpy()
                        .T,
                        soc,  # type:ignore
                    )

    return pl.concat([pl.DataFrame(r) for r in results], how="diagonal_relaxed")


def interp_ocv(soc: np.ndarray, direction: np.ndarray, ocv: pl.DataFrame) -> np.ndarray:
    mask_charge = direction == "Charge"
    mask_discharge = direction == "Discharge"
    ocv_charge = interpolate(
        *ocv.filter(pl.col("Direction / 1") == "Charge")
        .select("State of Charge / 1", "Open-circuit voltage / V")
        .to_numpy()
        .T,
        soc[mask_charge],  # type:ignore
    )
    ocv_discharge = interpolate(
        *ocv.filter(pl.col("Direction / 1") == "Discharge")
        .select("State of Charge / 1", "Open-circuit voltage / V")
        .to_numpy()
        .T,
        soc[mask_discharge],  # type:ignore
    )
    ocv_combined = np.empty_like(soc)
    ocv_combined[mask_charge] = ocv_charge
    ocv_combined[mask_discharge] = ocv_discharge
    return ocv_combined


if __name__ == "__main__":
    data = load_data()
    ocv = fit_ocv(data)
    expr_ocv = pl.struct("State of Charge / 1", "Direction / 1").map_batches(
        lambda s: interp_ocv(
            s.struct.field("State of Charge / 1").to_numpy(),
            s.struct.field("Direction / 1").to_numpy(),
            ocv,
        ),
        return_dtype=pl.Float64,
    )

    ecm = fit_ecm(
        data.with_columns(
            expr_ocv.alias("Open-circuit voltage / V"),
        )
    )

    combined = combine_results(
        ocv.select("Direction / 1", "State of Charge / 1", "Open-circuit voltage / V"),
        ecm.select(
            "Direction / 1",
            "State of Charge / 1",
            "R0 / Ohm",
            "R1 / Ohm",
            "Tau1 / s",
            "Log Tau1 / 1",
        ),
    )
    ocv.write_csv("data/gitt_ocv.csv")
    ecm.write_csv("data/gitt_ecm.csv")
    combined.write_csv("data/gitt_results.csv")
