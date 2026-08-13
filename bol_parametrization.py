import concurrent.futures
import pathlib

import numpy as np
import polars as pl
import pybamm
import pybop
import tqdm.auto as tqdm
from scipy.interpolate import interp1d


def build_optimizer(
    time: np.ndarray, current: np.ndarray, polarization: np.ndarray, n: int = 1
):
    """
    Builds a pybop optimizer for an ecm model
    time: Test time in seconds
    current: Current in Amperes - positive sign means charging
    polarization: Voltage minus OCV in Volts
    n: Number of RC elements in the model
    """
    model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": n})
    param = model.default_parameter_values
    # Set some parameters
    param.update(
        {
            "Open-circuit voltage [V]": 0,
            "Entropic change [V/K]": 0,
            "Upper voltage cut-off [V]": 1e6,
            "Lower voltage cut-off [V]": -1e6,
        }
    )
    # Optimization targets
    param.update(
        {
            "R0 [Ohm]": pybop.Parameter(
                initial_value=1e-2,
                distribution=pybop.LogUniform(1e-4, 1e0),
                transformation=pybop.LogTransformation(),
            ),
        }
    )
    for i in range(1, n + 1):
        param.update(
            {
                f"Element-{i} initial overpotential [V]": pybop.Parameter(
                    initial_value=0,
                    distribution=pybop.Uniform(-1, 1),
                ),
                f"R{i} [Ohm]": pybop.Parameter(
                    initial_value=1e-2,
                    distribution=pybop.LogUniform(1e-4, 1e0),
                    transformation=pybop.LogTransformation(),
                ),
                f"Tau{i} [s]": pybop.Parameter(
                    initial_value=100,
                    distribution=pybop.LogUniform(1e0, 1e4),
                    transformation=pybop.LogTransformation(),
                ),
                f"C{i} [F]": pybamm.Parameter(f"Tau{i} [s]")
                / pybamm.Parameter(f"R{i} [Ohm]"),
            }
        )

    dataset = pybop.Dataset(
        {
            "Time [s]": time,
            "Current [A]": -current,
            "Voltage [V]": polarization,
        }
    )
    cost = pybop.RootMeanSquaredError(dataset)
    simulator = pybop.pybamm.Simulator(
        model, parameter_values=param, protocol=dataset, solver=pybamm.IDAKLUSolver()
    )
    problem = pybop.Problem(simulator=simulator, cost=cost)

    options = pybop.SciPyDifferentialEvolutionOptions(seed=42, polish=True)
    optimizer = pybop.SciPyDifferentialEvolution(problem=problem, options=options)
    return optimizer


def fit_chunk(
    time: np.ndarray, current: np.ndarray, polarization: np.ndarray, n: int = 1
):
    """
    Fit a chunk of timeseries data to an ecm model
    time: Test time in seconds
    current: Current in Amperes - positive sign means charging
    polarization: Voltage minus OCV in Volts
    """
    optimizer = build_optimizer(time, current, polarization, n=n)
    results = optimizer.run()
    return results


def _direction(mode: pl.Series) -> pl.Series:
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


def _interpolate(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)(  # type: ignore
        z
    )


def load_data() -> pl.DataFrame:
    """
    Loads the GITT cell data
    """

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
            .otherwise(pl.col("Mode / 1").map_batches(_direction))
            .alias("Direction / 1")
        )
        .with_columns(expr_cap.alias("Capacity / Ah"))
        .with_columns(expr_soc.over("Direction / 1").alias("State of Charge / 1"))
        .with_columns((pl.col("Test Time / h") * 3600).alias("Test Time / s"))
        .with_columns(pl.len().over("Step Count / 1").alias("_step_length"))
    )
    return data.drop_nulls()


def extract_ocv(data: pl.DataFrame | None = None) -> pl.DataFrame:
    """
    Extracts the OCV from the GITT cell data
    """
    if data is None:
        data = load_data()

    ocv = (
        data.filter(pl.col("Mode / 1") == "Rest")
        .group_by("Step Count / 1")
        .agg(
            pl.col("Voltage / V").last().alias("Open-circuit voltage / V"),
            pl.col("State of Charge / 1").last().alias("State of Charge / 1"),
            pl.col("Direction / 1").last().alias("Direction / 1"),
        )
        .sort("Step Count / 1")
    )
    return ocv


def extract_parameters(
    data: pl.DataFrame | None = None, ocv: pl.DataFrame | None = None
) -> pl.DataFrame:
    """
    Extracts the parameters from the GITT cell data
    """
    if data is None:
        data = load_data()
    if ocv is None:
        ocv = extract_ocv(data)

    parameters = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        for step in data.filter(pl.col("Mode / 1") != "Rest").partition_by(
            "Step Count / 1", maintain_order=True
        ):
            i = step.select("Test Time / s").to_series().first() - 10  # type: ignore
            j = step.select("Test Time / s").to_series().last() + 60  # type: ignore
            chunk = data.filter(pl.col("Test Time / s").is_between(i, j, closed="both"))

            t = chunk.select("Test Time / s").to_series().to_numpy()
            I = chunk.select("Current / A").to_series().to_numpy()
            V = chunk.select("Voltage / V").to_series().to_numpy()
            soc = chunk.select("State of Charge / 1").to_series().to_numpy()
            direction = step.select("Direction / 1").to_series().first()  # type: ignore

            pol = V - _interpolate(
                *ocv.filter(pl.col("Direction / 1") == direction)
                .select("State of Charge / 1", "Open-circuit voltage / V")
                .to_numpy()
                .T,
                soc,  # type: ignore
            )

            futures[executor.submit(fit_chunk, t, I, pol, n=1)] = {
                "State of Charge / 1": soc.mean(),
                "Direction / 1": direction,
            }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            meta = futures[future]
            parameters.append(
                {
                    "State of Charge / 1": meta["State of Charge / 1"],
                    "Direction / 1": meta["Direction / 1"],
                    "R0 / Ohm": result.best_inputs["R0 [Ohm]"],
                    "R1 / Ohm": result.best_inputs["R1 [Ohm]"],
                    "Tau1 / s": result.best_inputs["Tau1 [s]"],
                }
            )
    return pl.DataFrame(parameters)


def combine_results(
    data: pl.DataFrame | None = None,
    ocv: pl.DataFrame | None = None,
    parameters: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Combines the OCV and parameters into a single dataframe
    """
    if data is None:
        data = load_data()
    if ocv is None:
        ocv = extract_ocv(data=data)
    if parameters is None:
        parameters = extract_parameters(data=data, ocv=ocv)
    results = {}
    soc = np.linspace(0, 1, 1001)
    for direction in ["Charge", "Discharge"]:
        d = {
            "State of Charge / 1": soc,
            "Open-circuit voltage / V": _interpolate(
                *ocv.filter(pl.col("Direction / 1") == direction)
                .select("State of Charge / 1", "Open-circuit voltage / V")
                .to_numpy()
                .T,
                soc,  # type: ignore
            ),
            "R0 / Ohm": _interpolate(
                *parameters.filter(pl.col("Direction / 1") == direction)
                .select("State of Charge / 1", "R0 / Ohm")
                .to_numpy()
                .T,
                soc,  # type: ignore
            ),
            "R1 / Ohm": _interpolate(
                *parameters.filter(pl.col("Direction / 1") == direction)
                .select("State of Charge / 1", "R1 / Ohm")
                .to_numpy()
                .T,
                soc,  # type: ignore
            ),
            "Tau1 / s": _interpolate(
                *parameters.filter(pl.col("Direction / 1") == direction)
                .select("State of Charge / 1", "Tau1 / s")
                .to_numpy()
                .T,
                soc,  # type: ignore
            ),
        }
        results[direction] = d
    return pl.concat(
        [
            pl.DataFrame(item).with_columns(pl.lit(key).alias("Direction / 1"))
            for key, item in results.items()
        ]
    )


if __name__ == "__main__":
    data = load_data()
    print(data)

    ocv = extract_ocv(data=data)
    ocv.drop("Step Count / 1").write_csv(
        pathlib.Path(__file__).resolve().parent.joinpath("data/gitt_ocv_soc.csv")
    )

    parameters = extract_parameters(data=data, ocv=ocv)
    parameters.write_csv(
        pathlib.Path(__file__).resolve().parent.joinpath("data/gitt_rc_parameters.csv")
    )
    results = combine_results(data=data, ocv=ocv, parameters=parameters)
    print(results)
    results.write_csv(
        pathlib.Path(__file__).resolve().parent.joinpath("data/gitt_ecm_parameters.csv")
    )
