import pathlib

import polars as pl


def load_data(path: str | pathlib.Path) -> pl.LazyFrame:
    # Path must be changed
    if isinstance(path, str):
        path = pathlib.Path(path).resolve()

    rename_to = {
        "Step": "_step",
        "Status": "_status",
        "Cycle": "_cycle",
        "Procedure": "_procedure",
        "AhStep#Ah": "Step Capacity / Ah",
        "Current#A": "Current / A",
        "Voltage#V": "Voltage / V",
        "_cell": "_cell",
        "_file": "_file",
    }

    data = (
        pl.scan_parquet(path, include_file_paths="_file")
        .with_columns(
            pl.col("_file")
            .str.split("/")
            .list.get(-1)
            .str.split("_")
            .list.get(2)
            .cast(pl.Int64)
            .alias("_cell")
        )
        .rename(rename_to, strict=False)
        .select(*rename_to.values())
    )

    return data


def aggregate_steps(data: pl.LazyFrame) -> pl.LazyFrame:
    steps = data.group_by("_file", "_procedure", "_step", "_cycle").agg(
        pl.col("Step Capacity / Ah").last().alias("Step Capacity / Ah"),
        pl.col("Current / A").mean().alias("Mean Current / A"),
        pl.col("Voltage / V").min().alias("Min Voltage / V"),
        pl.col("Voltage / V").max().alias("Max Voltage / V"),
        pl.col("_cell").first().alias("_cell"),
        pl.col("_status").first().alias("_status"),
    )
    return steps


def extract_soh(steps: pl.LazyFrame, data: pl.LazyFrame) -> pl.LazyFrame:

    mask = (
        (pl.col("_status") == "DCH")
        & (pl.col("Mean Current / A") < 0)
        & (pl.col("Max Voltage / V").is_between(4.1, 4.3))
        & (pl.col("Min Voltage / V").is_between(2.4, 2.6))
        & (pl.col("_procedure") == "DigiBatt_checkup_v2")
        & (pl.col("Mean Current / A").abs() < 0.3)
    )

    soh = steps.filter(mask).with_columns(
        (pl.col("Step Capacity / Ah") / pl.col("Step Capacity / Ah").max())
        .over("_cell")
        .alias("State of Health / 1"),
        pl.col("Step Capacity / Ah").max().over("_cell").alias("BOL Capacity / Ah"),
    )
    return soh


def extract_ocv(
    soh: pl.LazyFrame, steps: pl.LazyFrame, data: pl.LazyFrame
) -> pl.LazyFrame:
    data = data.join(
        soh.drop("Step Capacity / Ah"),
        on=["_file", "_procedure", "_step", "_cycle", "_status"],
        how="inner",
    ).with_columns(
        (1 - pl.col("Step Capacity / Ah") / pl.col("Step Capacity / Ah").max())
        .over("_file")
        .alias("State of Charge / 1")
    )
    return data


if __name__ == "__main__":
    path = pathlib.Path(
        "C:\\Users\\KristianFredrikKlepp\\OneDrive - Corvus Energy\\Documents\\data\\digibatt\\WP4\\checkups"
    ).resolve()

    data = load_data(path)
    steps = aggregate_steps(data)
    soh = extract_soh(steps, data)
    ocv = extract_ocv(soh, steps, data)
    ocv.select(
        ["State of Charge / 1", "Voltage / V", "State of Health / 1", "_file"]
    ).sink_parquet(
        pathlib.Path(__file__).resolve().parent.joinpath("data/ocv_soh.parquet")
    )
