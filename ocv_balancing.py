import pathlib

import numpy as np
import polars as pl
from scipy.interpolate import interp1d


def soc_to_sto(soc: np.ndarray, sto_0, sto_1) -> np.ndarray:
    return sto_0 + (sto_1 - sto_0) * soc


def sto_to_soc(sto: np.ndarray, sto_0, sto_1) -> np.ndarray:
    return (sto - sto_0) / (sto_1 - sto_0)


def soc_to_ocp(
    soc: np.ndarray, sto_0: float | np.ndarray, sto_1: float | np.ndarray, lut: interp1d
) -> np.ndarray:
    sto = soc_to_sto(soc, sto_0, sto_1)
    return lut(sto)


def soc_to_ocv(
    soc: np.ndarray,
    sto_0_ne: float | np.ndarray,
    sto_1_ne: float | np.ndarray,
    sto_0_pe: float | np.ndarray,
    sto_1_pe: float | np.ndarray,
    lut_ne: interp1d,
    lut_pe: interp1d,
) -> np.ndarray:
    ocp_ne = soc_to_ocp(soc, sto_0_ne, sto_1_ne, lut_ne)
    ocp_pe = soc_to_ocp(soc, sto_0_pe, sto_1_pe, lut_pe)
    return ocp_pe - ocp_ne


def load_ne():
    path = pathlib.Path(
        r"C:\Users\KristianFredrikKlepp\OneDrive - Corvus Energy\Documents\data\digibatt\WP5\N21700-BAK-CG50_electrode_level_data\OCV_measurements\anode"
    )
    files = {
        "Lithiation": path.joinpath(
            "N21700-BAK-CG50_anode_qOCV_lithiation_HEU_DigiBatt_DLR_20250730.bdf.csv"
        ),
        "Delithiation": path.joinpath(
            "N21700-BAK-CG50_anode_qOCV_delithiation_HEU_DigiBatt_DLR_20250730.bdf.csv"
        ),
    }
    data = pl.concat(
        [
            pl.read_csv(file).with_columns(pl.lit(mode).alias("Mode"))
            for mode, file in files.items()
        ]
    ).with_columns(
        pl.col("StateOfCharge").alias("Degree of Lithiation / 1"),
    )
    return data


def load_pe():

    ah = pl.col("ah-step (Ah)")
    chg = pl.read_csv(
        r"C:\Users\KristianFredrikKlepp\OneDrive - Corvus Energy\Documents\data\digibatt\WP5\N21700-BAK-CG50_electrode_level_data\OCV_measurements\cathode\N21700-BAK-CG50-DLR-cathode-02-qOCV-delithiation.csv"
    ).with_columns(
        **{
            "Degree of Lithiation / 1": 1 - ((ah - ah.min()) / (ah.max() - ah.min())),
            "Mode": pl.lit("Delithiation"),
        }
    )
    dhg = pl.read_csv(
        r"C:\Users\KristianFredrikKlepp\OneDrive - Corvus Energy\Documents\data\digibatt\WP5\N21700-BAK-CG50_electrode_level_data\OCV_measurements\cathode\N21700-BAK-CG50-DLR-cathode-02-qOCV-lithiation.csv"
    ).with_columns(
        **{
            "Degree of Lithiation / 1": 1 - ((ah - ah.min()) / (ah.max() - ah.min())),
            "Mode": pl.lit("Lithiation"),
        }
    )

    data = pl.concat([chg, dhg], how="diagonal_relaxed").rename(
        {"voltage (V)": "Voltage / V"}
    )

    return data


def load_cell():
    poass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    ne = load_ne()
    pe = load_pe()
    axs[0].plot(ne["Degree of Lithiation / 1"], ne["Voltage / V"])
    axs[1].plot(pe["Degree of Lithiation / 1"], pe["Voltage / V"])
    print(ne)
    print(pe)
    plt.show()
