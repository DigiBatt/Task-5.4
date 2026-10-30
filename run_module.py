import concurrent.futures
from typing import Literal, NamedTuple

import numpy as np

from module import Module, ModuleSolution


class Results(NamedTuple):
    gitt: ModuleSolution
    ici: ModuleSolution


def run_sim(
    *,
    time: np.ndarray,
    c_rate: np.ndarray,
    cells_in_parallel: int,
    cells_in_series: int,
    ecm_parameters: dict,
    module_parameters: dict,
    parameters_source: Literal["ici", "gitt"],
) -> ModuleSolution:
    module = Module(
        cells_in_parallel=cells_in_parallel,
        cells_in_series=cells_in_series,
        cell_parameters=ecm_parameters,
        module_parameters=module_parameters,
        parameters_source=parameters_source,
    )
    sim = module.simulate(
        time, c_rate * module.module_parameters["Nominal Capacity / Ah"]
    )
    return sim


def run_both(
    *,
    time: np.ndarray,
    c_rate: np.ndarray,
    cells_in_parallel: int,
    cells_in_series: int,
    ecm_parameters: dict,
    module_parameters: dict,
    parallel: bool = True,
):
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            gitt_future = executor.submit(
                run_sim,
                time=time,
                c_rate=c_rate,
                cells_in_parallel=cells_in_parallel,
                cells_in_series=cells_in_series,
                ecm_parameters=dict(ecm_parameters),
                module_parameters=dict(module_parameters),
                parameters_source="gitt",
            )
            ici_future = executor.submit(
                run_sim,
                time=time,
                c_rate=c_rate,
                cells_in_parallel=cells_in_parallel,
                cells_in_series=cells_in_series,
                ecm_parameters=dict(ecm_parameters),
                module_parameters=dict(module_parameters),
                parameters_source="ici",
            )
            gitt_result = gitt_future.result()
            ici_result = ici_future.result()
    else:
        gitt_result = run_sim(
            time=time,
            c_rate=c_rate,
            cells_in_parallel=cells_in_parallel,
            cells_in_series=cells_in_series,
            ecm_parameters=dict(ecm_parameters),
            module_parameters=dict(module_parameters),
            parameters_source="gitt",
        )
        ici_result = run_sim(
            time=time,
            c_rate=c_rate,
            cells_in_parallel=cells_in_parallel,
            cells_in_series=cells_in_series,
            ecm_parameters=dict(ecm_parameters),
            module_parameters=dict(module_parameters),
            parameters_source="ici",
        )
    return Results(gitt=gitt_result, ici=ici_result)
