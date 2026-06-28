from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from slmcontrol.slm import SLMDisplay
from tqdm import trange


def prepare_and_measure(
    prepare: Callable,
    measure: Callable,
    slm: SLMDisplay,
    sleep_time: float,
    nsamples: int,
):
    """
    Display a series of holograms on the SLM and perform a measurement for each,
    pipelining hologram computation with SLM settling and measurement.

    Hologram computation runs concurrently with SLM settling and measurement.
    Prepared frames are consumed in index order, even when later computations
    finish first. Each frame's measurement is completed before the SLM is
    advanced to the next hologram, so the camera never sees a partial
    transition. Exceptions raised in either thread are re-raised on the main
    thread.

    Parameters
    ----------
    prepare : callable (n: int) -> ndarray
        Given a frame index n, returns the hologram to display. Should be
        prepared via functools.partial if it requires additional arguments.
    measure : callable (n: int) -> None
        Called after the SLM has settled for frame n. Responsible for
        performing and storing the measurement — e.g. capturing a camera image
        into a preallocated array. Should be prepared via functools.partial if
        it requires additional arguments.
    slm : SLMDisplay
        SLM interface. Must expose updateArray(hologram, sleep_time).
    sleep_time : float
        Time in seconds to wait after updating the SLM, to allow the liquid
        crystal layer to settle physically.
    nsamples : int
        Number of frames to acquire.

    Returns
    -------
    futures : list of Future
        The completed Future objects for each prepare task, in frame order.
    """
    if nsamples < 0:
        raise ValueError("nsamples must be non-negative")
    if sleep_time < 0:
        raise ValueError("sleep_time must be non-negative")

    with (
        ThreadPoolExecutor(max_workers=2) as prepare_exec,
        ThreadPoolExecutor(max_workers=1) as measure_exec,
    ):
        prepare_futures = [
            prepare_exec.submit(prepare, n) for n in range(nsamples)
        ]
        measure_future = None

        for n in trange(nsamples):
            holo = prepare_futures[n].result()

            if measure_future is not None:
                # Finish the previous capture before changing the displayed frame.
                measure_future.result()

            slm.updateArray(holo, sleep_time=sleep_time)
            measure_future = measure_exec.submit(measure, n)

        if measure_future is not None:
            measure_future.result()

    return prepare_futures
