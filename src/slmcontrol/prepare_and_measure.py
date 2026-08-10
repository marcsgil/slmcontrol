from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from slmcontrol.slm import SLMDisplay
from tqdm import trange


def prepare_and_measure(
    prepare: Callable,
    measure: Callable,
    slm: SLMDisplay,
    sleep_time: float,
    nsamples: int,
) -> None:
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
    None

    The preparation buffer is limited to two frames. If a callback raises, any
    preparation or measurement work which has not yet started is cancelled and
    the exception is immediately re-raised. Callbacks already running in a
    worker thread cannot be interrupted.
    """
    if nsamples < 0:
        raise ValueError("nsamples must be non-negative")
    if sleep_time < 0:
        raise ValueError("sleep_time must be non-negative")

    prepare_exec = ThreadPoolExecutor(max_workers=2)
    measure_exec = ThreadPoolExecutor(max_workers=1)
    prepare_futures: dict[int, Future] = {}
    next_to_submit = 0
    measure_future = None

    def submit_next() -> None:
        nonlocal next_to_submit
        if next_to_submit < nsamples:
            prepare_futures[next_to_submit] = prepare_exec.submit(
                prepare, next_to_submit
            )
            next_to_submit += 1

    try:
        # Keep at most two hologram computations in flight or ready to consume.
        submit_next()
        submit_next()

        for n in trange(nsamples):
            if measure_future is not None:
                # Finish the previous capture before changing the displayed frame.
                measure_future.result()

            holo = prepare_futures.pop(n).result()
            slm.updateArray(holo, sleep_time=sleep_time)
            measure_future = measure_exec.submit(measure, n)
            submit_next()

        if measure_future is not None:
            measure_future.result()
    except BaseException:
        prepare_exec.shutdown(wait=False, cancel_futures=True)
        measure_exec.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        prepare_exec.shutdown(wait=True)
        measure_exec.shutdown(wait=True)
