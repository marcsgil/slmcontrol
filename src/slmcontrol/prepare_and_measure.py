from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Callable
from slmcontrol import SLMDisplay
from tqdm import trange

def prepare_and_measure(prepare: Callable, measure: Callable, slm: SLMDisplay, sleep_time: float, nsamples: int):
    """
    Display a series of holograms on the SLM and perform a measurement for each,
    pipelining hologram computation and measurement with SLM settling.

    Hologram computation runs concurrently with SLM settling and measurement,
    and each measurement runs concurrently with the next frame's SLM settling,
    so that neither step is on the critical path. Exceptions raised in either
    thread are re-raised on the main thread rather than silently hanging.

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
        The Future objects for each prepare task. Inspect these after the
        call to check for any deferred exceptions.
    """
    hologram_queue = queue.Queue(maxsize=2)

    def compute_and_enqueue(n):
        hologram_queue.put(prepare(n))

    with ThreadPoolExecutor(max_workers=2) as prepare_exec, \
         ThreadPoolExecutor(max_workers=1) as measure_exec:

        prepare_futures = [prepare_exec.submit(compute_and_enqueue, n) for n in range(nsamples)]
        measure_future = None

        for n in trange(nsamples):
            holo = hologram_queue.get()
            slm.updateArray(holo, sleep_time=sleep_time)

            if measure_future is not None:
                measure_future.result()

            measure_future = measure_exec.submit(measure, n)
            prepare_futures[n].result()

        if measure_future is not None:
            measure_future.result()

    return prepare_futures
