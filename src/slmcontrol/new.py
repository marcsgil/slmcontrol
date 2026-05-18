import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Callable
from slmcontrol import SLMDisplay
from tqdm import trange

def prepare_and_measure(prepare: Callable, measure: Callable, slm: SLMDisplay, sleep_time: float, nsamples: int):
    """
    Display a series of holograms on the SLM and perform a measurement for each,
    pipelining hologram computation with SLM settling and measurement.

    Hologram computation runs concurrently with SLM settling and measurement,
    so prepare is never on the critical path. Each frame's measurement is
    completed before the SLM is advanced to the next hologram, so the camera
    never sees a partial transition. Exceptions raised in either thread are
    re-raised on the main thread rather than silently hanging.

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

    prepare_queue = queue.Queue(maxsize=2)

    def prepare_and_enqueue(n):
        prepare_queue.put((n, prepare(n)))

    with ThreadPoolExecutor(max_workers=2) as exec_prepare, ThreadPoolExecutor(max_workers=2) as exec_measure:
        futures_prepare = [exec_prepare.submit(prepare_and_enqueue, n) for n in range(nsamples)]
        
        for _ in trange(nsamples):
            n, holo = prepare_queue.get()
            slm.updateArray(holo, sleep_time=sleep_time)
            exec_measure.submit(measure, n)

            