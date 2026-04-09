import cv2 as cv
import numpy as np
import screeninfo
from multiprocessing import Process, Event, Queue
from queue import Empty, Full
from time import sleep
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray

used_ids = []


class SLMDisplay:
    """
    A class to control a Spatial Light Modulator (SLM).
    This class uses multiprocessing to manage the display in a separate process.
    When using this class in Python scripts (not imported modules), you must protect
    the instantiation with an `if __name__ == '__main__':` guard to prevent errors
    on macOS and Windows:

    Example:
        ```python
        import slmcontrol

        if __name__ == '__main__':
            slm = slmcontrol.SLMDisplay()
            # ... use the SLM
            slm.close()
        ```

    Note: This guard is NOT required when:
        - Using in Jupyter notebooks or IPython
        - Importing and using in modules (not the main script)
        - Running via test frameworks (pytest, unittest)

    Attributes:
        monitor_id (int): The ID of the monitor to use.
        width (int): The width of the SLM.
        height (int): The height of the SLM.
    """

    def __init__(self, monitor_id: int = -1) -> None:
        """
        Initialize the SLM instance.

        Args:
            monitor_id (int): The ID of the monitor to use. Defaults to the last monitor.
        """
        assert monitor_id not in used_ids, (
            "SLMDisplay instance already exists for this monitor."
        )
        used_ids.append(monitor_id)
        self.monitor_id = monitor_id
        self.window_name = f"SLM Display - Monitor {monitor_id}"
        self.monitor = screeninfo.get_monitors()[monitor_id]
        self.height = self.monitor.height
        self.width = self.monitor.width

        # Create two shared memory buffers for double buffering
        buffer_size = self.height * self.width
        self.buffer_0 = SharedMemory(create=True, size=buffer_size)
        self.buffer_1 = SharedMemory(create=True, size=buffer_size)

        # Create numpy array views for both buffers
        self._array_0 = np.ndarray(
            (self.height, self.width), dtype=np.uint8, buffer=self.buffer_0.buf
        )
        self._array_1 = np.ndarray(
            (self.height, self.width), dtype=np.uint8, buffer=self.buffer_1.buf
        )

        # Initialize both buffers to black
        self._array_0.fill(0)
        self._array_1.fill(0)

        # Current write buffer index (0 or 1) - only used in the main process
        self._write_buffer_idx = 0

        # Queue to pass buffer indices to the display process (maxsize=1: latest frame wins)
        self._frame_queue: Queue = Queue(maxsize=1)

        # Event to signal shutdown
        self._shutdown = Event()

        self.process = Process(target=self.run)
        self.process.start()

    def run(self) -> None:
        # Attach to both shared memory buffers by name
        buffer_0 = SharedMemory(name=self.buffer_0.name)
        buffer_1 = SharedMemory(name=self.buffer_1.name)

        # Create numpy array views for both buffers
        array_0 = np.ndarray(
            (self.height, self.width), dtype=np.uint8, buffer=buffer_0.buf
        )
        array_1 = np.ndarray(
            (self.height, self.width), dtype=np.uint8, buffer=buffer_1.buf
        )

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.imshow(self.window_name, array_0)
        cv.moveWindow(self.window_name, self.monitor.x, self.monitor.y)
        cv.setWindowProperty(
            self.window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN
        )
        cv.waitKey(1)

        while not self._shutdown.is_set():
            try:
                buffer_idx = self._frame_queue.get(timeout=0.05)
                cv.imshow(self.window_name, array_0 if buffer_idx == 0 else array_1)
            except Empty:
                pass
            cv.waitKey(1)

        # Clean up
        cv.destroyWindow(self.window_name)
        buffer_0.close()
        buffer_1.close()

    def updateArray(
        self, holo: NDArray[np.uint8], sleep_time: float | int = 0.15
    ) -> None:
        """
        Update the hologram displayed on the SLM.

        Args:
            holo: A 2D matrix of UInt8 values representing the hologram.
            sleep_time (float | int): Time to sleep after updating (in seconds) to allow display to refresh.
                       Set to 0 for maximum throughput (no waiting).
        """
        assert holo.shape == (self.height, self.width), "Invalid hologram shape."

        # Write to the back buffer (opposite of what was last sent to the display)
        next_idx = 1 - self._write_buffer_idx
        np.copyto(self._array_0 if next_idx == 0 else self._array_1, holo)
        self._write_buffer_idx = next_idx

        # Send the new buffer index to the display process; if the previous frame
        # hasn't been consumed yet, replace it so the display always shows the latest
        try:
            self._frame_queue.put_nowait(next_idx)
        except Full:
            try:
                self._frame_queue.get_nowait()
            except Empty:
                pass
            self._frame_queue.put_nowait(next_idx)

        # Optional sleep to allow the display to update
        if sleep_time > 0:
            sleep(sleep_time)

    def close(self) -> None:
        """
        Close the SLM window.
        """
        assert self.monitor_id in used_ids, (
            "SLMDisplay instance not found for this monitor."
        )

        # Signal the child process to shutdown
        self._shutdown.set()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()

        # Clean up both shared memory buffers
        self.buffer_0.close()
        self.buffer_0.unlink()
        self.buffer_1.close()
        self.buffer_1.unlink()

        # Remove monitor_id from used_ids
        used_ids.remove(self.monitor_id)
