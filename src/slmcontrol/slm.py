import cv2 as cv
import numpy as np
import screeninfo
from multiprocessing import Process, Event, Value
from time import sleep
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray

used_ids = []


class SLMDisplay:
    """
    A class to control a Spatial Light Modulator (SLM).

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

        # Current write buffer index (0 or 1) - shared between processes
        self._write_buffer_idx = Value("i", 0)

        # Event to signal new frame is ready
        self._frame_ready = Event()

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

        # Start by displaying buffer 0
        display_buffer_idx = 0
        display_array = array_0

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.imshow(self.window_name, display_array)
        cv.moveWindow(self.window_name, self.monitor.x, self.monitor.y)
        cv.setWindowProperty(
            self.window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN
        )
        cv.waitKey(1)

        while not self._shutdown.is_set():
            # Check if a new frame is ready
            if self._frame_ready.is_set():
                # Swap to the buffer that was just written
                display_buffer_idx = self._write_buffer_idx.value
                display_array = array_0 if display_buffer_idx == 0 else array_1
                # Clear the event
                self._frame_ready.clear()

            # Display the current buffer
            cv.imshow(self.window_name, display_array)
            cv.waitKey(1)

        # Clean up
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

        # Determine which buffer to write to (opposite of current display buffer)
        # The child reads _write_buffer_idx AFTER we set the event, so we write to the
        # "next" buffer by toggling the index
        current_idx = self._write_buffer_idx.value
        next_idx = 1 - current_idx  # Toggle between 0 and 1

        # Write to the back buffer (not currently being displayed)
        write_array = self._array_0 if next_idx == 0 else self._array_1
        np.copyto(write_array, holo)

        # Update the write buffer index and signal the child process
        self._write_buffer_idx.value = next_idx
        self._frame_ready.set()

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
        self.process.join()

        # Clean up both shared memory buffers
        self.buffer_0.close()
        self.buffer_0.unlink()
        self.buffer_1.close()
        self.buffer_1.unlink()

        # Remove monitor_id from used_ids
        used_ids.remove(self.monitor_id)
