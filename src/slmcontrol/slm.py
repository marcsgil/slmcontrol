import cv2 as cv
import numpy as np
import screeninfo
import socket
import struct
from multiprocessing import Process, Event, Value
from time import sleep
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray

used_ids = []


def _recv_exactly(sock: socket.socket, buf, n: int) -> None:
    view = memoryview(buf)
    received = 0
    while received < n:
        chunk = sock.recv_into(view[received:], n - received)
        if chunk == 0:
            raise EOFError(f"Socket closed after {received}/{n} bytes")
        received += chunk


class SLMDisplay:
    """
    A class to control a Spatial Light Modulator (SLM).

    Can be used in two modes:

    **Local mode** (default): drives a display directly on the current machine.
    This uses multiprocessing to manage the OpenCV window in a separate process.
    When using this mode in Python scripts (not imported modules), you must protect
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

    **Remote mode**: connects to an [`SLMServer`][src.slmcontrol.server.SLMServer]
    running on the machine physically attached to the SLM.  No local display is
    needed — the hologram is computed locally and sent over TCP.  Pass
    ``host="localhost"`` when the script runs on the SLM machine itself (e.g. via
    SSH), or the machine's IP address for a fully remote setup:

    Example:
        ```python
        slm = slmcontrol.SLMDisplay(host="localhost")
        slm.updateArray(holo)
        slm.close()
        ```

    See the [Remote Control](remote.md) guide for full setup instructions.

    Attributes:
        monitor_id (int): The ID of the monitor to use.
        width (int): The width of the SLM in pixels.
        height (int): The height of the SLM in pixels.
    """

    def __init__(
        self, monitor_id: int = -1, host: str | None = None, port: int = 5555
    ) -> None:
        """
        Initialize the SLM instance.

        Args:
            monitor_id (int): The ID of the monitor to use. Defaults to the last monitor.
            host (str | None): If provided, connect to a remote SLMServer at this address
                instead of driving a local display.  Pass ``"localhost"`` to connect to a
                server running on the same machine (e.g. when controlling via SSH).
            port (int): TCP port of the remote SLMServer. Ignored when ``host`` is None.
        """
        if host is not None:
            self._remote = True
            self.monitor_id = monitor_id
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((host, port))
            raw = bytearray(8)
            try:
                _recv_exactly(self._sock, raw, len(raw))
            except (EOFError, OSError) as exc:
                self._sock.close()
                raise ConnectionError("SLMServer handshake failed") from exc
            self.width, self.height = struct.unpack(">II", raw)
            return

        self._remote = False
        if monitor_id in used_ids:
            raise RuntimeError("SLMDisplay instance already exists for this monitor")
        self.monitor_id = monitor_id
        self.window_name = f"SLM Display - Monitor {monitor_id}"
        self.monitor = screeninfo.get_monitors()[monitor_id]
        self.height = self.monitor.height
        self.width = self.monitor.width
        used_ids.append(monitor_id)

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
        if not isinstance(holo, np.ndarray):
            raise TypeError("holo must be a numpy.ndarray")
        if holo.shape != (self.height, self.width):
            raise ValueError(
                f"invalid hologram shape {holo.shape}; "
                f"expected {(self.height, self.width)}"
            )
        if holo.dtype != np.uint8:
            raise TypeError(f"holo must have dtype uint8, got {holo.dtype}")
        if self._remote:
            frame = np.ascontiguousarray(holo)
            self._sock.sendall(frame.tobytes())
            if self._sock.recv(1) != b"K":
                raise ConnectionError("Did not receive ACK from server")
            if sleep_time > 0:
                sleep(sleep_time)
            return

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
        if self._remote:
            try:
                self._sock.close()
            except OSError:
                pass
            return

        if self.monitor_id not in used_ids:
            raise RuntimeError("SLMDisplay instance is already closed")

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
