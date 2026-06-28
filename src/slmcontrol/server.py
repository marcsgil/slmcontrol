import socket
import struct
import threading
import logging
import signal
import numpy as np
from slmcontrol.slm import SLMDisplay

logger = logging.getLogger(__name__)


def _recv_exactly(sock: socket.socket, buf, n: int) -> None:
    """
    Receive exactly n bytes from sock into buf (a writable bytes-like object).
    Raises EOFError if the socket closes before n bytes arrive.
    """
    view = memoryview(buf)
    received = 0
    while received < n:
        chunk = sock.recv_into(view[received:], n - received)
        if chunk == 0:
            raise EOFError(f"Socket closed after {received}/{n} bytes")
        received += chunk


class SLMServer:
    """
    Wraps a local SLMDisplay and serves frame updates to a single remote client over TCP.

    Start the server on the SLM machine before connecting remotely:

    Example:
        ```python
        server = SLMServer(monitor_id=-1, port=5555)
        server.start()
        # ... server runs until server.stop() is called
        ```

    Or via the command line (uv)::

        uv run slmcontrol-server --monitor -1 --port 5555

    Or with pip inside an activated virtual environment::

        python -m slmcontrol.server --monitor -1 --port 5555

    Wire protocol (all integers big-endian):
        Handshake (server → client, on connect): 8 bytes: uint32 width + uint32 height
        Frame (client → server): width * height bytes (uint8, row-major C order)
        ACK (server → client): 1 byte b'K'
        Close: client closes the TCP connection

    Attributes:
        monitor_id (int): Monitor index passed to the local SLMDisplay.
        port (int): TCP port the server listens on.
    """

    def __init__(
        self,
        monitor_id: int = -1,
        port: int = 5555,
        bind_address: str = "127.0.0.1",
    ) -> None:
        self.monitor_id = monitor_id
        self.port = port
        self.bind_address = bind_address
        self._slm = None
        self._server_sock = None
        self._client_conn = None
        self._client_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._serve_thread = None

    def start(self) -> None:
        """
        Initialise the local SLMDisplay, bind the TCP socket, and begin
        accepting connections in a daemon thread.
        """
        if self._running:
            raise RuntimeError("SLMServer is already running")
        self._slm = SLMDisplay(monitor_id=self.monitor_id)
        try:
            self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_sock.bind((self.bind_address, self.port))
            self._server_sock.listen(1)
        except Exception:
            if self._server_sock is not None:
                self._server_sock.close()
                self._server_sock = None
            self._slm.close()
            self._slm = None
            raise
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info(
            "SLMServer listening on %s:%d", self.bind_address, self.port
        )

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
            except OSError:
                break
            with self._client_lock:
                if self._client_conn is not None:
                    logger.warning("Rejected second client from %s", addr)
                    conn.close()
                    continue
                self._client_conn = conn
            logger.info("Client connected from %s", addr)
            self._serve_thread = threading.Thread(
                target=self._serve_client_and_cleanup, args=(conn,), daemon=True
            )
            self._serve_thread.start()

    def _serve_client_and_cleanup(self, conn: socket.socket) -> None:
        self._serve_client(conn)
        with self._client_lock:
            self._client_conn = None
        logger.info("Client disconnected")

    def _serve_client(self, conn: socket.socket) -> None:
        h, w = self._slm.height, self._slm.width
        frame_bytes = w * h
        recv_buf = np.empty(frame_bytes, dtype=np.uint8)  # 1D for correct memoryview slicing
        try:
            conn.sendall(struct.pack(">II", w, h))
        except OSError as e:
            logger.error("Handshake failed: %s", e)
            conn.close()
            return
        while self._running:
            try:
                _recv_exactly(conn, recv_buf.data, frame_bytes)
            except (EOFError, OSError):
                break
            self._slm.updateArray(recv_buf.reshape(h, w), sleep_time=0)
            try:
                conn.sendall(b"K")
            except OSError:
                break
        try:
            conn.close()
        except OSError:
            pass

    def stop(self) -> None:
        """
        Shut down the server and close the local SLMDisplay.
        """
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        with self._client_lock:
            if self._client_conn:
                try:
                    self._client_conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    self._client_conn.close()
                except OSError:
                    pass
        if self._thread:
            self._thread.join(timeout=5)
        if self._serve_thread:
            self._serve_thread.join(timeout=5)
        if self._slm:
            self._slm.close()
            self._slm = None
        logger.info("SLMServer stopped")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="slmcontrol remote display server")
    parser.add_argument(
        "--monitor", type=int, default=-1, help="Monitor index (default: -1)"
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="TCP port to listen on (default: 5555)"
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Address to bind (default: 127.0.0.1; use 0.0.0.0 for remote clients)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    server = SLMServer(
        monitor_id=args.monitor, port=args.port, bind_address=args.bind
    )
    server.start()

    def _shutdown(signum, frame):
        logger.info("Signal received, shutting down")
        server.stop()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    server._thread.join()


if __name__ == "__main__":
    _main()
