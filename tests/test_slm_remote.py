import unittest
import socket
import struct
import threading
import numpy as np
from unittest.mock import MagicMock, patch

from slmcontrol.server import SLMServer, _recv_exactly
from slmcontrol.slm import SLMDisplay

WIDTH, HEIGHT = 64, 32


def _make_mock_slm():
    mock = MagicMock(spec=SLMDisplay)
    mock.width = WIDTH
    mock.height = HEIGHT
    mock.last_frame = None

    def _capture(arr, sleep_time=0):
        mock.last_frame = arr.copy()

    mock.updateArray.side_effect = _capture
    return mock


def _connect_and_handshake(port):
    """Return a connected socket with width, height already received."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    raw = sock.recv(8)
    w, h = struct.unpack(">II", raw)
    return sock, w, h


class SLMServerTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_slm = _make_mock_slm()
        self.server = SLMServer(port=0)
        patcher = patch("slmcontrol.server.SLMDisplay", return_value=self.mock_slm)
        self.mock_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.server.start()
        self.port = self.server._server_sock.getsockname()[1]

    def tearDown(self):
        self.server.stop()

    def test_handshake(self):
        sock, w, h = _connect_and_handshake(self.port)
        self.assertEqual(w, WIDTH)
        self.assertEqual(h, HEIGHT)
        sock.close()

    def test_frame_roundtrip(self):
        sock, w, h = _connect_and_handshake(self.port)
        frame = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        sock.sendall(frame.tobytes())
        ack = sock.recv(1)
        self.assertEqual(ack, b"K")
        np.testing.assert_array_equal(self.mock_slm.last_frame, frame)
        sock.close()

    def test_multiple_frames(self):
        sock, w, h = _connect_and_handshake(self.port)
        for _ in range(3):
            frame = np.random.randint(0, 256, (h, w), dtype=np.uint8)
            sock.sendall(frame.tobytes())
            ack = sock.recv(1)
            self.assertEqual(ack, b"K")
            np.testing.assert_array_equal(self.mock_slm.last_frame, frame)
        sock.close()

    def test_second_client_rejected(self):
        sock1, _, _ = _connect_and_handshake(self.port)
        # Give the server time to register the first client
        import time; time.sleep(0.05)
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock2.connect(("127.0.0.1", self.port))
        # Server closes the second socket without sending a handshake
        data = sock2.recv(8)
        self.assertEqual(data, b"")
        sock1.close()
        sock2.close()

    def test_server_stop_disconnects_client(self):
        sock, _, _ = _connect_and_handshake(self.port)
        self.server.stop()
        # After stop, recv should return empty (connection closed)
        data = sock.recv(1)
        self.assertEqual(data, b"")
        sock.close()


class SLMDisplayRemoteTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_slm = _make_mock_slm()
        self.server = SLMServer(port=0)
        patcher = patch("slmcontrol.server.SLMDisplay", return_value=self.mock_slm)
        self.mock_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.server.start()
        self.port = self.server._server_sock.getsockname()[1]

    def tearDown(self):
        self.server.stop()

    def test_slmdisplay_remote_connects(self):
        slm = SLMDisplay(host="127.0.0.1", port=self.port)
        self.assertEqual(slm.width, WIDTH)
        self.assertEqual(slm.height, HEIGHT)
        self.assertTrue(slm._remote)
        slm.close()

    def test_slmdisplay_remote_updatearray(self):
        slm = SLMDisplay(host="127.0.0.1", port=self.port)
        for _ in range(3):
            frame = np.random.randint(0, 256, (HEIGHT, WIDTH), dtype=np.uint8)
            slm.updateArray(frame, sleep_time=0)
            np.testing.assert_array_equal(self.mock_slm.last_frame, frame)
        slm.close()

    def test_slmdisplay_remote_shape_mismatch(self):
        slm = SLMDisplay(host="127.0.0.1", port=self.port)
        bad_frame = np.random.randint(0, 256, (HEIGHT, WIDTH + 1), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            slm.updateArray(bad_frame)
        slm.close()

    def test_slmdisplay_remote_close_is_idempotent(self):
        slm = SLMDisplay(host="127.0.0.1", port=self.port)
        slm.close()
        # Second close should not raise
        slm.close()

    def test_recv_exactly_helper(self):
        """_recv_exactly raises EOFError when socket closes early."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        def _send_partial():
            conn, _ = server_sock.accept()
            conn.sendall(b"\x01\x02")  # only 2 of 4 expected bytes
            conn.close()

        t = threading.Thread(target=_send_partial, daemon=True)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        buf = bytearray(4)
        with self.assertRaises(EOFError):
            _recv_exactly(client, buf, 4)
        client.close()
        server_sock.close()
        t.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
