import time
import threading
import unittest

import numpy as np

from slmcontrol.prepare_and_measure import prepare_and_measure


class FakeSLM:
    def __init__(self):
        self.frames = []

    def updateArray(self, holo, sleep_time):
        self.frames.append(int(holo[0, 0]))


class PrepareAndMeasureTestCase(unittest.TestCase):
    def test_frames_and_measurements_remain_in_order(self):
        slm = FakeSLM()
        measurements = []

        def prepare(n):
            if n == 0:
                time.sleep(0.03)
            return np.full((1, 1), n, dtype=np.uint8)

        result = prepare_and_measure(
            prepare, measurements.append, slm, sleep_time=0, nsamples=3
        )

        self.assertIsNone(result)
        self.assertEqual(slm.frames, [0, 1, 2])
        self.assertEqual(measurements, [0, 1, 2])

    def test_preparation_window_is_limited_to_two_frames(self):
        slm = FakeSLM()
        two_preparations_started = threading.Event()
        release_preparations = threading.Event()
        started = []
        started_lock = threading.Lock()
        errors = []

        def prepare(n):
            with started_lock:
                started.append(n)
                if len(started) == 2:
                    two_preparations_started.set()
            if n < 2:
                release_preparations.wait(timeout=2)
            return np.full((1, 1), n, dtype=np.uint8)

        def acquire():
            try:
                prepare_and_measure(prepare, lambda n: None, slm, 0, 4)
            except Exception as error:  # pragma: no cover - assertion below
                errors.append(error)

        acquisition = threading.Thread(target=acquire)
        acquisition.start()
        try:
            self.assertTrue(two_preparations_started.wait(timeout=1))
            time.sleep(0.05)
            with started_lock:
                self.assertEqual(set(started), {0, 1})
        finally:
            release_preparations.set()
            acquisition.join(timeout=2)

        self.assertFalse(acquisition.is_alive())
        self.assertEqual(errors, [])

    def test_prepare_failure_does_not_wait_for_running_preparation(self):
        second_prepare_started = threading.Event()
        release_second_prepare = threading.Event()

        def prepare(n):
            if n == 0:
                self.assertTrue(second_prepare_started.wait(timeout=1))
                raise LookupError("prepare failed")
            second_prepare_started.set()
            release_second_prepare.wait(timeout=2)
            return np.zeros((1, 1), dtype=np.uint8)

        try:
            started_at = time.monotonic()
            with self.assertRaisesRegex(LookupError, "prepare failed"):
                prepare_and_measure(prepare, lambda n: None, FakeSLM(), 0, 3)
            self.assertLess(time.monotonic() - started_at, 0.5)
        finally:
            release_second_prepare.set()

    def test_measure_failure_does_not_wait_for_running_preparation(self):
        second_prepare_started = threading.Event()
        release_second_prepare = threading.Event()

        def prepare(n):
            if n == 1:
                second_prepare_started.set()
                release_second_prepare.wait(timeout=2)
            return np.zeros((1, 1), dtype=np.uint8)

        def measure(n):
            self.assertTrue(second_prepare_started.wait(timeout=1))
            raise LookupError("measure failed")

        try:
            started_at = time.monotonic()
            with self.assertRaisesRegex(LookupError, "measure failed"):
                prepare_and_measure(prepare, measure, FakeSLM(), 0, 3)
            self.assertLess(time.monotonic() - started_at, 0.5)
        finally:
            release_second_prepare.set()

    def test_update_failure_does_not_wait_for_running_preparation(self):
        second_prepare_started = threading.Event()
        release_second_prepare = threading.Event()
        test_case = self

        class FailingSLM:
            def updateArray(self, holo, sleep_time):
                test_case.assertTrue(second_prepare_started.wait(timeout=1))
                raise LookupError("update failed")

        def prepare(n):
            if n == 1:
                second_prepare_started.set()
                release_second_prepare.wait(timeout=2)
            return np.zeros((1, 1), dtype=np.uint8)

        try:
            started_at = time.monotonic()
            with self.assertRaisesRegex(LookupError, "update failed"):
                prepare_and_measure(prepare, lambda n: None, FailingSLM(), 0, 3)
            self.assertLess(time.monotonic() - started_at, 0.5)
        finally:
            release_second_prepare.set()

    def test_prepare_exception_is_propagated(self):
        def prepare(n):
            if n == 1:
                raise LookupError("prepare failed")
            return np.full((1, 1), n, dtype=np.uint8)

        with self.assertRaisesRegex(LookupError, "prepare failed"):
            prepare_and_measure(prepare, lambda n: None, FakeSLM(), 0, 3)

    def test_measure_exception_is_propagated(self):
        def measure(n):
            raise LookupError("measure failed")

        with self.assertRaisesRegex(LookupError, "measure failed"):
            prepare_and_measure(
                lambda n: np.zeros((1, 1), dtype=np.uint8),
                measure,
                FakeSLM(),
                0,
                2,
            )

    def test_invalid_timing_arguments(self):
        with self.assertRaises(ValueError):
            prepare_and_measure(lambda n: n, lambda n: None, FakeSLM(), 0, -1)
        with self.assertRaises(ValueError):
            prepare_and_measure(lambda n: n, lambda n: None, FakeSLM(), -1, 1)
