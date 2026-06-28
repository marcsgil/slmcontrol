import time
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

        futures = prepare_and_measure(
            prepare, measurements.append, slm, sleep_time=0, nsamples=3
        )

        self.assertEqual(slm.frames, [0, 1, 2])
        self.assertEqual(measurements, [0, 1, 2])
        self.assertTrue(all(future.done() for future in futures))

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
