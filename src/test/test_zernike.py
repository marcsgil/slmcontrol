import numpy as np
from slmcontrol.zernike import R, zernike, zernike_combination


def test_R():
    for n in range(5):
        for m in range(n, 0, -2):
            if (n-m) % 2 == 0:
                assert np.isclose(R(1, n, m), 1)
            else:
                assert np.isclose(R(1, n, m), 0)
