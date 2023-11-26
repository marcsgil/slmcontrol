assert b(0, 0, 0) == 1

assert np.allclose(b(1, 0, 0), 1/np.sqrt(2))
assert np.allclose(b(0, 1, 0), 1/np.sqrt(2))
assert np.allclose(b(1, 0, 1), 1/np.sqrt(2))
assert np.allclose(b(0, 1, 1), -1/np.sqrt(2))


assert np.allclose(b(0, 2, 0), 1/2)
assert np.allclose(b(0, 2, 1), -1/np.sqrt(2))
assert np.allclose(b(0, 2, 2), 1/2)
assert np.allclose(b(1, 1, 0), 1/np.sqrt(2))
assert np.allclose(b(1, 1, 1), 0)
assert np.allclose(b(1, 1, 2), -1/np.sqrt(2))
assert np.allclose(b(2, 0, 0), 1/2)
assert np.allclose(b(2, 0, 1), 1/np.sqrt(2))
assert np.allclose(b(2, 0, 2), 1/2)
