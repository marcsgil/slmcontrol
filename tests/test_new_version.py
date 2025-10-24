import slmcontrol.structures_old as structures_old
import slmcontrol
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 128)
y = np.linspace(-5, 5, 128)

X, Y = np.meshgrid(x, y, sparse=True)

for p in range(3):
    for L in range(-3, 2):
        for w in [1, 2, 3]:
            print(f"Testing LG mode with p={p}, L={L}, w={w}")
            u1 = slmcontrol.lg(X, Y, p=p, L=L, w=w)
            u1 /= np.max(np.abs(u1))
            u2 = structures_old.lg(x, y, p=p, l=L, w=w)
            u2 /= np.max(np.abs(u2))

            np.testing.assert_allclose(np.abs(u1), np.abs(u2))

for m in range(3):
    for n in range(3):
        for w in [1, 2, 3]:
            print(f"Testing HG mode with m={m}, n={n}, w={w}")
            u1 = slmcontrol.hg(X, Y, m=m, n=n, w=w)
            u1 /= np.max(np.abs(u1))
            u2 = structures_old.hg(x, y, m=m, n=n, w=w)
            u2 /= np.max(np.abs(u2))

            np.testing.assert_allclose(np.abs(u1), np.abs(u2))

            print(f"Testing diagonal HG mode with m={m}, n={n}, w={w}")
            u1 = slmcontrol.diagonal_hg(X, Y, m=m, n=n, w=w)
            u1 /= np.max(np.abs(u1))
            u2 = structures_old.diagonal_hg(x, y, m=m, n=n, w=w)
            u2 /= np.max(np.abs(u2))

            np.testing.assert_allclose(np.abs(u1), np.abs(u2), atol=1e-5)


for a in [2, 4, 6]:
    print(f"Testing square aperture with L={a}")
    mask1 = slmcontrol.square(X, Y, L=a)
    mask2 = structures_old.square(x, y, l=a)
    np.testing.assert_array_equal(mask1, mask2)

    print(f"Testing single slit with a={a}")
    mask1 = slmcontrol.single_slit(X, Y, a=a)
    mask2 = structures_old.single_slit(x, y, a=a)
    np.testing.assert_array_equal(mask1, mask2)

    print(f"Testing circular aperture with a={a}")
    mask1 = slmcontrol.pupil(X, Y, radius=a)
    mask2 = structures_old.pupil(x, y, radius=a)
    np.testing.assert_array_equal(mask1, mask2)

    print(f"Testing triangular aperture with a={a}")
    mask1 = slmcontrol.triangle(X, Y, side_length=a)
    mask2 = structures_old.triangle(x, y, side_length=a)
    np.testing.assert_array_equal(mask1, mask2)

    for b in [2, 4, 6]:
        print(f"Testing rectangular aperture with a={a}, b={b}")
        mask1 = slmcontrol.rectangular_aperture(X, Y, a=a, b=b)
        mask2 = structures_old.rectangular_aperture(x, y, a=a, b=b)
        np.testing.assert_array_equal(mask1, mask2)

        print(f"Testing double slit with a={a}, d={b}")
        mask1 = slmcontrol.double_slit(X, Y, a=a, d=b)
        mask2 = structures_old.double_slit(x, y, a=a, d=b)
        np.testing.assert_array_equal(mask1, mask2)

print("All testing complete. No discrepancies found.")
