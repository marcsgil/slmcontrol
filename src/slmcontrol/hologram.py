from juliacall import Main as jl
jl.seval("using StructuredLight")
print(jl.Threads.nthreads())


def generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period, method='BesselJ1'):
    return jl.generate_hologram(desired, incoming, x, y,
                                max_modulation, x_period, y_period, jl.BesselJ1)
