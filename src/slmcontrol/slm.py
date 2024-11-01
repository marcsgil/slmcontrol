from juliacall import Main as jl
jl.seval("using SpatialLightModulator")

class SLM:
    def __init__(self, monitor_id = jl.lastindex(jl.GetMonitors()) - 1) -> None:
        self.monitor_id = monitor_id
        self.slm = jl.SLM(monitor_id+1)
        self.width = self.slm.width
        self.height = self.slm.height
        self.refreshrate = self.slm.refreshrate

    def update_hologram(self, holo) -> None:
        jl.update_hologram(self.slm, holo)

    def close(self) -> None:
        jl.close(self.slm)