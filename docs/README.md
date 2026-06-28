# slmcontrol

`slmcontrol` generates structured-light holograms and displays them on a
Spatial Light Modulator (SLM). It supports direct local display control and a
TCP client/server mode for laboratory computers controlled through SSH.

Python 3.11 or newer is required. Windows, macOS, and Linux are supported,
subject to OpenCV and monitor-driver availability.

## Installation

```bash
pip install slmcontrol
```

## Minimal example

```python
import numpy as np
import slmcontrol

if __name__ == "__main__":
    slm = slmcontrol.SLMDisplay()
    x = np.arange(slm.width) - slm.width / 2
    y = np.arange(slm.height) - slm.height / 2
    x, y = np.meshgrid(x, y, sparse=True)

    field = slmcontrol.lg(x, y, l=1, w=200)
    hologram = slmcontrol.generate_hologram(field, 255, 50, 100)
    slm.updateArray(hologram)
    slm.close()
```

See the [quickstart](quickstart.md), [remote-control guide](remote.md), and
[API reference](reference.md) for complete usage.

## Links

- [PyPI](https://pypi.org/project/slmcontrol/)
- [Documentation](https://marcsgil.github.io/slmcontrol/)
- [Source code](https://github.com/marcsgil/slmcontrol)
- [Issue tracker](https://github.com/marcsgil/slmcontrol/issues)
