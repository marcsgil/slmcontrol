# Remote Control

## The Problem

When connecting to a lab PC via SSH, the session has no access to the graphical
display.  This means both `screeninfo.get_monitors()` and OpenCV's `cv.imshow()`
fail — the SLM cannot be driven from an SSH session using the standard
[`SLMDisplay`][src.slmcontrol.slm.SLMDisplay] constructor.

## The Solution: Client-Server Architecture

`slmcontrol` solves this with a persistent **server** process that runs on the
SLM machine with display access, and a thin **client** mode built into
`SLMDisplay`.  The server owns the OpenCV window; the client connects over TCP,
sends hologram arrays, and receives an acknowledgement for each frame.

```
SLM machine                         Remote session (SSH)
┌────────────────────────┐          ┌────────────────────────┐
│  slmcontrol-server     │◄─TCP────►│  SLMDisplay(host=...)  │
│  (owns the display)    │          │  (sends holograms)     │
└────────────────────────┘          └────────────────────────┘
```

The hologram computation always happens on whichever machine runs your script —
nothing is outsourced to the server.

## Step 1 — Start the Server

On the **SLM machine**, open a terminal in a desktop session (not via SSH) and
run one of the following, depending on how you manage your Python environment:

=== "uv"

    ```bash
    uv run slmcontrol-server
    ```

=== "pip (venv)"

    ```bash
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    python -m slmcontrol.server
    ```

By default this uses the last monitor (`-1`), listens on port `5555`, and binds
to `127.0.0.1`. This loopback default is appropriate for same-machine clients
and SSH port forwarding.

=== "uv"

    ```bash
    uv run slmcontrol-server --monitor 1 --port 6000
    ```

=== "pip (venv)"

    ```bash
    python -m slmcontrol.server --monitor 1 --port 6000
    ```

The server keeps running until you press `Ctrl+C`.  You will see:

```
2026-06-27 19:00:00,000 INFO SLMServer listening on 127.0.0.1:5555
```

## Step 2 — Connect from your Script

### Same machine via SSH

If your script runs **on the SLM machine** (triggered via SSH), use
`host="localhost"`:

```py
import slmcontrol
import numpy as np

slm = slmcontrol.SLMDisplay(host="localhost")

width, height = slm.width, slm.height
x = np.linspace(-width/2, width/2, width)
y = np.linspace(-height/2, height/2, height)
x, y = np.meshgrid(x, y, sparse=True)

desired  = slmcontrol.lg(x, y, l=1, w=200)
incoming = slmcontrol.lg(x, y, w=500)
holo = slmcontrol.generate_hologram(desired / incoming, 255, 50, 100)

slm.updateArray(holo)
slm.close()
```

Hologram computation happens locally on the SLM machine; the data travels only
through the loopback interface (effectively zero network overhead).

### Fully remote machine

If your script runs on a **different machine**, pass the SLM machine's IP
address (or hostname). Start the server with an explicit network bind and
ensure the port is reachable:

```bash
slmcontrol-server --bind 0.0.0.0
```

```py
slm = slmcontrol.SLMDisplay(host="192.168.1.10", port=5555)
```

Alternatively, use SSH port forwarding so you never need to open a firewall
port. This is the recommended remote configuration and works with the default
loopback bind:

```bash
ssh -L 5555:localhost:5555 user@slm-machine
```

Then connect as if the server were local:

```py
slm = slmcontrol.SLMDisplay(host="localhost", port=5555)
```

## API at a Glance

`SLMDisplay` in remote mode has the same interface as in local mode:

| Method | Behaviour |
|---|---|
| `SLMDisplay(host=..., port=5555)` | Connect to server; receive display dimensions |
| `updateArray(holo, sleep_time=0.15)` | Send frame, wait for ACK, optional sleep |
| `close()` | Close the TCP connection |

The `width` and `height` attributes are set automatically from the server's
display resolution — no need to hard-code them.

## Notes

- Only **one client** can be connected at a time.  A second connection attempt
  while a client is active is rejected immediately.
- If the network connection drops mid-frame, the server exits the current
  session cleanly and waits for the next client.  The SLM keeps displaying the
  last frame it received.
- The `if __name__ == '__main__':` guard is **not** required for client scripts
  (only local-mode scripts that spawn a display process need it).
