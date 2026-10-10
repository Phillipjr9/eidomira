"""Answer one question about a candidate GPU host: can UDP reach it?

The live studio is WebRTC, so the server is one of the peers and the media is SRTP over UDP
(`app/rtc.py`). A host that forwards only TCP completes the signaling and then delivers no
frames, which looks like a hung page rather than a firewall — the most expensive kind of
surprise to hit after paying for hardware. This tool is the minute of checking that comes
before the invoice.

    # on the candidate host
    python -m tools.udp_probe serve --port 34789 --seconds 120

    # from a different network — laptop, phone hotspot, a cheap VPS
    python -m tools.udp_probe send --host <the host> --port 34789

`send` prints the round trip and exits 0 when the echo comes back, 1 when it does not, so it
can be a check in a deployment script. The payload is a nonce of the caller's choosing and the
listener echoes exactly what it receives: nothing here proves anything about the host beyond
whether a datagram can get in and out, and that is the only thing being asked.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

DEFAULT_PORT = 34789
DEFAULT_PAYLOAD = b"eidomira-udp-probe"
DEFAULT_TIMEOUT = 5.0


def serve(port: int, seconds: float, host: str = "0.0.0.0",
          on_ready=None, sock=None) -> int:
    """Echo every datagram back, until `seconds` have passed.

    Returns the number of datagrams echoed. `on_ready(port)` is called once the socket is
    bound — used by the tests, and by anyone wiring this into a script that needs the port the
    OS chose when `port=0`.
    """
    own = sock is None
    listener = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if own:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
    listener.settimeout(0.5)
    bound_port = listener.getsockname()[1]
    if on_ready is not None:
        on_ready(bound_port)

    deadline = time.monotonic() + seconds
    echoed = 0
    try:
        while time.monotonic() < deadline:
            try:
                payload, origin = listener.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            listener.sendto(payload, origin)
            echoed += 1
    finally:
        if own:
            listener.close()
    return echoed


def send(host: str, port: int, payload: bytes = DEFAULT_PAYLOAD,
         timeout: float = DEFAULT_TIMEOUT, sock=None) -> tuple[bool, float]:
    """Send `payload` and wait for it back. Returns (reached, seconds).

    Anything that is not an echo reads as "did not reach": a timeout, a refused write, a
    socket that is already closed. The caller is asking whether this host can carry UDP from
    here, and a raised exception would answer that question with a traceback and a non-zero
    exit code that means something else entirely to a deployment script.
    """
    own = sock is None
    started = time.monotonic()
    probe = sock
    try:
        if probe is None:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(timeout)
        probe.sendto(payload, (host, port))
        reply, _ = probe.recvfrom(65535)
    except (socket.timeout, OSError):
        return False, time.monotonic() - started
    finally:
        if own and probe is not None:
            probe.close()
    return reply == payload, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    verbs = parser.add_subparsers(dest="verb", required=True)

    listener = verbs.add_parser("serve", help="echo datagrams back (run this on the host)")
    listener.add_argument("--port", type=int, default=DEFAULT_PORT)
    listener.add_argument("--seconds", type=float, default=120.0)
    listener.add_argument("--host", default="0.0.0.0")

    caller = verbs.add_parser("send", help="check a host from somewhere else")
    caller.add_argument("--host", required=True)
    caller.add_argument("--port", type=int, default=DEFAULT_PORT)
    caller.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)

    options = parser.parse_args(argv)

    if options.verb == "serve":
        print(f"listening on udp/{options.host}:{options.port} for {options.seconds:.0f}s — "
              f"now run `send --host <this machine> --port {options.port}` from elsewhere")
        echoed = serve(options.port, options.seconds, options.host)
        print(f"echoed {echoed} datagram(s)")
        return 0 if echoed else 1

    reached, elapsed = send(options.host, options.port, timeout=options.timeout)
    if reached:
        print(f"UDP reached {options.host}:{options.port} and came back in {elapsed * 1000:.0f} ms")
        print("this host can carry WebRTC media; check the GPU separately — "
              "GET /api/health reports the engine and whether it is accelerated")
        return 0
    print(f"no reply from {options.host}:{options.port} after {elapsed:.1f}s",
          file=sys.stderr)
    print("either nothing is listening, or inbound UDP is blocked. Either way this host "
          "cannot run the live studio as it stands.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
