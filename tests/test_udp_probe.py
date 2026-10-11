"""The probe that decides whether a host can run the live studio.

It exists because the failure it detects is invisible until after the invoice: a TCP-only host
completes the WebRTC signaling handshake and then delivers no media, so the page looks hung
rather than firewalled. These tests drive both halves over loopback, where the answer is known,
and assert the negative case too — a probe that says "reachable" without a listener is worse
than no probe at all.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

from tools import udp_probe

PAYLOAD = b"a-nonce-only-this-test-knows"


@pytest.fixture
def echoing_host():
    """A listener on a port the OS picks, running until the test is done with it.

    The socket is owned here and closed on teardown, which is what stops the loop: `serve`
    treats a closed socket as the end of the run, so the fixture costs nothing and the suite
    does not wait out a timeout it does not need.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    thread = threading.Thread(
        target=lambda: udp_probe.serve(0, seconds=30.0, host="127.0.0.1", sock=listener),
        daemon=True)
    thread.start()
    try:
        yield port
    finally:
        listener.close()
        thread.join(timeout=2.0)


def test_a_datagram_reaches_a_listening_host_and_comes_back(echoing_host):
    reached, elapsed = udp_probe.send("127.0.0.1", echoing_host, payload=PAYLOAD)
    assert reached, "the echo did not come back over loopback"
    assert elapsed < 5.0


def test_the_listener_echoes_exactly_what_it_received(echoing_host):
    """Not a handshake and not a canned reply: the caller's own nonce, so a middlebox that
    answers on the host's behalf cannot make a dead host look alive."""
    reached, _ = udp_probe.send("127.0.0.1", echoing_host, payload=b"something-else")
    assert reached
    reached, _ = udp_probe.send("127.0.0.1", echoing_host, payload=PAYLOAD * 3)
    assert reached, "the listener is not echoing what it actually received"


def test_a_port_with_nothing_listening_reports_unreachable():
    """The negative case, and the one that matters. A port is bound and closed so the OS is
    certain not to hand the same one out mid-test."""
    occupied = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    occupied.bind(("127.0.0.1", 0))
    port = occupied.getsockname()[1]
    occupied.close()

    reached, elapsed = udp_probe.send("127.0.0.1", port, payload=PAYLOAD, timeout=0.5)
    assert not reached, "an unreachable host was reported as reachable"
    assert elapsed >= 0.4, "it did not actually wait for a reply"


def test_an_error_does_not_look_like_a_reply():
    """A closed socket raises rather than times out on some platforms; both must read as
    'no answer', never as success."""
    closed = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    closed.close()
    reached, _ = udp_probe.send("127.0.0.1", 9, payload=PAYLOAD, timeout=0.5, sock=closed)
    assert reached is False


def test_serve_returns_when_its_time_runs_out(echoing_host):
    """`serve` has to be usable as a bounded check in a script, not only as a daemon."""
    echoed = udp_probe.serve(0, seconds=0.3, host="127.0.0.1")
    assert echoed == 0, "it echoed a datagram nobody sent"


def test_send_reports_its_result_in_the_exit_code(echoing_host, capsys):
    assert udp_probe.main(["send", "--host", "127.0.0.1", "--port", str(echoing_host),
                           "--timeout", "2"]) == 0
    printed = capsys.readouterr().out
    assert "UDP reached" in printed
    assert "ms" in printed, "no round-trip time was reported"

    occupied = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    occupied.bind(("127.0.0.1", 0))
    dead = occupied.getsockname()[1]
    occupied.close()
    assert udp_probe.main(["send", "--host", "127.0.0.1", "--port", str(dead),
                           "--timeout", "0.5"]) == 1
    assert "no reply" in capsys.readouterr().err
