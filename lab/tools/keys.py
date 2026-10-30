"""Recording controls read from the terminal, as a drop-in for LeRobot's pynput listener.

pynput needs X11 focus, so it misses keys on Wayland desktops and over SSH. This reads the
controlling terminal instead, which works on both (and in tmux). Same events as upstream:

    ->  or n   end the episode (or skip the reset) and continue
    <-  or r   re-record the current episode
    Esc or q   stop recording (saves what is done)
"""

from __future__ import annotations

import atexit
import os
import select
import sys
import termios
import threading
import tty

KEYS = {
    "\x1b[C": "next", "n": "next",
    "\x1b[D": "rerecord", "r": "rerecord",
    "\x1b": "stop", "q": "stop",
}
LEGEND = "keys: [-> / n] next   [<- / r] re-record   [Esc / q] stop   (keep this terminal focused)"


class TerminalKeyListener:
    def __init__(self, events: dict[str, bool]):
        self.events = events
        self.fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self.fd)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="terminal_keys", daemon=True)
        atexit.register(self._restore)  # never leave the terminal in cbreak mode

    def start(self) -> None:
        tty.setcbreak(self.fd)  # unbuffered, no echo; Ctrl+C still works
        self._thread.start()
        print(LEGEND, flush=True)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._restore()

    def _restore(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not select.select([self.fd], [], [], 0.1)[0]:
                continue
            action = KEYS.get(os.read(self.fd, 16).decode(errors="ignore"))
            if action == "next":
                print("\n[next] ending episode / skipping reset", flush=True)
                self.events["exit_early"] = True
            elif action == "rerecord":
                print("\n[re-record] discarding this episode", flush=True)
                self.events["rerecord_episode"] = True
                self.events["exit_early"] = True
            elif action == "stop":
                print("\n[stop] finishing up and saving", flush=True)
                self.events["stop_recording"] = True
                self.events["exit_early"] = True


def init_keyboard_listener():
    """Same contract as lerobot.utils.control_utils.init_keyboard_listener: (listener, events)."""
    if not sys.stdin.isatty():
        from lerobot.utils.control_utils import init_keyboard_listener as upstream
        return upstream()
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    listener = TerminalKeyListener(events)
    listener.start()
    return listener, events
