#!/usr/bin/env python3
"""Phantom zellij client for headless cc-discord-bridge operation.

Allocates a PTY at a sane size and attaches a zellij client to the bridge's
session through it. Stdin/stdout/stderr are routed to /dev/null. The point
is to give the zellij session a real terminal client at a reasonable size
so TUIs spawned inside it (notably Claude Code v2.1.x) actually receive
keystrokes — `action write-chars` is dropped on the floor when zellij has
no client attached, because the TUI is stuck waiting on terminal queries
(DA1/DA2/cursor-position) that nobody answers.

Designed to be launched by launchctl with KeepAlive=true.

Env vars read:
  BRIDGE_ZELLIJ_SESSION   zellij session to attach to (default: meow)
  PHANTOM_COLUMNS         pty width  (default: 200)
  PHANTOM_LINES           pty height (default: 60)
"""
import errno
import fcntl
import os
import pty
import struct
import sys
import termios


def main() -> int:
    session = os.environ.get("BRIDGE_ZELLIJ_SESSION", "meow")
    cols = int(os.environ.get("PHANTOM_COLUMNS", "200"))
    rows = int(os.environ.get("PHANTOM_LINES", "60"))

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    pid = os.fork()
    if pid == 0:
        # Child: become session leader, claim slave as controlling tty, exec zellij.
        os.setsid()
        try:
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        except OSError:
            pass
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.close(master)
        if slave > 2:
            os.close(slave)
        os.environ["TERM"] = os.environ.get("TERM", "xterm-256color")
        os.environ["COLUMNS"] = str(cols)
        os.environ["LINES"] = str(rows)
        os.execvp("zellij", ["zellij", "attach", session])

    # Parent: drain master to /dev/null so the kernel buffer never fills.
    os.close(slave)
    while True:
        try:
            data = os.read(master, 4096)
        except OSError as e:
            if e.errno in (errno.EIO,):
                break
            raise
        if not data:
            break

    os.close(master)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    sys.exit(main())
