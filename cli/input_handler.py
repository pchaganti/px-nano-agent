"""Raw terminal input handling for escape key detection.

This module provides a simple InputHandler that:
- Puts the terminal in cbreak mode during async operations
- Detects the Escape key (distinguishing it from arrow key sequences)
- Restores terminal state reliably on exit

The handler is designed to be used as an async context manager during
agent execution, allowing users to press Escape to cancel operations.

Example:
    handler = InputHandler(on_escape=lambda: cancel_token.cancel())
    
    async with handler:
        # During this block, Escape key is detected
        await some_long_operation()
"""

from __future__ import annotations

import asyncio
import select
import sys
from dataclasses import dataclass, field
from typing import Callable

# Only import termios/tty on Unix systems
try:
    import termios
    import tty

    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

# Timeout to distinguish standalone Escape from escape sequences (arrow keys, etc.)
# Arrow keys send: ESC [ A/B/C/D - we wait this long after ESC to see if more comes
ESCAPE_SEQUENCE_TIMEOUT = 0.025  # 25ms


@dataclass
class InputHandler:
    """Handles raw terminal input for escape key detection.

    This class provides escape key detection during async operations.
    It uses terminal cbreak mode to read individual keypresses without
    waiting for Enter.

    The handler distinguishes between:
    - Standalone Escape key (triggers on_escape callback)
    - Escape sequences like arrow keys (ignored)

    Usage:
        handler = InputHandler(on_escape=my_callback)

        async with handler:
            # Escape key detection active
            await long_running_operation()
        # Terminal restored to normal

    Attributes:
        on_escape: Callback invoked when Escape key is pressed.
                   Should be a simple, fast function (e.g., setting a flag).
    """

    on_escape: Callable[[], None] | None = None

    _original_termios: list | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _paused: bool = field(default=False, init=False, repr=False)  # Pause listener for prompts

    def _is_tty(self) -> bool:
        """Check if stdin is a real terminal."""
        try:
            return sys.stdin.isatty()
        except Exception:
            return False

    def _enter_cbreak_mode(self) -> bool:
        """Put terminal in cbreak mode, saving original settings.

        Cbreak mode allows reading individual characters without waiting
        for Enter, while still allowing Ctrl+C for emergency exit.

        Returns:
            True if successful, False if not possible (not a TTY, etc.)
        """
        if not HAS_TERMIOS or not self._is_tty():
            return False

        try:
            fd = sys.stdin.fileno()
            self._original_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            return True
        except (termios.error, OSError):
            return False

    def _restore_terminal(self) -> None:
        """Restore original terminal settings.

        Safe to call multiple times or when no settings were saved.
        """
        if not HAS_TERMIOS or self._original_termios is None:
            return

        try:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, self._original_termios)
        except (termios.error, OSError):
            pass  # Best effort - terminal might be gone
        finally:
            self._original_termios = None

    async def _listener_loop(self) -> None:
        """Background task that listens for Escape key.

        Uses select() to poll stdin without blocking the event loop.
        Distinguishes Escape from escape sequences by timing.
        """
        fd = sys.stdin.fileno()

        while self._running:
            # If paused (for prompt_yn), just sleep and continue
            if self._paused:
                await asyncio.sleep(0.01)
                continue

            # Poll stdin with short timeout (allows clean shutdown)
            try:
                readable, _, _ = select.select([fd], [], [], 0.05)
            except (ValueError, OSError):
                # fd closed or invalid
                break

            if not readable:
                # No input, yield control to event loop
                await asyncio.sleep(0.01)
                continue

            # Read the character
            try:
                char = sys.stdin.read(1)
            except (OSError, IOError):
                break

            if char == "\x1b":  # ESC character
                # Wait briefly to see if this is an escape sequence
                try:
                    more_readable, _, _ = select.select(
                        [fd], [], [], ESCAPE_SEQUENCE_TIMEOUT
                    )
                except (ValueError, OSError):
                    break

                if more_readable:
                    # More characters coming - it's an escape sequence (arrow key, etc.)
                    # Consume the rest of the sequence
                    try:
                        next_char = sys.stdin.read(1)
                        if next_char == "[":
                            # CSI sequence (arrows, function keys)
                            # Read until we get a letter (the final character)
                            while True:
                                seq_char = sys.stdin.read(1)
                                if seq_char.isalpha() or seq_char == "~":
                                    break
                    except (OSError, IOError):
                        pass
                    continue

                # No more characters - it's a standalone Escape
                if self.on_escape:
                    self.on_escape()

            # Yield control back to event loop
            await asyncio.sleep(0)

    async def prompt_yn(self, prompt: str = "Continue? [y/n/Esc]: ") -> bool | None:
        """Prompt for yes/no/escape input using single character read.

        This method temporarily pauses the escape listener and reads a single
        character from stdin. It's designed to be used for permission prompts
        during agent execution.

        Args:
            prompt: The prompt message to display (caller should print this).
                   This is just for documentation - caller handles display.

        Returns:
            True if user pressed 'y' or 'Y'
            False if user pressed 'n' or 'N'
            None if user pressed Escape (caller should handle as cancel)

        Raises:
            RuntimeError: If called when handler is not running or no TTY.

        Note:
            The caller is responsible for printing the prompt before calling
            this method, and for handling the None (escape) case appropriately.
        """
        if not self._running:
            raise RuntimeError("InputHandler is not running")

        if not HAS_TERMIOS or not self._is_tty():
            # Fallback to regular input for non-TTY
            try:
                response = input(prompt).strip().lower()
                return response in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                return None

        # Pause the listener loop so we can read stdin directly
        self._paused = True
        try:
            fd = sys.stdin.fileno()

            while True:
                # Wait for input with timeout (allows cancellation check)
                try:
                    readable, _, _ = select.select([fd], [], [], 0.1)
                except (ValueError, OSError):
                    return None

                if not readable:
                    # Yield to event loop
                    await asyncio.sleep(0.01)
                    continue

                # Read single character
                try:
                    char = sys.stdin.read(1)
                except (OSError, IOError):
                    return None

                # Check for Escape
                if char == "\x1b":
                    # Wait briefly to distinguish from escape sequences
                    try:
                        more, _, _ = select.select([fd], [], [], ESCAPE_SEQUENCE_TIMEOUT)
                    except (ValueError, OSError):
                        return None

                    if more:
                        # It's an escape sequence (arrow key, etc.) - consume and ignore
                        try:
                            next_char = sys.stdin.read(1)
                            if next_char == "[":
                                while True:
                                    seq_char = sys.stdin.read(1)
                                    if seq_char.isalpha() or seq_char == "~":
                                        break
                        except (OSError, IOError):
                            pass
                        continue  # Keep waiting for valid input

                    # Standalone Escape - return None to indicate cancel
                    return None

                # Check for y/n
                if char.lower() == "y":
                    return True
                elif char.lower() == "n":
                    return False

                # Ignore other characters, keep waiting
        finally:
            self._paused = False

    async def start(self) -> bool:
        """Start listening for escape key.

        Returns:
            True if started successfully, False if not possible.
        """
        if self._running:
            return True

        if not self._enter_cbreak_mode():
            return False

        self._running = True
        self._task = asyncio.create_task(self._listener_loop())
        return True

    async def stop(self) -> None:
        """Stop listening and restore terminal."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._restore_terminal()

    async def __aenter__(self) -> "InputHandler":
        """Async context manager entry - starts listening."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - always restores terminal."""
        await self.stop()
        # Don't suppress exceptions
        return None
