"""Cancellation token for cooperative async operation cancellation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


@dataclass
class CancellationToken:
    """Token for cooperative cancellation of async operations.

    This class provides a way to cancel async operations cooperatively.
    The token tracks a cancellation state and can wrap coroutines in
    cancellable tasks.

    Usage:
        token = CancellationToken()

        # In agent loop:
        try:
            result = await token.run(api.send(dag))
        except asyncio.CancelledError:
            # Handle cancellation
            pass

        # To cancel (from another coroutine/callback):
        token.cancel()

        # To reuse for new operation:
        token.reset()

    Integration with executor.run():
        from nano_agent import run, CancellationToken

        token = CancellationToken()
        # In another task: token.cancel()
        dag = await run(api, dag, cancel_token=token)
    """

    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _current_task: asyncio.Task[Any] | None = field(default=None, init=False)

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation - sets flag and cancels current task."""
        self._event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def reset(self) -> None:
        """Reset for reuse with new operation."""
        self._event.clear()
        self._current_task = None

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine with cancellation support.

        Wraps coroutine in asyncio.Task so it can be cancelled mid-flight.
        Raises asyncio.CancelledError if cancel() was called.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            asyncio.CancelledError: If cancel() was called during execution
        """
        if self.is_cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

        task: asyncio.Task[T] = asyncio.create_task(coro)
        self._current_task = task
        try:
            return await task
        finally:
            self._current_task = None
