import asyncio
from collections import deque
from typing import AsyncIterator, Deque, Generic, TypeVar

T = TypeVar("T")


class AsyncDeque(Generic[T]):
    """An asynchronous double-ended queue."""

    def __init__(self, timeout=None) -> None:
        self._deque: Deque[T] = deque()
        self._condition = asyncio.Condition()
        self._terminated = False
        self._timeout = timeout

    async def put_left(self, item: T) -> None:
        """Add an item to the left end of the queue."""
        async with self._condition:
            self._deque.appendleft(item)
            self._terminated = False
            self._condition.notify()

    async def put_right(self, item: T) -> None:
        """Add an item to the right end of the queue."""
        async with self._condition:
            self._deque.append(item)
            self._terminated = False
            self._condition.notify()

    async def terminate(self) -> None:
        async with self._condition:
            self._deque.clear()
            self._terminated = True
            self._condition.notify_all()

    async def get_left(self) -> T:
        """Remove and return an item from the left end of the queue.

        Blocks if the queue is empty.
        """
        async with self._condition:
            while not self._deque:
                if self._terminated:
                    raise IndexError("deque is empty")
                try:
                    await asyncio.wait_for(
                        self._condition.wait(), timeout=self._timeout
                    )
                except asyncio.TimeoutError:
                    raise IndexError("deque is empty")
            return self._deque.popleft()

    async def get_right(self) -> T:
        """Remove and return an item from the right end of the queue.

        Blocks if the queue is empty.
        """
        async with self._condition:
            while not self._deque:
                if self._terminated:
                    raise IndexError("deque is empty")
                try:
                    await asyncio.wait_for(
                        self._condition.wait(), timeout=self._timeout
                    )
                except asyncio.TimeoutError:
                    raise IndexError("deque is empty")
            return self._deque.pop()

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.get_left()
        except IndexError:
            raise StopAsyncIteration
