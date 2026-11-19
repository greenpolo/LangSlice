"""One lazily made API client, closed when the process exits."""

from __future__ import annotations

import atexit
import logging
from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)


class Closable(Protocol):
    """A client that releases its transport when closed."""

    def close(self) -> None:
        """Release the client's transport."""
        ...


Client = TypeVar("Client", bound=Closable)


class CachedClient(Generic[Client]):
    """Hold one client, made on first use and closed at exit.

    Parameters
    ----------
    make
        Builds the client; its failure (a missing key, say) reaches the
        caller of :meth:`get` and nothing is held.
    name
        What the client is called in the log.
    """

    def __init__(self, make: Callable[[], Client], name: str) -> None:
        self._make = make
        self._name = name
        self._client: Client | None = None
        atexit.register(self.close)

    def get(self) -> Client:
        """Return the client, making it first when none is held.

        Returns
        -------
        Client
            The held client.
        """
        if self._client is None:
            self._client = self._make()
        return self._client

    def close(self) -> None:
        """Close the held client, if any, and let it go; a close that fails is logged."""
        client, self._client = self._client, None
        if client is None:
            return
        try:
            client.close()
        except (OSError, RuntimeError) as exc:
            logger.debug("Failed to close the %s cleanly: %s", self._name, exc)
