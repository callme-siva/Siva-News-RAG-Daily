"""Polite HTTP for all source adapters (REQUIREMENTS R7, FR1).

One `HttpFetcher` per fetch run: shared client, a clear User-Agent, robots.txt checks
(RFC 9309 semantics), per-host minimum spacing between requests, and retries with
backoff on network errors, 429 and 5xx responses.
"""

from __future__ import annotations

import asyncio
import time
import urllib.robotparser
from collections.abc import Mapping
from types import TracebackType
from urllib.parse import urlsplit

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from newsrag import __version__

USER_AGENT = f"newsrag/{__version__} (+personal news reader)"
ROBOTS_AGENT = "newsrag"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class BlockedByRobots(Exception):
    """The site's robots.txt disallows this URL for our user agent."""


class RetryableStatus(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, RetryableStatus | httpx.TransportError)


class HttpFetcher:
    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        attempts: int = 3,
        backoff_s: float = 2.0,
        min_interval_s: Mapping[str, float] | None = None,
        respect_robots: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            transport=transport,
        )
        self._attempts = attempts
        self._backoff = backoff_s
        self._min_interval = dict(min_interval_s or {})
        self._respect_robots = respect_robots
        self._robots: dict[str, urllib.robotparser.RobotFileParser | bool] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    async def __aenter__(self) -> HttpFetcher:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def allowed(self, url: str) -> bool:
        """robots.txt check. 4xx = allow all, 5xx or unreachable = disallow (RFC 9309)."""
        if not self._respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._robots_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._robots:
                self._robots[origin] = await self._load_robots(origin)
        rules = self._robots[origin]
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(ROBOTS_AGENT, url)

    async def _load_robots(self, origin: str) -> urllib.robotparser.RobotFileParser | bool:
        try:
            resp = await self._client.get(f"{origin}/robots.txt")
        except httpx.HTTPError:
            return False
        if 400 <= resp.status_code < 500:
            return True
        if resp.status_code >= 500:
            return False
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser

    async def _space(self, host: str) -> None:
        interval = self._min_interval.get(host, 0.0)
        if interval <= 0:
            return
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last_request.get(host, 0.0) + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request[host] = time.monotonic()

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET with robots check, host spacing and retries. Raises on final failure."""
        if not await self.allowed(url):
            raise BlockedByRobots(
                f"robots.txt disallows {urlsplit(url).netloc}{urlsplit(url).path}"
            )
        host = urlsplit(url).netloc
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._attempts),
            wait=wait_exponential(
                multiplier=self._backoff, min=self._backoff, max=self._backoff * 8
            ),
            retry=retry_if_exception(_retryable),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await self._space(host)
                resp = await self._client.get(url, params=params, headers=headers)
                if resp.status_code in RETRY_STATUSES:
                    raise RetryableStatus(resp)
                resp.raise_for_status()
                return resp
        raise AssertionError("unreachable")  # pragma: no cover
