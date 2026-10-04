"""Thin HTTP client for Open-Meteo with retries and back-off."""

import time

import requests

from src.config import settings


class UpstreamError(RuntimeError):
    """Open-Meteo could not be reached or returned an error."""


def get_json(url, params, retries=None, timeout=None):
    cfg = settings()["api"]
    retries = cfg["max_retries"] if retries is None else retries
    timeout = cfg["timeout_s"] if timeout is None else timeout

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            last_error = UpstreamError(f"HTTP {response.status_code}: {response.text[:200]}")
            # Free-tier minute limits reset quickly; hourly ones need longer waits.
            time.sleep(min(60 * (attempt + 1), 300) if response.status_code == 429 else min(5 * 2 ** attempt, 60))
            continue

        data = response.json()
        if response.status_code >= 400 or data.get("error"):
            raise UpstreamError(data.get("reason", f"HTTP {response.status_code}"))
        return data

    raise UpstreamError(f"Open-Meteo unreachable after {retries} attempts: {last_error}")
