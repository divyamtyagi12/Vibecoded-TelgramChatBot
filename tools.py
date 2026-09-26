"""Modular, bounded integrations for external information services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Awaitable, Protocol

import httpx


class ExternalTool(Protocol):
    """A capability that can be exposed by a command or a future AI tool call."""

    name: str


class ToolRegistry:
    """Small registry to keep new integrations out of the Telegram adapter."""

    def __init__(self, *tools: ExternalTool) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> ExternalTool | None:
        return self._tools.get(name)


class WeatherTool:
    """Current weather via Open-Meteo; no API key is required for this endpoint."""

    name = "weather"
    _conditions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "rime fog",
        51: "light drizzle",
        53: "drizzle",
        55: "heavy drizzle",
        61: "slight rain",
        63: "rain",
        65: "heavy rain",
        71: "slight snow",
        73: "snow",
        75: "heavy snow",
        80: "rain showers",
        81: "rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
        96: "thunderstorm with hail",
        99: "thunderstorm with heavy hail",
    }

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    async def get_current(self, location: str) -> str:
        location = location.strip()
        if not 2 <= len(location) <= 100:
            raise ValueError("Enter a city or location between 2 and 100 characters.")

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                geocode = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1, "language": "en", "format": "json"},
                )
                geocode.raise_for_status()
                results = geocode.json().get("results", [])
                if not results:
                    raise ValueError("I couldn't find that location. Try a city and country.")
                place = results[0]
                forecast = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": place["latitude"],
                        "longitude": place["longitude"],
                        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                        "timezone": "auto",
                    },
                )
                forecast.raise_for_status()
                current = forecast.json()["current"]
            condition = self._conditions.get(current.get("weather_code"), "unavailable conditions")
            place_name = ", ".join(
                value for value in (place.get("name"), place.get("country")) if value
            )
            return (
                f"Weather in {place_name}: {condition}, {current['temperature_2m']}°C "
                f"(feels like {current['apparent_temperature']}°C), wind {current['wind_speed_10m']} km/h. "
                f"Observed: {current.get('time', 'current model data')}."
            )
        except ValueError:
            raise
        except (httpx.HTTPError, KeyError, TypeError) as error:
            raise RuntimeError("Weather data is temporarily unavailable.") from error
