"""Real-time web search and live data retrieval utility."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from lxml import html

logger = logging.getLogger(__name__)

try:
    import primp
    _HAS_PRIMP = True
except ImportError:
    _HAS_PRIMP = False

import httpx

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class WebSearchClient:
    """Client for real-time web search via DuckDuckGo Lite and direct webpage inspection."""

    def __init__(self, timeout_seconds: float = 6.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int = 5) -> str:
        """Perform a web search and return structured snippets for the LLM."""
        clean_query = query.strip()
        if not clean_query:
            return "No search query provided."

        logger.info("Executing web search for: %r", clean_query)
        html_text = await self._post_search(clean_query)
        if not html_text:
            return f"No results found or search failed for: {clean_query}"

        try:
            tree = html.fromstring(html_text)
            links = tree.xpath('//a[contains(@class, "result-link")]')
            snippets = tree.xpath('//td[contains(@class, "result-snippet")]')

            if not links:
                return f"No search results returned for query: {clean_query}"

            formatted: list[str] = []
            for idx, (a, s) in enumerate(zip(links[:max_results], snippets[:max_results]), start=1):
                title = a.text_content().strip()
                url = a.get("href", "").strip()
                snippet = s.text_content().strip()
                if len(snippet) > 300:
                    snippet = snippet[:297] + "..."
                formatted.append(f"[{idx}] {title}\nURL: {url}\nSnippet: {snippet}")

            return "\n\n".join(formatted)
        except Exception as exc:
            logger.warning("Failed to parse search HTML: %s", exc)
            return f"Error parsing search results for query: {clean_query}"

    async def fetch_page(self, url: str, max_chars: int = 2500) -> str:
        """Fetch and extract readable text from a URL for deep details."""
        clean_url = url.strip()
        if not clean_url or not clean_url.startswith(("http://", "https://")):
            return "Invalid URL provided."

        logger.info("Fetching webpage text from: %s", clean_url)
        html_text = None
        if _HAS_PRIMP:
            try:
                def _do_primp_get() -> str | None:
                    client = primp.Client(timeout=int(self.timeout_seconds))
                    res = client.get(clean_url)
                    if res.status_code < 400:
                        return res.text
                    return None
                html_text = await asyncio.to_thread(_do_primp_get)
            except Exception as exc:
                logger.debug("primp get failed for %s: %s", clean_url, exc)

        if not html_text:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                    resp = await client.get(clean_url, headers=_HEADERS)
                if resp.status_code < 400:
                    html_text = resp.text
            except Exception as exc:
                logger.warning("httpx get failed for %s: %s", clean_url, exc)

        if not html_text:
            return f"Could not retrieve webpage content from {clean_url}"

        try:
            tree = html.fromstring(html_text)
            for tag in tree.xpath('//script | //style | //nav | //footer | //header | //aside'):
                tag.drop_tree()
            text = " ".join(tree.text_content().split())
            if not text:
                return "Page contains no readable text."
            return text[:max_chars]
        except Exception as exc:
            logger.warning("Failed to fetch webpage %s: %s", clean_url, exc)
            return f"Error retrieving page: {exc}"

    async def _post_search(self, query: str) -> str | None:
        """Query DuckDuckGo Lite with fallback from primp to httpx."""
        # Method 1: primp (TLS fingerprint bypass)
        if _HAS_PRIMP:
            try:
                def _do_primp() -> str | None:
                    client = primp.Client(timeout=int(self.timeout_seconds))
                    res = client.post("https://lite.duckduckgo.com/lite/", data={"q": query})
                    if res.status_code == 200:
                        return res.text
                    return None

                result = await asyncio.to_thread(_do_primp)
                if result:
                    return result
            except Exception as exc:
                logger.debug("primp search attempt failed: %s", exc)

        # Method 2: httpx fallback
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                res = await client.post(
                    "https://lite.duckduckgo.com/lite/",
                    data={"q": query},
                    headers=_HEADERS,
                )
                if res.status_code == 200:
                    return res.text
        except Exception as exc:
            logger.warning("httpx search attempt failed: %s", exc)

        return None
