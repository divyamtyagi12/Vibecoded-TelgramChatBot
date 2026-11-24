"""Real-time web search and live data retrieval utility."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any
import xml.etree.ElementTree as ET

from lxml import html

logger = logging.getLogger(__name__)

primp: Any = None
try:
    import primp as _primp
    primp = _primp
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


_FINANCE_QUERY_RE = re.compile(
    r"\b(stock|stocks|share|shares|price|quote|market|nasdaq|nyse|nifty|sensex|ticker|earnings|crypto|bitcoin|btc|eth|nvda|nvidia|aapl|tsla|msft|googl|amzn|meta)\b",
    re.IGNORECASE,
)

_NEWS_QUERY_RE = re.compile(
    r"\b(news|latest|breaking|today|update|updates|happening|recent|headline|headlines|stock|stocks)\b",
    re.IGNORECASE,
)

_FINANCE_STOPWORDS = {
    "stock", "stocks", "share", "shares", "price", "prices", "live", "quote", "quotes",
    "today", "news", "latest", "current", "any", "on", "of", "the", "what", "is", "whats",
    "for", "in", "check", "pull", "get", "me", "about", "update", "updates", "market",
    "value", "how", "much", "close", "closed", "closing", "open", "opening", "trading",
    "performance", "chart", "usd", "inr", "rate", "now", "right",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "2024", "2025", "2026", "2027", "2028",
}

_KNOWN_DIRECT_TICKERS = {
    "nvidia": "NVDA",
    "nvidi": "NVDA",
    "nvidea": "NVDA",
    "nvda": "NVDA",
    "apple": "AAPL",
    "aple": "AAPL",
    "aapl": "AAPL",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "goog": "GOOG",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "amd": "AMD",
    "intel": "INTC",
    "intc": "INTC",
    "netflix": "NFLX",
    "nflx": "NFLX",
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
    "eth": "ETH-USD",
    "ethereum": "ETH-USD",
    "sol": "SOL-USD",
    "solana": "SOL-USD",
    "nifty": "^NSEI",
    "sensex": "^BSESN",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "gold": "GC=F",
    "silver": "SI=F",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
}

_COMMON_TICKER_FIXES = _KNOWN_DIRECT_TICKERS


_AGGREGATOR_DOMAINS = (
    "news.google.com",
    "bing.com",
    "duckduckgo.com",
    "facebook.com",
    "instagram.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
)

_OFFICIAL_HINTS = (
    ".gov",
    ".edu",
    ".org",
    "docs.",
    "official",
    "reuters.com",
    "cnbc.com",
    "bbc.com",
    "apnews.com",
    "bloomberg.com",
    "espncricinfo.com",
    "cricbuzz.com",
    "icc-cricket.com",
    "bcci.tv",
    "liquipedia.net",
    "wikipedia.org",
    "nvidia.com",
    "apple.com",
    "microsoft.com",
    "google.com",
    "openai.com",
    "irctc.co.in",
)


def extract_official_urls(search_text: str, max_urls: int = 2) -> list[str]:
    """Extract and prioritize official/authoritative webpage URLs from search output."""
    if not search_text:
        return []
    raw_urls = re.findall(r"URL:\s*(https?://[^\s]+)", search_text)
    seen: set[str] = set()
    scored: list[tuple[int, str]] = []
    for url in raw_urls:
        clean = url.rstrip(").,;\"'")
        low = clean.lower()
        if clean in seen:
            continue
        seen.add(clean)
        if any(agg in low for agg in _AGGREGATOR_DOMAINS):
            continue
        score = 0
        if any(hint in low for hint in _OFFICIAL_HINTS):
            score += 10
        # Prefer shorter / root or direct official article paths over deep tracking URLs
        if len(clean) < 90:
            score += 2
        scored.append((score, clean))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [u for _, u in scored[:max_urls]]


class WebSearchClient:
    """Client for real-time web search with datacenter-safe fallbacks (Bing RSS, Google News RSS, Yahoo Finance)."""

    def __init__(self, timeout_seconds: float = 4.5) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int = 5) -> str:
        """Perform fast targeted search (parallel only across relevant sources)."""
        clean_query = query.strip()
        if not clean_query:
            return "No search query provided."

        logger.info("Executing web search for: %r", clean_query)

        is_finance = bool(_FINANCE_QUERY_RE.search(clean_query))
        is_news = bool(_NEWS_QUERY_RE.search(clean_query))
        on_vercel = bool(os.getenv("VERCEL"))

        # Run only relevant providers in parallel; skip DDG Lite on Vercel where AWS IPs get HTTP 202
        task_specs: list[tuple[str, Any]] = []
        if is_finance:
            task_specs.append(("Yahoo Finance", self._search_yahoo_finance(clean_query)))
        if not on_vercel:
            task_specs.append(("DuckDuckGo Web", self._search_ddg(clean_query, max_results=max_results)))
        task_specs.append(("Bing Web", self._search_bing_rss(clean_query, max_results=max_results)))
        if is_news or on_vercel:
            task_specs.append(("Google News", self._search_google_news_rss(clean_query, max_results=min(max_results, 4))))

        raw_results = await asyncio.gather(*(coro for _, coro in task_specs), return_exceptions=True)

        sections: list[str] = []
        for (label, _), res in zip(task_specs, raw_results):
            if isinstance(res, str) and res:
                sections.append(f"=== Source: {label} ===\n{res}")

        if not sections:
            return f"No results found or search failed for: {clean_query}"

        return "\n\n".join(sections)

    async def _search_ddg(self, clean_query: str, max_results: int = 5) -> str | None:
        html_text = await self._post_search(clean_query)
        if not html_text:
            return None
        try:
            tree = html.fromstring(html_text)
            links = tree.xpath('//a[contains(@class, "result-link")]')
            snippets = tree.xpath('//td[contains(@class, "result-snippet")]')
            if not links:
                return None

            formatted: list[str] = []
            for idx, (a, s) in enumerate(zip(links[:max_results], snippets[:max_results]), start=1):
                title = a.text_content().strip()
                url = a.get("href", "").strip()
                snippet = s.text_content().strip()
                if len(snippet) > 300:
                    snippet = snippet[:297] + "..."
                formatted.append(f"[{idx}] {title}\nURL: {url}\nSnippet: {snippet}")
            return "\n\n".join(formatted) if formatted else None
        except Exception as exc:
            logger.warning("Failed to parse DDG search HTML: %s", exc)
            return None

    async def _search_bing_rss(self, clean_query: str, max_results: int = 5) -> str | None:
        """Fetch Bing search results via RSS (works reliably from Vercel/AWS datacenter IPs)."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                res = await client.get(
                    "https://www.bing.com/search",
                    params={"q": clean_query, "format": "rss"},
                    headers=_HEADERS,
                )
            if res.status_code != 200 or not res.text:
                return None
            root = ET.fromstring(res.text)
            items = root.findall(".//item")
            if not items:
                return None
            formatted: list[str] = []
            for idx, item in enumerate(items[:max_results], start=1):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                desc = (item.findtext("description") or "").strip()
                if len(desc) > 300:
                    desc = desc[:297] + "..."
                formatted.append(f"[{idx}] {title}\nURL: {link}\nSnippet: {desc}")
            return "\n\n".join(formatted) if formatted else None
        except Exception as exc:
            logger.warning("Bing RSS search failed for %r: %s", clean_query, exc)
            return None

    async def _search_google_news_rss(self, clean_query: str, max_results: int = 4) -> str | None:
        """Fetch real-time news headlines via Google News RSS."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                res = await client.get(
                    "https://news.google.com/rss/search",
                    params={"q": clean_query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                    headers=_HEADERS,
                )
            if res.status_code != 200 or not res.text:
                return None
            root = ET.fromstring(res.text)
            items = root.findall(".//item")
            if not items:
                return None
            lines: list[str] = ["Latest News Headlines:"]
            for idx, item in enumerate(items[:max_results], start=1):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                line = f"- {title} ({pub_date})" if pub_date else f"- {title}"
                if link:
                    line += f"\n  URL: {link}"
                lines.append(line)
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Google News RSS failed for %r: %s", clean_query, exc)
            return None

    async def _search_yahoo_finance(self, clean_query: str) -> str | None:
        """Fetch live stock/crypto quote and headlines from Yahoo Finance."""
        words = re.findall(r"[A-Za-z0-9.\-^]+", clean_query)
        direct_symbol: str | None = None
        candidates: list[str] = []
        for w in words:
            low = w.lower()
            if low in _FINANCE_STOPWORDS or w.isdigit() or len(w) < 2:
                continue
            mapped = _COMMON_TICKER_FIXES.get(low)
            if mapped:
                if direct_symbol is None:
                    direct_symbol = mapped
                if mapped not in candidates:
                    candidates.append(mapped)
            elif w not in candidates:
                candidates.append(w)

        if not candidates and not direct_symbol:
            return None

        search_term = direct_symbol or " ".join(candidates[:2])
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                symbol: str | None = direct_symbol
                shortname: str | None = None
                news: list[dict[str, Any]] = []

                # Try Yahoo search API to resolve unknown symbols and grab finance news
                for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
                    try:
                        search_res = await client.get(
                            f"https://{host}/v1/finance/search",
                            params={"q": search_term, "quotesCount": 2, "newsCount": 3},
                            headers=_HEADERS,
                        )
                        if search_res.status_code == 200:
                            data = search_res.json()
                            quotes = data.get("quotes") or []
                            news = data.get("news") or []
                            if quotes:
                                if not symbol:
                                    symbol = quotes[0].get("symbol")
                                shortname = quotes[0].get("shortname") or quotes[0].get("longname") or symbol
                            break
                    except Exception:
                        continue

                # If multi-word search returned no quotes, retry with just the first candidate
                if not symbol and len(candidates) > 1:
                    try:
                        retry_res = await client.get(
                            "https://query2.finance.yahoo.com/v1/finance/search",
                            params={"q": candidates[0], "quotesCount": 2, "newsCount": 3},
                            headers=_HEADERS,
                        )
                        if retry_res.status_code == 200:
                            data = retry_res.json()
                            quotes = data.get("quotes") or []
                            if not news:
                                news = data.get("news") or []
                            if quotes:
                                symbol = quotes[0].get("symbol")
                                shortname = quotes[0].get("shortname") or quotes[0].get("longname") or symbol
                    except Exception:
                        pass

                out_lines: list[str] = []
                if symbol:
                    chart_data = None
                    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
                        try:
                            chart_res = await client.get(
                                f"https://{host}/v8/finance/chart/{symbol}",
                                params={"interval": "1d", "range": "1d"},
                                headers=_HEADERS,
                            )
                            if chart_res.status_code == 200:
                                chart_data = chart_res.json()
                                break
                        except Exception:
                            continue

                    if chart_data:
                        meta = (chart_data.get("chart", {}).get("result") or [{}])[0].get("meta", {})
                        price = meta.get("regularMarketPrice")
                        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
                        currency = meta.get("currency", "USD")
                        display_name = shortname or meta.get("shortName") or meta.get("longName") or symbol
                        market_ts = meta.get("regularMarketTime")
                        ts_str = ""
                        if isinstance(market_ts, (int, float)):
                            from datetime import datetime, timezone as _tz
                            dt = datetime.fromtimestamp(market_ts, tz=_tz.utc)
                            ts_str = f" [last trade: {dt.strftime('%a, %d %b %Y %H:%M UTC')}]"
                        if price is not None:
                            change_str = ""
                            if prev:
                                diff = price - prev
                                pct = (diff / prev) * 100
                                sign = "+" if diff >= 0 else ""
                                change_str = f" ({sign}{diff:.2f} / {sign}{pct:.2f}% vs prev close {prev:.2f})"
                            out_lines.append(
                                f"AUTHORITATIVE LIVE MARKET QUOTE (Yahoo Finance — use this exact price and prev close): "
                                f"{display_name} ({symbol}) = {price:.2f} {currency}{change_str}{ts_str}\n"
                                f"URL: https://finance.yahoo.com/quote/{symbol}/"
                            )
                if news:
                    out_lines.append("Recent Finance News:")
                    for item in news[:3]:
                        title = (item.get("title") or "").strip()
                        if title:
                            out_lines.append(f"- {title}")
                return "\n".join(out_lines) if out_lines else None
        except Exception as exc:
            logger.debug("Yahoo Finance lookup failed for %r: %s", search_term, exc)
            return None

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
