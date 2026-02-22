"""
BS-Opt Market Data Router
Smart routing between global (Polygon/CCXT) and local (NSE) data sources
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class Exchange(str, Enum):
    """Supported exchanges"""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    NSE_KENYA = "NSE_KE"
    CRYPTO = "CRYPTO"


@dataclass
class MarketTick:
    """Normalized market tick"""
    timestamp: datetime
    symbol: str
    exchange: Exchange
    price: float
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "source": self.source,
        }


@dataclass
class OHLCV:
    """OHLCV bar"""
    timestamp: datetime
    symbol: str
    exchange: Exchange
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ticks(cls, ticks: list[MarketTick], symbol: str, exchange: Exchange) -> "OHLCV":
        """Synthesize OHLCV from sparse tick data (frontier markets)"""
        if not ticks:
            raise ValueError("Cannot create OHLCV from empty tick list")

        prices = [t.price for t in ticks]
        volumes = [t.volume or 0 for t in ticks]

        return cls(
            timestamp=ticks[-1].timestamp,
            symbol=symbol,
            exchange=exchange,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
        )


# =============================================================================
# Data Source Interface
# =============================================================================

class DataSource(ABC):
    """Abstract base for market data sources"""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols"""
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[MarketTick]:
        """Stream ticks"""
        pass


# =============================================================================
# Polygon.io Data Source (US Equities)
# =============================================================================

class PolygonDataSource(DataSource):
    """Polygon.io WebSocket data source for US markets"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._ws = None
        self._subscribed_symbols = []

    async def connect(self) -> None:
        """Connect to Polygon WebSocket"""
        # In production: websockets.connect(f"wss://socket.polygon.io/stocks")
        logger.info("Connecting to Polygon.io...")
        # Placeholder - actual WebSocket connection would go here

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()

    async def subscribe(self, symbols: list[str]) -> None:
        self._subscribed_symbols = symbols
        logger.info(f"Subscribed to {len(symbols)} US symbols")

    async def stream(self) -> AsyncIterator[MarketTick]:
        """Stream ticks from Polygon"""
        while True:
            # Placeholder - would parse WebSocket messages
            await asyncio.sleep(0.1)
            yield MarketTick(
                timestamp=datetime.utcnow(),
                symbol="AAPL",
                exchange=Exchange.NASDAQ,
                price=175.50,
                volume=1000,
                source="polygon",
            )


# =============================================================================
# NSE Kenya Scraper (Frontier Market)
# =============================================================================

class NSEKenyaDataSource(DataSource):
    """
    Web scraper for Nairobi Securities Exchange.
    Handles sparse data characteristic of frontier markets.
    """

    NSE_URL = "https://www.nse.co.ke"

    def __init__(self):
        self._browser = None
        self._last_prices: dict[str, float] = {}

    async def connect(self) -> None:
        """Initialize Playwright browser"""
        try:
            from playwright.async_api import async_playwright

            logger.info("Launching headless browser for NSE scraping...")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            logger.info("NSE scraper ready")
        except ImportError:
            logger.warning("Playwright not installed - NSE scraping disabled")

    async def disconnect(self) -> None:
        if self._browser:
            await self._browser.close()
        if hasattr(self, "_playwright"):
            await self._playwright.stop()

    async def subscribe(self, symbols: list[str]) -> None:
        """Store symbols to scrape"""
        self._subscribed_symbols = symbols
        logger.info(f"Will scrape {len(symbols)} NSE symbols")

    async def stream(self) -> AsyncIterator[MarketTick]:
        """Periodically scrape NSE and yield ticks"""
        if not self._browser:
            logger.error("Browser not initialized")
            return

        while True:
            try:
                ticks = await self._scrape_nse()
                for tick in ticks:
                    yield tick
            except Exception as e:
                logger.error(f"NSE scrape error: {e}")

            # NSE data updates infrequently - poll every 60s
            await asyncio.sleep(60)

    async def _scrape_nse(self) -> list[MarketTick]:
        """Scrape current prices from NSE website"""
        page = await self._browser.new_page()
        ticks = []

        try:
            await page.goto(f"{self.NSE_URL}/market-statistics", timeout=30000)

            # Wait for price table to load
            await page.wait_for_selector("table.market-stats", timeout=10000)

            # Extract prices (placeholder selectors)
            rows = await page.query_selector_all("table.market-stats tbody tr")

            for row in rows:
                symbol = await row.query_selector("td:nth-child(1)")
                price = await row.query_selector("td:nth-child(3)")
                volume = await row.query_selector("td:nth-child(4)")

                if symbol and price:
                    symbol_text = await symbol.inner_text()
                    price_text = await price.inner_text()
                    volume_text = await volume.inner_text() if volume else "0"

                    ticks.append(MarketTick(
                        timestamp=datetime.utcnow(),
                        symbol=symbol_text.strip(),
                        exchange=Exchange.NSE_KENYA,
                        price=float(price_text.replace(",", "")),
                        volume=float(volume_text.replace(",", "")) if volume_text else None,
                        source="nse_scraper",
                    ))
        finally:
            await page.close()

        return ticks


# =============================================================================
# Smart Market Router
# =============================================================================

class MarketDataRouter:
    """
    Unified market data router.
    Routes requests to appropriate data source based on symbol/exchange.
    """

    def __init__(
        self,
        polygon_api_key: str | None = None,
        enable_nse: bool = True,
    ):
        self.sources: dict[Exchange, DataSource] = {}

        # Initialize sources based on configuration
        if polygon_api_key:
            self.sources[Exchange.NYSE] = PolygonDataSource(polygon_api_key)
            self.sources[Exchange.NASDAQ] = self.sources[Exchange.NYSE]

        if enable_nse:
            self.sources[Exchange.NSE_KENYA] = NSEKenyaDataSource()

    async def start(self) -> None:
        """Connect all data sources"""
        for exchange, source in self.sources.items():
            try:
                await source.connect()
                logger.info(f"Connected to {exchange.value}")
            except Exception as e:
                logger.error(f"Failed to connect to {exchange.value}: {e}")

    async def stop(self) -> None:
        """Disconnect all sources"""
        for source in set(self.sources.values()):
            await source.disconnect()

    def route(self, symbol: str) -> Exchange:
        """Determine which exchange a symbol belongs to"""
        symbol = symbol.upper()

        # Kenya-specific symbols
        if symbol.endswith(".KE") or symbol in ["SCOM", "EQTY", "KCB", "SBIC", "COOP"]:
            return Exchange.NSE_KENYA

        # Crypto
        if symbol.endswith("USD") or symbol.endswith("USDT"):
            return Exchange.CRYPTO

        # Default to US
        return Exchange.NASDAQ

    async def get_latest(self, symbol: str) -> MarketTick | None:
        """Get latest tick for a symbol"""
        exchange = self.route(symbol)
        source = self.sources.get(exchange)

        if not source:
            logger.warning(f"No source for {exchange.value}")
            return None

        # In production, would query cache or latest from stream
        return None

    async def stream_all(self) -> AsyncIterator[MarketTick]:
        """Unified stream from all sources"""

        async def source_stream(source: DataSource):
            async for tick in source.stream():
                yield tick

        # Merge streams from all sources
        sources = list(set(self.sources.values()))

        # Create tasks for each source
        tasks = [asyncio.create_task(source_stream(s).__anext__()) for s in sources]

        while tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    tick = task.result()
                    yield tick
                except StopAsyncIteration:
                    pass


# =============================================================================
# Illiquidity Adjustment (Frontier Markets)
# =============================================================================

def apply_illiquidity_discount(
    theoretical_price: float,
    bid_ask_spread_pct: float,
    volume: float,
    volume_threshold: float = 10000,
    max_discount: float = 0.15,
) -> tuple[float, float]:
    """
    Apply discount to theoretical price for illiquid frontier markets.

    Returns:
        (adjusted_price, discount_factor)
    """
    discount = 0.0

    # Spread-based discount
    if bid_ask_spread_pct > 0.02:  # > 2% spread
        discount += min(bid_ask_spread_pct * 2, 0.10)

    # Volume-based discount
    if volume < volume_threshold:
        volume_discount = (1 - volume / volume_threshold) * 0.05
        discount += volume_discount

    # Cap total discount
    discount = min(discount, max_discount)

    adjusted_price = theoretical_price * (1 - discount)

    return adjusted_price, discount
