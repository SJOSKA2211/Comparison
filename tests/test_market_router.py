"""
Tests for MarketDataRouter logic
"""
import pytest
from src.data.market_router import MarketDataRouter, Exchange

class TestMarketRouterRouting:
    """Tests for symbol routing logic"""

    def setup_method(self):
        """Setup router instance"""
        # Disable NSE to avoid any potential connection attempts, though routing is pure logic
        self.router = MarketDataRouter(enable_nse=False, polygon_api_key=None)

    def test_kenya_symbols_suffix(self):
        """Symbols ending in .KE should route to NSE_KENYA"""
        assert self.router.route("SAFARICOM.KE") == Exchange.NSE_KENYA
        assert self.router.route("ABC.KE") == Exchange.NSE_KENYA
        assert self.router.route("X.KE") == Exchange.NSE_KENYA

    def test_kenya_explicit_list(self):
        """Specific symbols should route to NSE_KENYA"""
        kenya_symbols = ["SCOM", "EQTY", "KCB", "SBIC", "COOP"]
        for symbol in kenya_symbols:
            assert self.router.route(symbol) == Exchange.NSE_KENYA

    def test_crypto_symbols(self):
        """Crypto symbols (USD/USDT suffix) should route to CRYPTO"""
        assert self.router.route("BTCUSD") == Exchange.CRYPTO
        assert self.router.route("ETHUSDT") == Exchange.CRYPTO
        assert self.router.route("DOGEUSD") == Exchange.CRYPTO
        assert self.router.route("SOLUSDT") == Exchange.CRYPTO

    def test_default_us_equity(self):
        """Other symbols should default to NASDAQ"""
        assert self.router.route("AAPL") == Exchange.NASDAQ
        assert self.router.route("GOOGL") == Exchange.NASDAQ
        assert self.router.route("TSLA") == Exchange.NASDAQ
        assert self.router.route("IBM") == Exchange.NASDAQ

    def test_case_insensitivity(self):
        """Routing should handle mixed case inputs"""
        assert self.router.route("scom") == Exchange.NSE_KENYA
        assert self.router.route("Safaraicom.ke") == Exchange.NSE_KENYA
        assert self.router.route("btcusd") == Exchange.CRYPTO
        assert self.router.route("ethUsdt") == Exchange.CRYPTO
        assert self.router.route("aapl") == Exchange.NASDAQ

    def test_ambiguous_cases(self):
        """Verify precedence rules"""
        # .KE check comes before crypto check
        assert self.router.route("BTCUSD.KE") == Exchange.NSE_KENYA

        # Exact match SCOM check
        # SCOMUSD does not end with .KE, is not in ["SCOM", ...].
        # It ends with USD. So it should be CRYPTO.
        assert self.router.route("SCOMUSD") == Exchange.CRYPTO

    def test_empty_input(self):
        """Empty string should default to NASDAQ"""
        assert self.router.route("") == Exchange.NASDAQ
