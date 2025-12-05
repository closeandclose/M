#!/usr/bin/env python3
"""
ETH 1-Hour Binary Price Prediction Model
Uses advanced technical indicators to predict 1-hour price movements.
Returns 2-dimensional embedding in range [-1, 1].
"""

import numpy as np
import requests
from typing import Dict, List, Optional


class BinaryETH1HPredictor:
    """
    Predicts ETH 1-hour binary price movements using technical analysis.
    
    Features:
    - Moving Averages (SMA5, SMA10, SMA20, EMA5, EMA10, EMA20)
    - RSI (EMA-based)
    - Momentum (multi-timeframe)
    - Volume Analysis
    - Volatility
    - MACD
    - Bollinger Bands
    - ADX (Average Directional Index)
    """
    
    def __init__(self):
        """Initialize the predictor."""
        pass
    
    def get_klines(self, symbol: str = "ETHUSDT", interval: str = "5m", limit: int = 2880) -> Optional[List]:
        """
        Fetch historical klines from Binance API.
        
        Args:
            symbol: Trading pair symbol (default: ETHUSDT)
            interval: Kline interval (default: 5m for 5-minute candles)
            limit: Number of klines to fetch (default: 2880 = 10 days of 5-minute data)
        
        Returns:
            List of klines or None if error
        """
        try:
            url = "https://api.binance.us/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return None
    
    def klines_to_dataframe(self, klines: List) -> Dict[str, np.ndarray]:
        """
        Convert Binance klines to numpy arrays.
        
        Args:
            klines: List of kline data from Binance API
        
        Returns:
            Dictionary with 'prices', 'volumes', 'highs', 'lows' arrays
        """
        if not klines or len(klines) == 0:
            return {}
        
        prices = np.array([float(k[4]) for k in klines], dtype=np.float64)  # Close prices
        volumes = np.array([float(k[5]) for k in klines], dtype=np.float64)
        highs = np.array([float(k[2]) for k in klines], dtype=np.float64)
        lows = np.array([float(k[3]) for k in klines], dtype=np.float64)
        
        return {
            "prices": prices,
            "volumes": volumes,
            "highs": highs,
            "lows": lows
        }
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate RSI using EMA method for more accurate results.
        
        Args:
            prices: Price array
            period: RSI period (default: 14)
        
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use EMA for smoothing
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price array
            period: EMA period
        
        Returns:
            EMA value
        """
        if len(prices) < period:
            return float(np.mean(prices))
        
        multiplier = 2.0 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
    
    def calculate_moving_averages(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate Simple and Exponential Moving Averages.
        
        Args:
            prices: Price array
        
        Returns:
            Dictionary with SMA5, SMA10, SMA20, EMA5, EMA10, EMA20
        """
        result = {}
        
        for period in [5, 10, 20]:
            if len(prices) >= period:
                result[f"sma_{period}"] = float(np.mean(prices[-period:]))
                result[f"ema_{period}"] = self.calculate_ema(prices, period)
            else:
                result[f"sma_{period}"] = float(np.mean(prices))
                result[f"ema_{period}"] = float(np.mean(prices))
        
        return result
    
    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """
        Calculate price momentum using multi-timeframe approach.
        
        Args:
            prices: Price array
            period: Momentum period
        
        Returns:
            Momentum value (normalized)
        """
        if len(prices) < period * 2:
            return 0.0
        
        # Short-term momentum (60% weight)
        short_momentum = (prices[-1] - prices[-period]) / prices[-period]
        
        # Medium-term momentum (40% weight)
        medium_momentum = (prices[-1] - prices[-period * 2]) / prices[-period * 2]
        
        # Weighted combination
        momentum = 0.6 * short_momentum + 0.4 * medium_momentum
        
        return float(momentum)
    
    def calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """
        Calculate volatility as standard deviation of returns.
        
        Args:
            prices: Price array
            period: Period for volatility calculation
        
        Returns:
            Volatility value
        """
        if len(prices) < period + 1:
            return 0.0
        
        returns = np.diff(prices[-period-1:]) / prices[-period-1:-1]
        volatility = np.std(returns)
        
        return float(volatility)
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price array
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
        
        Returns:
            Dictionary with macd_line, signal_line, histogram
        """
        if len(prices) < slow + signal:
            return {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        # For simplicity, use recent MACD values
        macd_values = []
        for i in range(slow, len(prices)):
            ema_f = self.calculate_ema(prices[:i+1], fast)
            ema_s = self.calculate_ema(prices[:i+1], slow)
            macd_values.append(ema_f - ema_s)
        
        if len(macd_values) >= signal:
            signal_line = self.calculate_ema(np.array(macd_values[-signal:]), signal)
        else:
            signal_line = np.mean(macd_values) if macd_values else 0.0
        
        histogram = macd_line - signal_line
        
        return {
            "macd_line": float(macd_line),
            "signal_line": float(signal_line),
            "histogram": float(histogram)
        }
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price array
            period: Period for moving average
            std_dev: Standard deviation multiplier
        
        Returns:
            Dictionary with upper, middle, lower, width, position
        """
        if len(prices) < period:
            return {
                "upper": float(prices[-1]),
                "middle": float(np.mean(prices)),
                "lower": float(prices[-1]),
                "width": 0.0,
                "position": 0.5
            }
        
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        width = (upper - lower) / middle if middle > 0 else 0.0
        
        # Position of current price relative to bands (0 = lower, 1 = upper)
        if upper != lower:
            position = (prices[-1] - lower) / (upper - lower)
        else:
            position = 0.5
        
        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
            "width": float(width),
            "position": float(position)
        }
    
    def calculate_adx(self, prices: np.ndarray, high: np.ndarray = None, low: np.ndarray = None, period: int = 14) -> float:
        """
        Calculate ADX (Average Directional Index) for trend strength.
        Simplified version using price changes.
        
        Args:
            prices: Price array
            high: High prices (optional, uses prices if None)
            low: Low prices (optional, uses prices if None)
            period: ADX period
        
        Returns:
            ADX value (0-100)
        """
        if len(prices) < period * 2:
            return 25.0  # Neutral value
        
        if high is None:
            high = prices
        if low is None:
            low = prices
        
        # Simplified ADX calculation
        price_changes = np.abs(np.diff(prices[-period-1:]))
        avg_change = np.mean(price_changes)
        
        if avg_change == 0:
            return 25.0
        
        # Normalize to 0-100 range
        max_change = np.max(price_changes)
        adx = min(100.0, (avg_change / max_change) * 100.0) if max_change > 0 else 25.0
        
        return float(adx)
    
    def _calculate_bullish_score(self, indicators: Dict) -> float:
        """
        Calculate bullish score (0-1) by combining all technical indicators.
        
        Args:
            indicators: Dictionary with all calculated indicators
        
        Returns:
            Bullish score between 0 and 1
        """
        base_score = 0.5  # Neutral starting point
        adjustment = 0.0
        
        # 1. Moving Averages (35% weight, max ±0.30)
        mas = indicators.get("moving_averages", {})
        ma5_sma = mas.get("sma_5", 0)
        ma10_sma = mas.get("sma_10", 0)
        ma20_sma = mas.get("sma_20", 0)
        ma5_ema = mas.get("ema_5", 0)
        ma10_ema = mas.get("ema_10", 0)
        ma20_ema = mas.get("ema_20", 0)
        current_price = indicators.get("current_price", 0)
        
        if current_price > 0:
            # SMA crossovers
            sma_signal = 0.0
            if ma5_sma > ma10_sma > ma20_sma:
                sma_signal = 0.15
            elif ma5_sma < ma10_sma < ma20_sma:
                sma_signal = -0.15
            
            # EMA crossovers
            ema_signal = 0.0
            if ma5_ema > ma10_ema > ma20_ema:
                ema_signal = 0.115
            elif ma5_ema < ma10_ema < ma20_ema:
                ema_signal = -0.115
            
            # Price position relative to MA20
            position_signal = 0.0
            if ma20_sma > 0:
                price_ratio = (current_price - ma20_sma) / ma20_sma
                position_signal = np.clip(price_ratio * 0.035, -0.035, 0.035)
            
            adjustment += sma_signal + ema_signal + position_signal
        
        # 2. RSI (20% weight, max ±0.08)
        rsi = indicators.get("rsi", 50.0)
        rsi_signal = 0.0
        if rsi < 25:
            rsi_signal = 0.08  # Strongly oversold
        elif rsi < 35:
            rsi_signal = 0.04
        elif rsi > 75:
            rsi_signal = -0.08  # Strongly overbought
        elif rsi > 65:
            rsi_signal = -0.04
        adjustment += rsi_signal
        
        # 3. Momentum (20% weight, max ±0.10)
        momentum = indicators.get("momentum", 0.0)
        momentum_signal = np.clip(momentum * 0.10, -0.10, 0.10)
        adjustment += momentum_signal
        
        # 4. Volume Confirmation (8% weight, max ±0.04)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        volume_signal = 0.0
        if volume_ratio > 1.5:  # High volume
            # Confirm existing trend
            if adjustment > 0:
                volume_signal = 0.04
            elif adjustment < 0:
                volume_signal = -0.04
        adjustment += volume_signal
        
        # 5. MACD (12% weight, max ±0.06)
        macd = indicators.get("macd", {})
        macd_histogram = macd.get("histogram", 0.0)
        macd_line = macd.get("macd_line", 0.0)
        macd_signal_line = macd.get("signal_line", 0.0)
        
        macd_signal = 0.0
        if macd_histogram > 0 and macd_line > macd_signal_line:
            macd_signal = 0.06
        elif macd_histogram < 0 and macd_line < macd_signal_line:
            macd_signal = -0.06
        else:
            macd_signal = np.clip(macd_histogram * 0.03, -0.06, 0.06)
        adjustment += macd_signal
        
        # 6. Bollinger Bands (8% weight, max ±0.04)
        bb = indicators.get("bollinger_bands", {})
        bb_position = bb.get("position", 0.5)
        bb_width = bb.get("width", 0.0)
        
        bb_signal = 0.0
        if bb_position < 0.2:  # Near lower band (oversold)
            bb_signal = 0.04
            if bb_width < 0.02:  # Narrow bands strengthen signal
                bb_signal = 0.04
        elif bb_position > 0.8:  # Near upper band (overbought)
            bb_signal = -0.04
            if bb_width < 0.02:  # Narrow bands strengthen signal
                bb_signal = -0.04
        adjustment += bb_signal
        
        # 7. ADX (5% weight, max ±0.01)
        adx = indicators.get("adx", 25.0)
        adx_signal = 0.0
        if adx > 50:  # Strong trend
            # Strengthen existing signal
            adx_signal = np.clip(adjustment * 0.01, -0.01, 0.01)
        elif adx < 20:  # Weak trend
            # Reduce signal strength
            adx_signal = -np.clip(abs(adjustment) * 0.005, 0, 0.01)
        adjustment += adx_signal
        
        # 8. Volatility Adjustment (2% weight, max ±0.01)
        volatility = indicators.get("volatility", 0.0)
        vol_signal = 0.0
        if volatility > 0.05:  # High volatility
            vol_signal = -0.01  # Reduce confidence
        elif volatility < 0.01:  # Low volatility
            vol_signal = 0.01  # Increase confidence
        adjustment += vol_signal
        
        # Calculate final score
        bullish_score = base_score + adjustment
        bullish_score = np.clip(bullish_score, 0.0, 1.0)
        
        return float(bullish_score)
    
    def predict(self, symbol: str = "ETHUSDT") -> List[float]:
        """
        Predict 1-hour binary price movement.
        
        Args:
            symbol: Trading pair symbol (default: ETHUSDT)
        
        Returns:
            2-dimensional embedding in range [-1, 1]
        """
        # Fetch data
        klines = self.get_klines(symbol)
        if not klines:
            return [0.0, 0.0]  # Neutral prediction on error
        
        # Convert to arrays
        data = self.klines_to_dataframe(klines)
        if not data:
            return [0.0, 0.0]
        
        prices = data["prices"]
        volumes = data["volumes"]
        highs = data.get("highs", prices)
        lows = data.get("lows", prices)
        
        if len(prices) < 50:
            return [0.0, 0.0]
        
        # Calculate indicators
        mas = self.calculate_moving_averages(prices)
        rsi = self.calculate_rsi(prices)
        momentum = self.calculate_momentum(prices)
        volatility = self.calculate_volatility(prices)
        macd = self.calculate_macd(prices)
        bb = self.calculate_bollinger_bands(prices)
        adx = self.calculate_adx(prices, highs, lows)
        
        # Volume ratio (current vs average)
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1] if len(volumes) > 0 else 1.0
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # Prepare indicators dict
        indicators = {
            "current_price": float(prices[-1]),
            "moving_averages": mas,
            "rsi": rsi,
            "momentum": momentum,
            "volatility": volatility,
            "macd": macd,
            "bollinger_bands": bb,
            "adx": adx,
            "volume_ratio": float(volume_ratio)
        }
        
        # Calculate bullish score
        bullish_score = self._calculate_bullish_score(indicators)
        
        # Convert to embedding: [bullish_probability - 0.5, bearish_probability - 0.5]
        # Both values in range [-0.5, 0.5], then scale to [-1, 1]
        bullish_embedding = (bullish_score - 0.5) * 2.0
        bearish_embedding = ((1.0 - bullish_score) - 0.5) * 2.0
        
        return [bullish_embedding, bearish_embedding]
    
    def predict_with_info(self, symbol: str = "ETHUSDT") -> Dict:
        """
        Predict with detailed indicator information.
        
        Args:
            symbol: Trading pair symbol (default: ETHUSDT)
        
        Returns:
            Dictionary with 'prediction' (embedding) and 'indicators'
        """
        # Fetch data
        klines = self.get_klines(symbol)
        if not klines:
            return {
                "prediction": [0.0, 0.0],
                "indicators": {}
            }
        
        # Convert to arrays
        data = self.klines_to_dataframe(klines)
        if not data:
            return {
                "prediction": [0.0, 0.0],
                "indicators": {}
            }
        
        prices = data["prices"]
        volumes = data["volumes"]
        highs = data.get("highs", prices)
        lows = data.get("lows", prices)
        
        if len(prices) < 50:
            return {
                "prediction": [0.0, 0.0],
                "indicators": {}
            }
        
        # Calculate indicators
        mas = self.calculate_moving_averages(prices)
        rsi = self.calculate_rsi(prices)
        momentum = self.calculate_momentum(prices)
        volatility = self.calculate_volatility(prices)
        macd = self.calculate_macd(prices)
        bb = self.calculate_bollinger_bands(prices)
        adx = self.calculate_adx(prices, highs, lows)
        
        # Volume ratio
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1] if len(volumes) > 0 else 1.0
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # Prepare indicators dict
        indicators = {
            "current_price": float(prices[-1]),
            "moving_averages": mas,
            "rsi": rsi,
            "momentum": momentum,
            "volatility": volatility,
            "macd": macd,
            "bollinger_bands": bb,
            "adx": adx,
            "volume_ratio": float(volume_ratio)
        }
        
        # Calculate bullish score
        bullish_score = self._calculate_bullish_score(indicators)
        
        # Convert to embedding
        bullish_embedding = (bullish_score - 0.5) * 2.0
        bearish_embedding = ((1.0 - bullish_score) - 0.5) * 2.0
        
        return {
            "prediction": [bullish_embedding, bearish_embedding],
            "indicators": indicators,
            "bullish_score": bullish_score
        }


def generate_eth_binary_prediction() -> List[float]:
    """
    Convenience function to generate ETH binary prediction.
    
    Returns:
        2-dimensional embedding in range [-1, 1]
    """
    predictor = BinaryETH1HPredictor()
    return predictor.predict("ETHUSDT")


if __name__ == "__main__":
    # Test the predictor
    predictor = BinaryETH1HPredictor()
    result = predictor.predict_with_info("ETHUSDT")
    
    print("=" * 70)
    print("ETH 1-Hour Binary Prediction")
    print("=" * 70)
    print(f"Prediction (embedding): {result['prediction']}")
    print(f"Bullish Score: {result.get('bullish_score', 0.5):.4f}")
    
    indicators = result.get("indicators", {})
    if indicators:
        print("\nTechnical Indicators:")
        print(f"  Current Price: ${indicators.get('current_price', 0):.2f}")
        
        mas = indicators.get("moving_averages", {})
        print(f"  SMA5: ${mas.get('sma_5', 0):.2f}")
        print(f"  SMA10: ${mas.get('sma_10', 0):.2f}")
        print(f"  SMA20: ${mas.get('sma_20', 0):.2f}")
        print(f"  EMA5: ${mas.get('ema_5', 0):.2f}")
        print(f"  EMA10: ${mas.get('ema_10', 0):.2f}")
        print(f"  EMA20: ${mas.get('ema_20', 0):.2f}")
        
        print(f"  RSI: {indicators.get('rsi', 50):.2f}")
        print(f"  Momentum: {indicators.get('momentum', 0):.4f}")
        print(f"  Volatility: {indicators.get('volatility', 0):.4f}")
        print(f"  Volume Ratio: {indicators.get('volume_ratio', 1):.2f}")
        
        macd = indicators.get("macd", {})
        print(f"  MACD Line: {macd.get('macd_line', 0):.4f}")
        print(f"  MACD Signal: {macd.get('signal_line', 0):.4f}")
        print(f"  MACD Histogram: {macd.get('histogram', 0):.4f}")
        
        bb = indicators.get("bollinger_bands", {})
        print(f"  BB Upper: ${bb.get('upper', 0):.2f}")
        print(f"  BB Middle: ${bb.get('middle', 0):.2f}")
        print(f"  BB Lower: ${bb.get('lower', 0):.2f}")
        print(f"  BB Width: {bb.get('width', 0):.4f}")
        print(f"  BB Position: {bb.get('position', 0.5):.2f}")
        
        print(f"  ADX: {indicators.get('adx', 25):.2f}")
    
    print("=" * 70)

