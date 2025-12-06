#!/usr/bin/env python3
"""
Multi-Exchange Dataset Generator for Forex Pairs
Generates sliding window datasets with 5-day windows (1440 prices at 5-minute intervals) 
and 1-hour future price predictions.
Uses Yahoo Finance API (yfinance) for data fetching.
Note: Yahoo Finance provides maximum 60 days of 5-minute interval data.
"""

import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Yahoo Finance symbol mapping for forex pairs
FOREX_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "CADUSD": "CADUSD=X",
    "NZDUSD": "NZDUSD=X",
    "CHFUSD": "CHFUSD=X",
    "XAUUSD": "GC=F",  # Gold Futures
    "XAGUSD": "SI=F",  # Silver Futures
}


def fetch_forex_klines_yahoo(symbol: str, start_time: datetime, end_time: datetime) -> List[List]:
    """
    Fetch forex 5-minute klines from Yahoo Finance API (yfinance).
    
    Note: Yahoo Finance provides maximum 60 days of 5-minute interval data.
    
    Args:
        symbol: Forex pair symbol (e.g., "EURUSD")
        start_time: Start datetime
        end_time: End datetime
    
    Returns:
        List of klines (each kline is a list with timestamp_ms, open, high, low, close, volume)
    """
    yf_symbol = FOREX_SYMBOLS.get(symbol, symbol)
    logger.info(f"📊 [Yahoo Finance API] Fetching data for {symbol}")
    logger.info(f"   Symbol: {symbol} -> {yf_symbol}")
    logger.info(f"   Interval: 5m")
    logger.info(f"   Period: {start_time} to {end_time}")
    
    all_klines = []
    
    try:
        ticker = yf.Ticker(yf_symbol)
        logger.info(f"   Initialized yfinance.Ticker for {yf_symbol}")
        
        # Yahoo Finance maximum: 60 days for 5-minute data
        # Calculate days difference
        days_diff = (end_time - start_time).days
        
        if days_diff > 60:
            logger.warning(f"   ⚠️ Requested period ({days_diff} days) exceeds Yahoo Finance limit (60 days)")
            logger.warning(f"   ⚠️ Will only fetch last 60 days of data")
            # Adjust start_time to last 60 days
            start_time = end_time - timedelta(days=60)
            logger.info(f"   Adjusted start_time to: {start_time}")
        
        # Calculate period based on interval
        # For 5m interval: maximum 60 days
        total_days = min(60, days_diff) if days_diff > 0 else 60
        period = f"{total_days}d"
        
        logger.info(f"   Requesting data from Yahoo Finance API: period={period}, interval=5m")
        
        # Fetch data using the specified interval
        # Yahoo Finance supports: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        try:
            data = ticker.history(period=period, interval="5m")
            logger.info(f"   ✅ Yahoo Finance API response received: {len(data)} rows")
            
            if data.empty:
                # Try with a fixed period if calculated period fails
                logger.warning(f"   ⚠️ Empty data with period={period}, trying 60d period")
                data = ticker.history(period="60d", interval="5m")
                logger.info(f"   ✅ Retry with 60d period: {len(data)} rows")
        except Exception as e:
            # Fallback: try with a fixed period if interval fails
            logger.warning(f"   ⚠️ Failed to fetch 5m data with period {period}: {e}")
            logger.info(f"   Retrying with fixed 60d period...")
            try:
                data = ticker.history(period="60d", interval="5m")
                logger.info(f"   ✅ Retry successful: {len(data)} rows")
            except Exception as e2:
                logger.error(f"   ❌ Failed to fetch data with interval 5m: {e2}")
                return []
        
        if data.empty:
            logger.warning(f"   ❌ No data returned for {symbol} with interval 5m")
            return []
        
        logger.info(f"   Processing {len(data)} rows from Yahoo Finance API")
        
        # Convert to list format similar to klines
        # Yahoo Finance DataFrame has: Open, High, Low, Close, Volume columns
        for idx, row in data.iterrows():
            # Convert timestamp to milliseconds
            timestamp_ms = int(idx.timestamp() * 1000)
            all_klines.append([
                timestamp_ms,          # Open time (timestamp in milliseconds)
                float(row["Open"]),    # Open
                float(row["High"]),    # High
                float(row["Low"]),     # Low
                float(row["Close"]),   # Close
                float(row["Volume"]) if "Volume" in row else 0.0,  # Volume
            ])
        
        # Sort by timestamp to ensure chronological order
        all_klines.sort(key=lambda x: x[0])
        
        logger.info(f"   ✅ Successfully processed {len(all_klines)} candles from Yahoo Finance API")
        if all_klines:
            logger.info(f"   Data range: {datetime.fromtimestamp(all_klines[0][0] / 1000)} to {datetime.fromtimestamp(all_klines[-1][0] / 1000)}")
        
        return all_klines
        
    except Exception as e:
        logger.error(f"   ❌ Error fetching data from Yahoo Finance API for {symbol}: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return []


def create_and_save_dataset(klines: List[List], filename: str = "forex_dataset.data"):
    """
    Create dataset structure and save directly to .data file.
    - Each big array contains: [[x0], [y0]] where:
      - x0: 1440 prices as PyTorch tensor (5 days at 5-minute intervals: 5 * 24 * 12 = 1440)
      - y0: price 1 hour (12 periods of 5 minutes) later as PyTorch tensor
    - Each big array starts 5 minutes later than previous
    
    Args:
        klines: List of klines from EOD API
        filename: Output filename (will be converted to absolute path)
    
    Returns:
        True if successful, False otherwise
    """
    # Get absolute path based on script directory to ensure files are saved in correct location
    script_dir = Path(__file__).parent.absolute()
    if not os.path.isabs(filename):
        filename = str(script_dir / filename)
    else:
        filename = os.path.abspath(filename)
    
    # Window size: 1440 prices (5 days at 5-minute intervals)
    window_size = 1440
    # Future offset: 12 periods (1 hour = 12 * 5 minutes)
    future_offset = 12
    
    if len(klines) < window_size + future_offset:
        print(f"Error: Not enough data. Need at least {window_size + future_offset} klines, got {len(klines)}")
        return False
    
    # Extract close prices (index 4 in kline array)
    prices = [float(kline[4]) for kline in klines]
    
    # Calculate how many windows we can create
    # Each window starts 5 minutes later (1 period)
    max_windows = len(prices) - window_size - future_offset + 1
    
    if max_windows <= 0:
        print(f"Error: Not enough data to create even one window.")
        return False
    
    print(f"Creating {max_windows} sliding windows...")
    
    # Collect all windows in memory and save directly to .data file
    all_windows = []
    
    for i in range(max_windows):
        # Extract 1440 consecutive prices and convert to PyTorch tensor
        price_window = torch.tensor(prices[i:i + window_size], dtype=torch.float64)
        
        # Get price 1 hour (12 periods) after the last price in window
        future_price_index = i + window_size + future_offset - 1
        
        # Validate future price index (defensive check)
        if future_price_index >= len(prices):
            print(f"ERROR: Future price index {future_price_index} exceeds prices length {len(prices)}")
            return False
        
        future_price = torch.tensor(float(prices[future_price_index]), dtype=torch.float64)
        
        # Create big array: [[x0], [y0]] format
        # x0: PyTorch tensor with 1440 prices
        # y0: PyTorch tensor with future price
        big_array = [[price_window], [future_price]]
        
        # Add to all windows list
        all_windows.append(big_array)
        
        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"  Created {i + 1}/{max_windows} windows ({(i + 1) * 100 / max_windows:.1f}%)...")
    
    if len(all_windows) == 0:
        print("ERROR: No windows were created!")
        return False
    
    print(f"\nAll {len(all_windows)} windows created. Saving directly to {os.path.basename(filename)}...")
    
    # Save directly to final .data file using PyTorch format
    torch.save(all_windows, filename)
    
    print(f"✓ Dataset saved successfully!")
    print(f"  Total windows: {len(all_windows)}")
    if len(all_windows) > 0:
        x0 = all_windows[0][0][0]  # [[x0], [y0]] structure
        y0 = all_windows[0][1][0]
        print(f"  Each window contains: [[x0], [y0]] format")
        print(f"  x0 shape: {x0.shape}, y0 shape: {y0.shape}")
    
    return True


def generate_dataset_for_symbol(symbol: str, start_date: datetime, end_date: datetime):
    """
    Generate dataset for a single forex symbol.
    
    Args:
        symbol: Forex pair symbol (e.g., "EURUSD")
        start_date: Start date for data fetching
        end_date: End date for data fetching
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"{symbol} Dataset Generator")
    print("=" * 70)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Duration: {end_date - start_date}")
    print(f"⚠️  Note: Yahoo Finance provides max 60 days of 5-minute data")
    print("=" * 70)
    
    # Fetch klines
    klines = fetch_forex_klines_yahoo(symbol, start_date, end_date)
    
    if not klines:
        print(f"No klines fetched for {symbol}. Skipping.")
        return False
    
    # Create and save dataset directly to .data file
    script_dir = Path(__file__).parent.absolute()
    output_file = str(script_dir / f"{symbol.lower()}_dataset.data")
    success = create_and_save_dataset(klines, output_file)
    
    if not success:
        print(f"Dataset creation failed for {symbol}. Skipping.")
        return False
    
    # Verify saved dataset
    print(f"\nVerifying saved dataset for {symbol}...")
    try:
        loaded = torch.load(output_file)
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of windows: {len(loaded)}")
        if len(loaded) > 0:
            x0 = loaded[0][0][0]  # [[x0], [y0]] structure
            y0 = loaded[0][1][0]
            print(f"  First window structure: [[x0], [y0]]")
            print(f"  x0 type: {type(x0)}, shape: {x0.shape}")
            print(f"  y0 type: {type(y0)}, shape: {y0.shape}")
            print(f"  First window prices range: {float(torch.min(x0)):.5f} - {float(torch.max(x0)):.5f}")
            print(f"  First window future price: {float(y0):.5f}")
    except Exception as e:
        print(f"Error verifying dataset: {e}")
    
    return True


def main():
    """
    Main function to generate datasets for all forex pairs.
    
    Note: Yahoo Finance provides maximum 60 days of 5-minute interval data.
    The script will fetch the most recent 60 days of data.
    """
    # End date: now
    end_date = datetime.now()
    
    # Start date: 60 days ago (Yahoo Finance maximum for 5-minute data)
    # Note: If you want more historical data, you'll need a different API
    start_date = end_date - timedelta(days=60)
    
    print("=" * 70)
    print("Multi-Exchange Forex Dataset Generator")
    print("=" * 70)
    print(f"Data Source: Yahoo Finance (yfinance)")
    print(f"⚠️  LIMITATION: Yahoo Finance provides max 60 days of 5-minute data")
    print(f"Start Date: {start_date} (60 days ago)")
    print(f"End Date: {end_date}")
    print(f"Duration: {end_date - start_date} (60 days)")
    print(f"Symbols to process: {list(FOREX_SYMBOLS.keys())}")
    print("=" * 70)
    
    # Generate datasets for each forex pair
    results = {}
    for symbol in FOREX_SYMBOLS.keys():
        try:
            success = generate_dataset_for_symbol(symbol, start_date, end_date)
            results[symbol] = "Success" if success else "Failed"
        except Exception as e:
            print(f"\n❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = f"Error: {str(e)}"
    
    # Summary
    print("\n" + "=" * 70)
    print("Dataset Generation Summary")
    print("=" * 70)
    for symbol, status in results.items():
        status_icon = "✓" if status == "Success" else "✗"
        print(f"{status_icon} {symbol}: {status}")
    print("=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

