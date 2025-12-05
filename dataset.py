#!/usr/bin/env python3
"""
Dataset Generator for ETH Price Data
Generates a sliding window dataset with 5-day windows (7200 minutes) and 1-hour future price.
"""

import numpy as np
import torch
import requests
from datetime import datetime, timedelta
from typing import List
import time
import os
from pathlib import Path


def fetch_eth_klines(start_time: datetime, end_time: datetime) -> List[List]:
    """
    Fetch ETH 1-minute klines from Binance US API.
    
    Args:
        start_time: Start datetime
        end_time: End datetime
    
    Returns:
        List of klines (each kline is a list with close price at index 4)
    """
    url = "https://api.binance.us/api/v3/klines"
    all_klines = []
    
    current_start = start_time
    batch_size = 1000  # Binance allows up to 1000 klines per request
    
    print(f"Fetching ETH price data from {start_time} to {end_time}...")
    
    while current_start < end_time:
        current_end = min(current_start + timedelta(minutes=batch_size - 1), end_time)
        
        params = {
            "symbol": "ETHUSDT",
            "interval": "1m",
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(current_end.timestamp() * 1000),
            "limit": batch_size
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Update start time for next batch (add 1ms to avoid duplicate)
            last_time = klines[-1][0]  # First element is timestamp
            current_start = datetime.fromtimestamp(last_time / 1000) + timedelta(milliseconds=1)
            
            print(f"  Fetched {len(all_klines)} klines so far... (last: {datetime.fromtimestamp(last_time / 1000)})")
            
            # Rate limiting - Binance has rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching klines: {e}")
            time.sleep(1)
            continue
    
    print(f"Total klines fetched: {len(all_klines)}")
    return all_klines


def create_and_save_dataset(klines: List[List], filename: str = "eth_dataset.data"):
    """
    Create dataset structure and save directly to .data file.
    - Each big array contains: [[x0], [y0]] where:
      - x0: 7200 ETH prices as PyTorch tensor (5 days, 1 per minute)
      - y0: price 1 hour (60 minutes) later as PyTorch tensor
    - Each big array starts 1 minute later than previous
    
    Args:
        klines: List of klines from Binance API
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
    if len(klines) < 7200 + 60:
        print(f"Error: Not enough data. Need at least {7200 + 60} klines, got {len(klines)}")
        return False
    
    # Extract close prices (index 4 in kline array)
    prices = [float(kline[4]) for kline in klines]
    
    # Calculate how many windows we can create
    window_size = 7200
    future_offset = 60  # 1 hour = 60 minutes
    
    max_windows = len(prices) - window_size - future_offset + 1
    
    if max_windows <= 0:
        print(f"Error: Not enough data to create even one window.")
        return False
    
    print(f"Creating {max_windows} sliding windows...")
    
    # Collect all windows in memory and save directly to .data file
    all_windows = []
    
    for i in range(max_windows):
        # Extract 7200 consecutive prices and convert to PyTorch tensor
        price_window = torch.tensor(prices[i:i + window_size], dtype=torch.float64)
        
        # Get price 1 hour (60 minutes) after the last price in window
        future_price_index = i + window_size + future_offset - 1
        
        # Validate future price index (defensive check)
        if future_price_index >= len(prices):
            print(f"ERROR: Future price index {future_price_index} exceeds prices length {len(prices)}")
            return False
        
        future_price = torch.tensor(float(prices[future_price_index]), dtype=torch.float64)
        
        # Create big array: [[x0], [y0]] format
        # x0: PyTorch tensor with 7200 prices
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


def main():
    """Main function to generate dataset from 2025-01-01 until now."""
    # Start date: 2025-01-01 00:00:00 UTC
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    
    # End date: now
    end_date = datetime.now()
    
    print("=" * 70)
    print("ETH Dataset Generator")
    print("=" * 70)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Duration: {end_date - start_date}")
    print("=" * 70)
    
    # Fetch klines
    klines = fetch_eth_klines(start_date, end_date)
    
    if not klines:
        print("No klines fetched. Exiting.")
        return
    
    # Create and save dataset directly to .data file
    # Use absolute path to ensure files are saved in Dataset directory
    script_dir = Path(__file__).parent.absolute()
    output_file = str(script_dir / "eth_dataset.data")
    success = create_and_save_dataset(klines, output_file)
    
    if not success:
        print("Dataset creation failed. Exiting.")
        return
    
    # Verify saved dataset
    print("\nVerifying saved dataset...")
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
            print(f"  First window prices range: {float(torch.min(x0)):.2f} - {float(torch.max(x0)):.2f}")
            print(f"  First window future price: {float(y0):.2f}")
    except Exception as e:
        print(f"Error verifying dataset: {e}")
    
    print("=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

