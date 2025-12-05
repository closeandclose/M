#!/usr/bin/env python3
"""
MANTIS Miner - V2 Payload Implementation
Generates multi-asset embeddings and V2 encrypted payloads.
Based on MINER_GUIDE.md, PAYLOAD_MIGRATION_GUIDE.md, and README.md
"""

import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import boto3
import bittensor as bt
from dotenv import load_dotenv

# Import MANTIS modules for proper V2 encryption
import config
from generate_and_encrypt import generate_v2

# Import prediction models
try:
    from binary_eth_1h import BinaryETH1HPredictor
    HAS_ETH_PREDICTOR = True
except ImportError:
    HAS_ETH_PREDICTOR = False
    print("⚠️ Warning: binary_eth_1h.py not found, using random predictions for ETH")

try:
    from binary_forex_1h import (
        generate_eurusd_binary_prediction,
        generate_gbpusd_binary_prediction,
        generate_cadusd_binary_prediction,
        generate_nzdusd_binary_prediction,
        generate_chfusd_binary_prediction,
        generate_xauusd_binary_prediction,
        generate_xagusd_binary_prediction,
    )
    HAS_FOREX_PREDICTOR = True
except ImportError:
    HAS_FOREX_PREDICTOR = False
    print("⚠️ Warning: binary_forex_1h.py not found, using random predictions for forex pairs")

# Load environment variables
load_dotenv(override=True)

# Constants
LOCK_TIME_SECONDS = 30
MINING_INTERVAL = 45  # Upload every 45 seconds


def generate_binary_embedding(dim: int = 2) -> List[float]:
    """
    Generate random embedding for binary prediction challenge.
    Values in range [-1, 1].
    """
    return [random.uniform(-1, 1) for _ in range(dim)]


def generate_lbfgs_embedding() -> List[float]:
    """
    Generate embedding for LBFGS challenge (17-dim).
    
    Layout:
    - [0:5]   p[0..4]: 5-bucket class probabilities (must sum to 1)
              0: z ≤ -2σ, 1: -2σ < z < -1σ, 2: -1σ ≤ z ≤ 1σ, 3: 1σ < z < 2σ, 4: z ≥ 2σ
    - [5:8]   Q(c=0): opposite-move (UP) probs at 0.5σ, 1.0σ, 2.0σ
    - [8:11]  Q(c=1): opposite-move (UP) probs
    - [11:14] Q(c=3): opposite-move (DOWN) probs
    - [14:17] Q(c=4): opposite-move (DOWN) probs
    """
    # p[0:5]: Dirichlet distribution ensures sum=1
    p = np.random.dirichlet(np.ones(5)).tolist()
    
    # Q[5:17]: 12 probability values in (0, 1)
    # Using Beta(2,2) for realistic probability distribution
    q_values = []
    for _ in range(12):
        q = np.random.beta(2, 2)
        q = max(1e-6, min(1 - 1e-6, q))  # Clamp to valid range
        q_values.append(q)
    
    embedding = p + q_values
    return embedding


def generate_multi_asset_embeddings(hotkey: str = None) -> Dict[str, List[float]]:
    """
    Generate embeddings for ALL challenges defined in config.CHALLENGES.
    Returns a dictionary with ticker keys and hotkey (matching PAYLOAD_MIGRATION_GUIDE.md plaintext structure).
    
    Challenges:
    - ETH-1H-BINARY (dim=2) - Uses actual prediction model
    - ETH-LBFGS (dim=17) - Random
    - BTC-LBFGS-6H (dim=17) - Random
    - EURUSD-1H-BINARY (dim=2) - Uses actual prediction model
    - GBPUSD-1H-BINARY (dim=2) - Uses actual prediction model
    - CADUSD-1H-BINARY (dim=2) - Uses actual prediction model
    - NZDUSD-1H-BINARY (dim=2) - Uses actual prediction model
    - CHFUSD-1H-BINARY (dim=2) - Uses actual prediction model
    - XAUUSD-1H-BINARY (dim=2) - Uses actual prediction model
    - XAGUSD-1H-BINARY (dim=2) - Uses actual prediction model
    
    Returns:
        Dictionary with structure: {"ETH": [...], "EURUSD": [...], ..., "hotkey": "<SS58>"}
    """
    embeddings_dict = {}
    eth_predictor = None
    
    # Initialize ETH predictor if available
    if HAS_ETH_PREDICTOR:
        try:
            eth_predictor = BinaryETH1HPredictor()
        except Exception as e:
            print(f"⚠️ Failed to initialize ETH predictor: {e}")
            eth_predictor = None
    
    # Forex ticker mapping to prediction functions
    forex_prediction_functions = {
        "EURUSD": generate_eurusd_binary_prediction,
        "GBPUSD": generate_gbpusd_binary_prediction,
        "CADUSD": generate_cadusd_binary_prediction,
        "NZDUSD": generate_nzdusd_binary_prediction,
        "CHFUSD": generate_chfusd_binary_prediction,
        "XAUUSD": generate_xauusd_binary_prediction,
        "XAGUSD": generate_xagusd_binary_prediction,
    }
    
    for challenge in config.CHALLENGES:
        dim = challenge["dim"]
        loss_func = challenge.get("loss_func", "binary")
        ticker = challenge["ticker"]
        
        if loss_func == "lbfgs" and dim == 17:
            emb = generate_lbfgs_embedding()
            print(f"  {ticker}: LBFGS (17-dim), p_sum={sum(emb[:5]):.4f}")
        else:
            # Use actual model for ETH-1H-BINARY
            if ticker == "ETH" and loss_func == "binary" and eth_predictor:
                try:
                    emb = eth_predictor.predict("ETHUSDT")
                    print(f"  {ticker}: Binary ({dim}-dim) [MODEL], values={[f'{v:.3f}' for v in emb]}")
                except Exception as e:
                    print(f"  ⚠️ ETH prediction failed: {e}, using random")
                    emb = generate_binary_embedding(dim)
                    print(f"  {ticker}: Binary ({dim}-dim) [RANDOM], values={[f'{v:.3f}' for v in emb]}")
            # Use actual model for Forex pairs
            elif ticker in forex_prediction_functions and HAS_FOREX_PREDICTOR:
                try:
                    pred_func = forex_prediction_functions[ticker]
                    emb = pred_func()
                    print(f"  {ticker}: Binary ({dim}-dim) [MODEL], values={[f'{v:.3f}' for v in emb]}")
                except Exception as e:
                    print(f"  ⚠️ {ticker} prediction failed: {e}, using random")
                    emb = generate_binary_embedding(dim)
                    print(f"  {ticker}: Binary ({dim}-dim) [RANDOM], values={[f'{v:.3f}' for v in emb]}")
            else:
                # Use random for other assets
                emb = generate_binary_embedding(dim)
                print(f"  {ticker}: Binary ({dim}-dim) [RANDOM], values={[f'{v:.3f}' for v in emb]}")
        
        embeddings_dict[ticker] = emb
    
    # Add hotkey at the end (as per PAYLOAD_MIGRATION_GUIDE.md line 64)
    if hotkey:
        embeddings_dict["hotkey"] = hotkey
    
    return embeddings_dict


def create_v2_payload(hotkey: str, file_path: str) -> bool:
    """
    Create V2 encrypted payload with proper structure.
    
    V2 payload structure (matches PAYLOAD_MIGRATION_GUIDE.md):
    - v, round, hk, owner_pk, C, W_owner, W_time, binding, alg
    - Plaintext inside C.ct: {"ETH": [...], ..., "hotkey": "<SS58>"}
    """
    try:
        print(f"Generating V2 payload for hotkey: {hotkey}")
        print(f"Generating embeddings for {len(config.CHALLENGES)} challenges:")
        
        # Generate embeddings for all challenges (returns dict with ticker keys + hotkey)
        embeddings_dict = generate_multi_asset_embeddings(hotkey=hotkey)
        
        # Display embeddings structure
        print(f"\n📊 Embeddings in payload:")
        print(f"   Total keys: {len(embeddings_dict)}")
        print(f"   Tickers: {[k for k in embeddings_dict.keys() if k != 'hotkey']}")
        print(f"   Hotkey field: {embeddings_dict.get('hotkey', 'MISSING')}")
        print(f"\n   Embedding values:")
        for ticker, emb in embeddings_dict.items():
            if ticker == "hotkey":
                print(f"     {ticker}: {emb}")
            else:
                if isinstance(emb, list) and len(emb) > 0:
                    if len(emb) <= 5:
                        # Show all values for small embeddings
                        print(f"     {ticker}: {[f'{v:.4f}' for v in emb]}")
                    else:
                        # Show first few and last few for large embeddings (LBFGS)
                        print(f"     {ticker}: [{', '.join([f'{v:.4f}' for v in emb[:3]])} ... {', '.join([f'{v:.4f}' for v in emb[-3:]])}] (len={len(emb)})")
                else:
                    print(f"     {ticker}: {emb}")
        
        # Convert to JSON string for payload_text (matching PAYLOAD_MIGRATION_GUIDE.md structure)
        payload_text = json.dumps(embeddings_dict, ensure_ascii=False, separators=(",", ":"))
        print(f"\n   Payload text size: {len(payload_text)} bytes")
        
        # Generate V2 encrypted payload using MANTIS's generate_v2
        # This function uses Drand API for timelock encryption
        print(f"\n🔐 [Drand Integration] Generating V2 payload with timelock encryption")
        print(f"   Drand API: {config.DRAND_API}, Beacon: {config.DRAND_BEACON_ID}, Lock: {LOCK_TIME_SECONDS}s")
        
        payload = generate_v2(
            hotkey=hotkey,
            lock_seconds=LOCK_TIME_SECONDS,
            owner_pk_hex=config.OWNER_HPKE_PUBLIC_KEY_HEX,
            payload_text=payload_text,
            embeddings=[],  # Empty since we're using payload_text
        )
        
        print(f"✅ V2 payload generated: v={payload['v']}, round={payload['round']}, alg={payload['alg']}")
        
        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        
        print(f"✅ Payload saved to: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating V2 payload: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_to_r2(bucket_name: str, object_key: str, file_path: str) -> bool:
    """Upload payload file to R2 bucket"""
    try:
        account_id = os.environ.get('R2_ACCOUNT_ID')
        access_key = os.environ.get('R2_WRITE_ACCESS_KEY_ID')
        secret_key = os.environ.get('R2_WRITE_SECRET_ACCESS_KEY')
        
        if not all([account_id, access_key, secret_key]):
            print("❌ Missing R2 credentials in environment variables")
            return False
        
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
        
        s3.upload_file(file_path, bucket_name, object_key)
        print(f"✅ Uploaded to R2: s3://{bucket_name}/{object_key}")
        return True
    except Exception as e:
        print(f"❌ Error uploading to R2: {e}")
        return False


def commit_url_to_subnet(wallet_name: str, hotkey_name: str, r2_public_hash: str, hotkey: str) -> bool:
    """
    Commit the R2 URL to the subnet (run once).
    
    According to PAYLOAD_MIGRATION_GUIDE.md:
    - URL must point to Cloudflare R2 (*.r2.dev or *.r2.cloudflarestorage.com)
    - Object key must be exactly your hotkey (no directories)
    """
    try:
        print(f"\n📝 Committing URL to subnet (network: finney, netuid: {config.NETUID})...")
        
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        subtensor = bt.subtensor(network="finney")  # Use "mainnet" for production
        
        r2_public_url = f"https://pub-{r2_public_hash}.r2.dev/{hotkey}"
        subtensor.commit(wallet=wallet, netuid=config.NETUID, data=r2_public_url)
        print(f"✅ Successfully committed URL to subnet: {r2_public_url}")
        return True
    except Exception as e:
        print(f"❌ Error committing URL to subnet: {e}")
        import traceback
        traceback.print_exc()
        return False


def mining_cycle(
    hotkey: str,
    wallet_name: str,
    hotkey_name: str,
    r2_public_hash: str,
    bucket_name: str,
    object_key: str,
    file_path: str,
    cycle_num: int = 1,
    commit_url: bool = True
) -> bool:
    """
    Complete mining cycle with V2 payload format.
    
    Steps (from MINER_GUIDE.md):
    1. Generate multi-asset embeddings (Step 1)
    2. Create V2 encrypted payload (Step 2)
    3. Upload to R2 bucket (Step 3)
    4. Commit URL to subnet (Step 5 - only once if commit_url=True)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"Mining Cycle #{cycle_num} | {timestamp}")
    print(f"Wallet: {wallet_name} ({hotkey_name})")
    print(f"{'='*60}")
    
    # Step 1 & 2: Generate embeddings and create V2 payload
    if not create_v2_payload(hotkey, file_path):
        print("❌ Failed to create V2 payload")
        return False
    
    # Step 3: Upload to R2
    if not upload_to_r2(bucket_name, object_key, file_path):
        print("❌ Failed to upload to R2")
        return False

    # Step 5: Commit URL to subnet (only if requested - usually done once)
    if commit_url:
        commit_success = commit_url_to_subnet(wallet_name, hotkey_name, r2_public_hash, hotkey)
        if not commit_success:
            print("⚠️  Warning: Failed to commit URL to subnet (continuing anyway)")
    
    print(f"✅ Mining cycle #{cycle_num} completed!")
    return True


def check_configuration(hotkey: str, wallet_name: str, hotkey_name: str, r2_public_hash: str) -> bool:
    """Check if all required configuration is set."""
    print(f"--- Configuration Check for {wallet_name} ({hotkey_name}) ---")
    
    # Check environment variables
    required_env_vars = ['R2_ACCOUNT_ID', 'R2_WRITE_ACCESS_KEY_ID', 'R2_WRITE_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set them in your .env file")
        return False
    
    # Check configuration values
    config_issues = []
    
    if not hotkey or len(hotkey) < 10:
        config_issues.append("HOTKEY appears to be invalid or too short")
    
    if not wallet_name or len(wallet_name) < 1:
        config_issues.append("WALLET_NAME appears to be invalid")
    
    if not hotkey_name or len(hotkey_name) < 1:
        config_issues.append("HOTKEY_NAME appears to be invalid")
    
    if not r2_public_hash or len(r2_public_hash) < 10:
        config_issues.append("R2_PUBLIC_HASH appears to be invalid or too short")
    
    if config_issues:
        print("❌ Configuration issues found:")
        for issue in config_issues:
            print(f"   - {issue}")
        return False
    
    print(f"✅ Configuration looks good for {wallet_name}!")
    return True


def load_miner_data(filepath: str = "hotkeydata.json") -> Optional[Dict]:
    """Load single miner configuration from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filepath}: {e}")
        return None


def main():
    """Main function - Single Miner with continuous loop"""
    print("=" * 60)
    print("MANTIS Miner - V2 Payload Generator")
    print(f"Upload interval: {MINING_INTERVAL} seconds")
    print("=" * 60)
    
    # Load miner data from hotkeydata.json
    miner = load_miner_data("hotkeydata.json")
    
    if not miner:
        print("❌ No miner data found. Please create hotkeydata.json")
        print("\nExample format:")
        print(json.dumps({
            "hotkey": "5G...(ss58_address)",
            "wallet_name": "your_wallet",
            "hotkey_name": "your_hotkey",
            "r2_public_hash": "your_r2_hash",
            "bucket_name": "your_bucket",
            "object_key": "5G...(same_as_hotkey)",
            "commit_url_once": True  # Set to True for first run to commit URL
        }, indent=2))
        return
    
    # Extract miner configuration
    HOTKEY = miner.get('hotkey', '')
    WALLET_NAME = miner.get('wallet_name', '')
    HOTKEY_NAME = miner.get('hotkey_name', '')
    R2_PUBLIC_HASH = miner.get('r2_public_hash', '')
    bucket_name = miner.get('bucket_name', '')
    object_key = miner.get('object_key', HOTKEY)
    file_path = object_key
    # Default to True if not specified - commit URL on first cycle
    commit_url_once = miner.get('commit_url_once', True)
    
    print(f"Wallet: {WALLET_NAME}")
    print(f"Hotkey: {HOTKEY_NAME}")
    print(f"Address: {HOTKEY}")
    print(f"Bucket: {bucket_name}")
    print(f"Object Key: {object_key}")
    
    # Check configuration
    if not check_configuration(HOTKEY, WALLET_NAME, HOTKEY_NAME, R2_PUBLIC_HASH):
        print("❌ Configuration check failed!")
        return
    
    # Continuous mining loop
    cycle_num = 0
    print(f"\n🔄 Starting continuous mining loop (every {MINING_INTERVAL}s)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            cycle_num += 1
            
            # Run mining cycle
            # commit_url=True only on first cycle if commit_url_once is True
            should_commit = commit_url_once and cycle_num == 1
            if should_commit:
                print(f"📝 Will commit URL to subnet on cycle #{cycle_num}")
            
            success = mining_cycle(
                HOTKEY, WALLET_NAME, HOTKEY_NAME, R2_PUBLIC_HASH,
                bucket_name, object_key, file_path, cycle_num,
                commit_url=should_commit
            )
            
            # After first commit, set commit_url_once to False to prevent re-committing
            if should_commit and success:
                commit_url_once = False
                print("✅ URL committed. Will not commit again in future cycles.")
            
            if success:
                print(f"✅ Cycle #{cycle_num} done. Next upload in {MINING_INTERVAL}s...")
            else:
                print(f"❌ Cycle #{cycle_num} failed. Retrying in {MINING_INTERVAL}s...")
            
            # Wait for next cycle
            time.sleep(MINING_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print(f"🛑 Miner stopped after {cycle_num} cycles")
        print("=" * 60)


if __name__ == "__main__":
    main()

