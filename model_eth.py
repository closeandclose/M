#!/usr/bin/env python3
"""
ETH Price Prediction Model using LSTM
Trains an LSTM model to predict ETH price 1 hour later from 7200 prices (5 days at 1-minute intervals).

Input: 7200 ETH prices (1-minute interval for 5 days)
Output: 1 ETH price (1 hour later from last price in input)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import random
from typing import Tuple, List, Optional, Dict
import logging
# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not available. Progress bars will be disabled. Install with: pip install tqdm")
    # Fallback: create a dummy tqdm class that acts like tqdm but does nothing
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else range(kwargs.get('total', 0))
            self.desc = kwargs.get('desc', '')
            self.unit = kwargs.get('unit', 'it')
        
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def set_postfix(self, **kwargs):
            pass
        
        def close(self):
            pass
        
        def update(self, n=1):
            pass


def normalize_min_max(x: torch.Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[torch.Tensor, float, float]:
    """
    Normalize tensor using min-max normalization.
    
    Args:
        x: Input tensor
        min_val: Optional minimum value (if None, computed from x)
        max_val: Optional maximum value (if None, computed from x)
    
    Returns:
        Normalized tensor, min_val, max_val
    """
    if min_val is None:
        min_val = float(x.min())
    if max_val is None:
        max_val = float(x.max())
    
    if max_val - min_val == 0:
        return x, min_val, max_val
    
    normalized = (x - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize_min_max(normalized: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Denormalize tensor using min-max normalization.
    
    Args:
        normalized: Normalized tensor
        min_val: Minimum value used for normalization
        max_val: Maximum value used for normalization
    
    Returns:
        Denormalized tensor
    """
    return normalized * (max_val - min_val) + min_val


def normalize_z_score(x: torch.Tensor, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[torch.Tensor, float, float]:
    """
    Normalize tensor using z-score (standardization).
    
    Args:
        x: Input tensor
        mean: Optional mean value (if None, computed from x)
        std: Optional standard deviation (if None, computed from x)
    
    Returns:
        Normalized tensor, mean, std
    """
    if mean is None:
        mean = float(x.mean())
    if std is None:
        std = float(x.std())
    
    if std == 0:
        return x, mean, std
    
    normalized = (x - mean) / std
    return normalized, mean, std


def denormalize_z_score(normalized: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Denormalize tensor using z-score normalization.
    
    Args:
        normalized: Normalized tensor
        mean: Mean value used for normalization
        std: Standard deviation used for normalization
    
    Returns:
        Denormalized tensor
    """
    return normalized * std + mean


class ETHLSTMModel(nn.Module):
    """
    LSTM model for ETH price prediction.
    Predicts 1-hour-ahead price from 7200-minute sequence.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of features per time step (1 for price only)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
        """
        super(ETHLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False  # (seq_len, batch, features)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: single price value
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights for LSTM
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights for LSTM
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Bias initialization
                param.data.fill_(0)
                # Set forget gate bias to 1 to help with gradient flow
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
            elif 'weight' in name and len(param.shape) >= 2:
                # Linear layer weights
                nn.init.xavier_uniform_(param.data)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (sequence_length, batch_size, input_size)
               For our case: (7200, batch_size, 1)
        
        Returns:
            Predicted price: (batch_size,)
        """
        # Initialize hidden and cell states (must match input dtype)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=x.dtype).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=x.dtype).to(x.device)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Take the last output (final hidden state)
        last_output = lstm_out[-1]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)  # (batch_size, 1)
        
        return out.squeeze(-1)  # (batch_size,)


def load_eth_dataset(
    dataset_path: str, 
    normalize: bool = True, 
    norm_method: str = 'min_max'
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict]:
    """
    Load ETH dataset from .data file with optional normalization.
    
    Args:
        dataset_path: Path to eth_dataset.data file
        normalize: Whether to normalize the data
        norm_method: Normalization method ('min_max' or 'z_score')
    
    Returns:
        Tuple of:
        - List of (x, y, stats) tuples where:
          - x: Normalized input tensor of shape (7200,) - 7200 prices
          - y: Normalized target tensor (scalar) - 1 hour later price
          - stats: Dict with normalization statistics {'min': float, 'max': float} or {'mean': float, 'std': float}
        - Global statistics dict for logging
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Get absolute path
    if not Path(dataset_path).is_absolute():
        dataset_path = Path(__file__).parent.absolute() / dataset_path
    
    # Load data
    logger.info("Loading .data file...")
    data = torch.load(str(dataset_path))
    logger.info(f"Loaded dataset with {len(data)} windows")
    
    # Extract x and y pairs with progress bar
    # Convert to float32 for LSTM compatibility (datasets are stored as float64)
    logger.info("Extracting and converting data...")
    dataset = []
    
    # Collect all prices for global statistics
    all_x_prices = []
    all_y_prices = []
    
    pbar = tqdm(data, desc="Loading Dataset", unit="window") if HAS_TQDM else data
    for window in pbar:
        if len(window) >= 2:
            x = window[0][0].float()  # Shape: (7200,), convert to float32
            y = window[1][0].float()  # Shape: () - scalar, convert to float32
            all_x_prices.extend(x.tolist())
            all_y_prices.append(float(y))
            dataset.append((x, y))
            
            # Update progress for tqdm
            if HAS_TQDM and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'extracted': len(dataset)})
    
    if HAS_TQDM:
        pbar.close()
    
    logger.info(f"✅ Extracted {len(dataset)} training examples")
    if len(dataset) > 0:
        logger.info(f"Input shape: {dataset[0][0].shape}, Output shape: {dataset[0][1].shape}")
    
    # Calculate global statistics
    all_x_tensor = torch.tensor(all_x_prices, dtype=torch.float32)
    all_y_tensor = torch.tensor(all_y_prices, dtype=torch.float32)
    
    global_stats = {
        'x_min': float(all_x_tensor.min()),
        'x_max': float(all_x_tensor.max()),
        'x_mean': float(all_x_tensor.mean()),
        'x_std': float(all_x_tensor.std()),
        'y_min': float(all_y_tensor.min()),
        'y_max': float(all_y_tensor.max()),
        'y_mean': float(all_y_tensor.mean()),
        'y_std': float(all_y_tensor.std()),
    }
    
    # Apply normalization if requested
    if normalize:
        logger.info(f"Applying {norm_method} normalization...")
        normalized_dataset = []
        
        pbar = tqdm(dataset, desc="Normalizing Data", unit="window") if HAS_TQDM else dataset
        for x, y in pbar:
            if norm_method == 'min_max':
                # Normalize x and y using x's statistics (preserves relative relationship)
                x_min = float(x.min())
                x_max = float(x.max())
                if x_max - x_min > 0:
                    x_norm = (x - x_min) / (x_max - x_min)
                    y_norm = (y - x_min) / (x_max - x_min)
                    stats = {'min': x_min, 'max': x_max, 'method': 'min_max'}
                else:
                    x_norm = x
                    y_norm = y
                    stats = {'min': x_min, 'max': x_max, 'method': 'min_max'}
            elif norm_method == 'z_score':
                # Normalize using z-score
                x_mean = float(x.mean())
                x_std = float(x.std())
                if x_std > 0:
                    x_norm = (x - x_mean) / x_std
                    y_norm = (y - x_mean) / x_std
                    stats = {'mean': x_mean, 'std': x_std, 'method': 'z_score'}
                else:
                    x_norm = x
                    y_norm = y
                    stats = {'mean': x_mean, 'std': x_std, 'method': 'z_score'}
            else:
                raise ValueError(f"Unknown normalization method: {norm_method}")
            
            normalized_dataset.append((x_norm, y_norm, stats))
            
            if HAS_TQDM and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'normalized': len(normalized_dataset)})
        
        if HAS_TQDM:
            pbar.close()
        
        logger.info(f"✅ Normalized {len(normalized_dataset)} examples")
        return normalized_dataset, global_stats
    else:
        # Return data without normalization, with dummy stats
        dataset_with_stats = [(x, y, {'method': 'none'}) for x, y in dataset]
        return dataset_with_stats, global_stats


def prepare_batch(
    x_batch: List[torch.Tensor], 
    y_batch: List[torch.Tensor],
    stats_batch: Optional[List[Dict]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
    """
    Prepare batch for LSTM input.
    
    Args:
        x_batch: List of input tensors, each of shape (7200,)
        y_batch: List of target scalars
        stats_batch: Optional list of normalization statistics dicts
    
    Returns:
        x: Tensor of shape (seq_len, batch_size, input_size) = (7200, batch_size, 1)
        y: Tensor of shape (batch_size,)
        stats_batch: Optional list of normalization statistics
    """
    # Stack x_batch: (batch_size, 7200) -> transpose -> (7200, batch_size)
    # Then add feature dimension: (7200, batch_size, 1)
    # Convert to float32 for LSTM compatibility
    x_stacked = torch.stack(x_batch).float()  # (batch_size, 7200), convert to float32
    x_transposed = x_stacked.transpose(0, 1)  # (7200, batch_size)
    x = x_transposed.unsqueeze(-1)  # (7200, batch_size, 1)
    
    # Stack y_batch: (batch_size,)
    # Convert to float32 for consistency
    y = torch.stack(y_batch).float()  # (batch_size,), convert to float32
    
    return x, y, stats_batch


def train_model(
    model: nn.Module,
    train_data: List[Tuple],
    val_data: List[Tuple],
    batch_size: int = 32,
    learning_rate: float = 0.001,
    epochs: int = 100,
    device: str = 'cpu',
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-6
):
    """
    Train the LSTM model with early stopping.
    
    Args:
        model: LSTM model
        train_data: Training dataset (list of (x, y, stats) tuples)
        val_data: Validation dataset (list of (x, y, stats) tuples)
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of epochs
        device: Device to train on ('cpu' or 'cuda')
        early_stopping_patience: Number of epochs to wait before stopping
        early_stopping_min_delta: Minimum change to qualify as an improvement
    
    Returns:
        Tuple of (train_losses, val_losses, best_epoch)
    """
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # Regression problem
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    logger.info(f"Starting training on device: {device}")
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
    logger.info(f"Early stopping patience: {early_stopping_patience} epochs")
    logger.info("=" * 70)
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        num_batches = 0
        
        # Shuffle training data
        random.shuffle(train_data)
        
        # Training batches with progress bar
        train_pbar = tqdm(
            range(0, len(train_data), batch_size),
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            leave=False,
            unit="batch"
        )
        
        for i in train_pbar:
            batch = train_data[i:i + batch_size]
            x_batch = [item[0] for item in batch]
            y_batch = [item[1] for item in batch]
            stats_batch = [item[2] for item in batch] if len(batch[0]) > 2 else None
            
            # Prepare batch
            x, y, _ = prepare_batch(x_batch, y_batch, stats_batch)
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            mae = torch.mean(torch.abs(y_pred - y))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_mae_sum += mae.item()
            num_batches += 1
            
            # Update progress bar with current metrics
            if HAS_TQDM and hasattr(train_pbar, 'set_postfix'):
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'mae': f'{mae.item():.6f}'
                })
        
        avg_train_loss = train_loss_sum / num_batches
        avg_train_mae = train_mae_sum / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase with progress bar
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        num_val_batches = 0
        
        val_pbar = tqdm(
            range(0, len(val_data), batch_size),
            desc=f"Epoch {epoch+1}/{epochs} [Val]",
            leave=False,
            unit="batch"
        )
        
        with torch.no_grad():
            for i in val_pbar:
                batch = val_data[i:i + batch_size]
                x_batch = [item[0] for item in batch]
                y_batch = [item[1] for item in batch]
                stats_batch = [item[2] for item in batch] if len(batch[0]) > 2 else None
                
                x, y, _ = prepare_batch(x_batch, y_batch, stats_batch)
                x = x.to(device)
                y = y.to(device)
                
                y_pred = model(x)
                loss = criterion(y_pred, y)
                mae = torch.mean(torch.abs(y_pred - y))
                
                val_loss_sum += loss.item()
                val_mae_sum += mae.item()
                num_val_batches += 1
                
                # Update progress bar with current metrics
                if HAS_TQDM and hasattr(val_pbar, 'set_postfix'):
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'mae': f'{mae.item():.6f}'
                    })
        
        avg_val_loss = val_loss_sum / num_val_batches
        avg_val_mae = val_mae_sum / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"✨ New best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = old_lr != new_lr
        
        # Update epoch progress bar
        if HAS_TQDM and hasattr(epoch_pbar, 'set_postfix'):
            epoch_pbar.set_postfix({
                'Train Loss': f'{avg_train_loss:.6f}',
                'Train MAE': f'{avg_train_mae:.6f}',
                'Val Loss': f'{avg_val_loss:.6f}',
                'Val MAE': f'{avg_val_mae:.6f}',
                'Best Val': f'{best_val_loss:.6f}',
                'Patience': f'{patience_counter}/{early_stopping_patience}',
                'LR': f'{new_lr:.6f}'
            })
        
        # Detailed logging every 10 epochs or first epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch [{epoch+1}/{epochs}] Summary")
            logger.info(f"{'='*70}")
            logger.info(f"Training:")
            logger.info(f"  Loss (MSE): {avg_train_loss:.6f}")
            logger.info(f"  MAE:        {avg_train_mae:.6f}")
            logger.info(f"Validation:")
            logger.info(f"  Loss (MSE): {avg_val_loss:.6f}")
            logger.info(f"  MAE:        {avg_val_mae:.6f}")
            logger.info(f"  Best Loss:  {best_val_loss:.6f} (epoch {best_epoch})")
            logger.info(f"  Patience:   {patience_counter}/{early_stopping_patience}")
            logger.info(f"Learning Rate: {new_lr:.6f}")
            if lr_changed:
                logger.info(f"  ⚠️  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            logger.info(f"{'='*70}\n")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"\n⏹️  Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
            logger.info(f"Restoring best model from epoch {best_epoch} (val_loss: {best_val_loss:.6f})")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    if HAS_TQDM:
        epoch_pbar.close()
    
    if best_model_state is not None and patience_counter < early_stopping_patience:
        logger.info(f"Training completed! Best model at epoch {best_epoch} (val_loss: {best_val_loss:.6f})")
    else:
        logger.info("✅ Training completed!")
    
    return train_losses, val_losses, best_epoch


def evaluate_model(
    model: nn.Module,
    test_data: List[Tuple],
    batch_size: int = 32,
    device: str = 'cpu',
    denormalize: bool = True
) -> dict:
    """
    Evaluate model on test data with optional denormalization.
    
    Args:
        model: Trained model
        test_data: Test dataset (list of (x, y, stats) tuples)
        batch_size: Batch size
        device: Device to evaluate on
        denormalize: Whether to denormalize predictions and targets for metrics
    
    Returns:
        Dictionary with evaluation metrics (both normalized and denormalized if applicable)
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    
    all_predictions_norm = []
    all_targets_norm = []
    all_predictions_denorm = []
    all_targets_denorm = []
    total_loss = 0.0
    num_batches = 0
    
    logger.info("Evaluating on test set...")
    test_pbar = tqdm(
        range(0, len(test_data), batch_size),
        desc="Testing Progress",
        unit="batch"
    )
    
    with torch.no_grad():
        for i in test_pbar:
            batch = test_data[i:i + batch_size]
            x_batch = [item[0] for item in batch]
            y_batch = [item[1] for item in batch]
            stats_batch = [item[2] for item in batch] if len(batch[0]) > 2 else None
            
            x, y, _ = prepare_batch(x_batch, y_batch, stats_batch)
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store normalized predictions and targets
            predictions_norm = y_pred.cpu().numpy()
            targets_norm = y.cpu().numpy()
            all_predictions_norm.extend(predictions_norm)
            all_targets_norm.extend(targets_norm)
            
            # Denormalize if requested and stats are available
            if denormalize and stats_batch:
                for j, (pred, tgt, stats) in enumerate(zip(predictions_norm, targets_norm, stats_batch)):
                    if stats.get('method') == 'min_max':
                        pred_denorm = denormalize_min_max(
                            torch.tensor(pred), stats['min'], stats['max']
                        ).item()
                        tgt_denorm = denormalize_min_max(
                            torch.tensor(tgt), stats['min'], stats['max']
                        ).item()
                    elif stats.get('method') == 'z_score':
                        pred_denorm = denormalize_z_score(
                            torch.tensor(pred), stats['mean'], stats['std']
                        ).item()
                        tgt_denorm = denormalize_z_score(
                            torch.tensor(tgt), stats['mean'], stats['std']
                        ).item()
                    else:
                        pred_denorm = pred
                        tgt_denorm = tgt
                    all_predictions_denorm.append(pred_denorm)
                    all_targets_denorm.append(tgt_denorm)
            
            # Update progress bar
            if HAS_TQDM:
                batch_mae = torch.mean(torch.abs(y_pred - y))
                test_pbar.set_postfix({
                    'MSE': f'{loss.item():.6f}',
                    'MAE': f'{batch_mae.item():.6f}'
                })
    
    if HAS_TQDM:
        test_pbar.close()
    
    # Calculate metrics from normalized predictions
    all_predictions_norm = np.array(all_predictions_norm)
    all_targets_norm = np.array(all_targets_norm)
    
    mse_norm = total_loss / num_batches
    mae_norm = np.mean(np.abs(all_predictions_norm - all_targets_norm))
    rmse_norm = np.sqrt(mse_norm)
    
    metrics = {
        'mse_norm': float(mse_norm),
        'mae_norm': float(mae_norm),
        'rmse_norm': float(rmse_norm),
        'predictions_norm': all_predictions_norm.tolist(),
        'targets_norm': all_targets_norm.tolist()
    }
    
    # Calculate denormalized metrics if available
    if denormalize and all_predictions_denorm:
        all_predictions_denorm = np.array(all_predictions_denorm)
        all_targets_denorm = np.array(all_targets_denorm)
        
        mse = np.mean((all_predictions_denorm - all_targets_denorm) ** 2)
        mae = np.mean(np.abs(all_predictions_denorm - all_targets_denorm))
        rmse = np.sqrt(mse)
        percentage_error = np.mean(np.abs((all_predictions_denorm - all_targets_denorm) / (all_targets_denorm + 1e-8))) * 100
        
        metrics.update({
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'percentage_error': float(percentage_error),
            'predictions': all_predictions_denorm.tolist(),
            'targets': all_targets_denorm.tolist()
        })
    
    return metrics


def main():
    """Main training function."""
    # Configuration
    DATASET_PATH = "eth_dataset.data"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 20
    NORMALIZE = True
    NORM_METHOD = 'min_max'  # 'min_max' or 'z_score'
    # Test split is the remainder (1 - TRAIN_SPLIT - VAL_SPLIT)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load dataset with normalization
    logger.info("=" * 70)
    logger.info("Loading ETH Dataset")
    logger.info("=" * 70)
    dataset, global_stats = load_eth_dataset(
        DATASET_PATH, 
        normalize=NORMALIZE, 
        norm_method=NORM_METHOD
    )
    
    # Log global statistics
    if NORMALIZE:
        logger.info(f"Global dataset statistics:")
        logger.info(f"  Input prices: min={global_stats['x_min']:.2f}, max={global_stats['x_max']:.2f}, "
                   f"mean={global_stats['x_mean']:.2f}, std={global_stats['x_std']:.2f}")
        logger.info(f"  Target prices: min={global_stats['y_min']:.2f}, max={global_stats['y_max']:.2f}, "
                   f"mean={global_stats['y_mean']:.2f}, std={global_stats['y_std']:.2f}")
        logger.info(f"  Normalization method: {NORM_METHOD}")
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = int(total_size * VAL_SPLIT)
    test_size = total_size - train_size - val_size
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_data):,} samples ({len(train_data)/total_size*100:.1f}%)")
    logger.info(f"  Val:   {len(val_data):,} samples ({len(val_data)/total_size*100:.1f}%)")
    logger.info(f"  Test:  {len(test_data):,} samples ({len(test_data)/total_size*100:.1f}%)")
    
    # Create model
    logger.info("=" * 70)
    logger.info("Creating Model")
    logger.info("=" * 70)
    model = ETHLSTMModel(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train model
    logger.info("=" * 70)
    logger.info("Training Model")
    logger.info("=" * 70)
    train_losses, val_losses, best_epoch = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Evaluate on test set
    logger.info("=" * 70)
    logger.info("Evaluating on Test Set")
    logger.info("=" * 70)
    test_metrics = evaluate_model(
        model, 
        test_data, 
        batch_size=BATCH_SIZE, 
        device=device,
        denormalize=NORMALIZE
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Test Set Evaluation Results")
    logger.info(f"{'='*70}")
    
    if NORMALIZE and 'mse' in test_metrics:
        logger.info(f"Denormalized Metrics (Original Price Scale):")
        logger.info(f"  MSE (Mean Squared Error):     {test_metrics['mse']:.6f}")
        logger.info(f"  MAE (Mean Absolute Error):    {test_metrics['mae']:.6f}")
        logger.info(f"  RMSE (Root Mean Squared Error): {test_metrics['rmse']:.6f}")
        logger.info(f"  Percentage Error:             {test_metrics['percentage_error']:.2f}%")
        logger.info(f"\nNormalized Metrics:")
        logger.info(f"  MSE (Normalized):             {test_metrics['mse_norm']:.6f}")
        logger.info(f"  MAE (Normalized):             {test_metrics['mae_norm']:.6f}")
        logger.info(f"  RMSE (Normalized):            {test_metrics['rmse_norm']:.6f}")
    else:
        logger.info(f"  MSE (Mean Squared Error):     {test_metrics['mse_norm']:.6f}")
        logger.info(f"  MAE (Mean Absolute Error):    {test_metrics['mae_norm']:.6f}")
        logger.info(f"  RMSE (Root Mean Squared Error): {test_metrics['rmse_norm']:.6f}")
    
    logger.info(f"{'='*70}\n")
    
    # Save model
    model_path = Path(__file__).parent.absolute() / "eth_lstm_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info(f"Best model was at epoch {best_epoch}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

