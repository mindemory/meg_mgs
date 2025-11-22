"""
Training script for the Multi-Network RNN Model

Trains the network to report target angles and tracks performance over time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from rnn_model import create_rnn_model
from typing import Tuple, List
import time


class AngleReadout(nn.Module):
    """
    Readout layer to convert network outputs [12] to a reported angle.
    Uses population vector decoding.
    """
    def __init__(self, preferred_angles: torch.Tensor):
        super(AngleReadout, self).__init__()
        self.register_buffer('preferred_angles', preferred_angles)
        # Learnable weights for readout
        self.readout_weights = nn.Parameter(torch.ones(len(preferred_angles)))
        
    def forward(self, network_outputs: torch.Tensor) -> torch.Tensor:
        """
        Convert network outputs to reported angle using population vector.
        
        Parameters:
        -----------
        network_outputs : torch.Tensor
            Shape: [batch_size, n_networks] or [n_networks]
            
        Returns:
        --------
        torch.Tensor
            Reported angles in degrees [batch_size] or scalar
        """
        # Ensure 2D
        if network_outputs.dim() == 1:
            network_outputs = network_outputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Weight the network outputs
        weighted_outputs = network_outputs * self.readout_weights.unsqueeze(0)
        
        # Convert preferred angles to radians for vector computation
        angles_rad = torch.deg2rad(self.preferred_angles)
        
        # Compute population vector (complex representation)
        # Each network contributes a vector with magnitude = output, angle = preferred_angle
        x_components = weighted_outputs * torch.cos(angles_rad).unsqueeze(0)
        y_components = weighted_outputs * torch.sin(angles_rad).unsqueeze(0)
        
        # Sum across networks
        x_sum = x_components.sum(dim=1)
        y_sum = y_components.sum(dim=1)
        
        # Compute angle from population vector
        reported_angles = torch.rad2deg(torch.atan2(y_sum, x_sum))
        # Convert from [-180, 180] to [0, 360]
        reported_angles = (reported_angles + 360) % 360
        
        if squeeze_output:
            return reported_angles.squeeze(0)
        return reported_angles


def circular_distance(angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
    """
    Compute circular distance between two angles (handles wrap-around).
    
    Parameters:
    -----------
    angle1, angle2 : torch.Tensor
        Angles in degrees
        
    Returns:
    --------
    torch.Tensor
        Circular distance in degrees
    """
    diff = angle1 - angle2
    # Handle circular distance
    diff = torch.minimum(torch.abs(diff), 360.0 - torch.abs(diff))
    return diff


def train_epoch(
    model: nn.Module,
    readout: nn.Module,
    optimizer: optim.Optimizer,
    n_samples: int = 100,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
    --------
    Tuple[float, float]
        (average_loss, average_error)
    """
    model.train()
    readout.train()
    
    total_loss = 0.0
    total_error = 0.0
    
    for _ in range(n_samples):
        # Sample random angle
        target_angle = torch.rand(1).item() * 360.0
        
        # Reset network state
        model.reset_state()
        
        # Simulate trial
        time_points, outputs = model.simulate_trial(
            angle=target_angle,
            stimulus_duration=200.0,
            total_duration=200.0
        )
        
        # Get final output (at 200ms)
        final_output = outputs[-1]  # [n_networks]
        
        # Readout angle
        reported_angle = readout(final_output)
        
        # Compute circular distance error
        error = circular_distance(
            torch.tensor(target_angle),
            reported_angle
        )
        
        # Loss is the error
        loss = error
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_error += error.item()
    
    return total_loss / n_samples, total_error / n_samples


def evaluate(
    model: nn.Module,
    readout: nn.Module,
    n_samples: int = 50,
    device: str = 'cpu'
) -> Tuple[float, List[float], List[float]]:
    """
    Evaluate model performance.
    
    Returns:
    --------
    Tuple[float, List[float], List[float]]
        (average_error, target_angles, reported_angles)
    """
    model.eval()
    readout.eval()
    
    total_error = 0.0
    target_angles = []
    reported_angles = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Sample random angle
            target_angle = torch.rand(1).item() * 360.0
            target_angles.append(target_angle)
            
            # Reset network state
            model.reset_state()
            
            # Simulate trial
            time_points, outputs = model.simulate_trial(
                angle=target_angle,
                stimulus_duration=200.0,
                total_duration=200.0
            )
            
            # Get final output
            final_output = outputs[-1]
            
            # Readout angle
            reported_angle = readout(final_output).item()
            reported_angles.append(reported_angle)
            
            # Compute error
            error = circular_distance(
                torch.tensor(target_angle),
                torch.tensor(reported_angle)
            )
            total_error += error.item()
    
    return total_error / n_samples, target_angles, reported_angles


def evaluate_delay(
    model: nn.Module,
    readout: nn.Module,
    delays: List[float],
    stimulus_duration: float = 200.0,
    n_samples: int = 50,
    device: str = 'cpu'
) -> Tuple[List[float], List[float]]:
    """
    Evaluate model performance at different delays after stimulus offset.
    
    Parameters:
    -----------
    model : nn.Module
        The trained RNN model
    readout : nn.Module
        The readout layer
    delays : List[float]
        List of delay durations in ms (e.g., [1000, 2000, 3000, 4000])
    stimulus_duration : float
        Duration of stimulus presentation in ms (default: 200ms)
    n_samples : int
        Number of test samples per delay
    device : str
        Device to run on
        
    Returns:
    --------
    Tuple[List[float], List[float]]
        (delays, average_errors) - delay values and corresponding average errors
    """
    model.eval()
    readout.eval()
    
    delay_errors = []
    
    print(f"\nEvaluating performance at different delays...")
    print(f"Stimulus duration: {stimulus_duration}ms")
    print(f"Test samples per delay: {n_samples}")
    print("-" * 60)
    
    with torch.no_grad():
        for delay in delays:
            total_error = 0.0
            valid_count = 0
            total_duration = stimulus_duration + delay
            
            for _ in range(n_samples):
                # Sample random angle
                target_angle = torch.rand(1).item() * 360.0
                
                # Reset network state
                model.reset_state()
                
                # Simulate trial with delay
                time_points, outputs = model.simulate_trial(
                    angle=target_angle,
                    stimulus_duration=stimulus_duration,
                    total_duration=total_duration
                )
                
                # Get final output (at end of delay period)
                final_output = outputs[-1]
                
                # Check for NaN or Inf values
                if torch.isnan(final_output).any() or torch.isinf(final_output).any():
                    # Skip this sample if network became unstable
                    continue
                
                # Readout angle
                reported_angle = readout(final_output).item()
                
                # Check if reported angle is valid
                if np.isnan(reported_angle) or np.isinf(reported_angle):
                    continue
                
                # Compute error
                error = circular_distance(
                    torch.tensor(target_angle),
                    torch.tensor(reported_angle)
                )
                error_val = error.item()
                if not (np.isnan(error_val) or np.isinf(error_val)):
                    total_error += error_val
                    valid_count += 1
            
            # Count valid samples
            if valid_count > 0:
                avg_error = total_error / valid_count
            else:
                avg_error = float('inf')
            delay_errors.append(avg_error)
            if not np.isinf(avg_error):
                print(f"Delay: {delay:6.0f}ms | Total duration: {total_duration:6.0f}ms | Error: {avg_error:7.2f}° | Valid samples: {valid_count}/{n_samples}")
            else:
                print(f"Delay: {delay:6.0f}ms | Total duration: {total_duration:6.0f}ms | Error: Unstable (all samples failed)")
    
    print("-" * 60)
    return delays, delay_errors


def train_model(
    n_epochs: int = 100,
    n_train_samples: int = 100,
    n_test_samples: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Train the model and track performance.
    """
    print("Creating model...")
    model = create_rnn_model(
        n_networks=12,
        n_units_per_network=500,
        excitatory_ratio=0.8,
        inhibition_strength=1.0,
        gaussian_sigma=2.0,
        receptive_field_sigma=30.0,
        input_strength=1.0,
        device=device
    )
    
    # Create readout layer
    readout = AngleReadout(model.preferred_angles).to(device)
    
    # Optimizer - only train readout weights for now
    # (Could also train network weights if needed)
    optimizer = optim.Adam(readout.parameters(), lr=learning_rate)
    
    # Track performance
    train_errors = []
    test_errors = []
    epochs_list = []
    
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Training samples per epoch: {n_train_samples}")
    print(f"Test samples per evaluation: {n_test_samples}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Train
        train_loss, train_error = train_epoch(
            model, readout, optimizer, n_train_samples, device
        )
        
        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Evaluate more frequently
            test_error, _, _ = evaluate(
                model, readout, n_test_samples, device
            )
            
            train_errors.append(train_error)
            test_errors.append(test_error)
            epochs_list.append(epoch + 1)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d} | Train Error: {train_error:7.2f}° | "
                  f"Test Error: {test_error:7.2f}° | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.1f} seconds")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_error, target_angles, reported_angles = evaluate(
        model, readout, n_samples=50, device=device  # Reduced from 200
    )
    print(f"Final test error: {final_error:.2f}°")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_errors, 'b-', label='Train Error', linewidth=2)
    plt.plot(epochs_list, test_errors, 'r-', label='Test Error', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error (degrees)', fontsize=12)
    plt.title('Training Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(target_angles, reported_angles, alpha=0.5, s=20)
    plt.plot([0, 360], [0, 360], 'r--', linewidth=2, label='Perfect')
    plt.xlabel('Target Angle (degrees)', fontsize=12)
    plt.ylabel('Reported Angle (degrees)', fontsize=12)
    plt.title(f'Final Performance (Error: {final_error:.2f}°)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.ylim(0, 360)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()
    
    return model, readout, train_errors, test_errors, epochs_list


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Starting training script...")
    print("=" * 60)
    
    # Train the model for 30 epochs
    model, readout, train_errors, test_errors, epochs = train_model(
        n_epochs=30,
        n_train_samples=50,
        n_test_samples=10,
        learning_rate=1e-3,
        device='cpu',
        save_path=None  # Set to 'training_results.png' to save
    )
    
    # Freeze the model (stop training)
    print("\n" + "=" * 60)
    print("Freezing model (no further training)...")
    model.eval()
    readout.eval()
    for param in model.parameters():
        param.requires_grad = False
    for param in readout.parameters():
        param.requires_grad = False
    print("Model frozen.")
    
    # Evaluate performance at different delays
    delays = [1000.0, 2000.0, 3000.0, 4000.0]  # 1s, 2s, 3s, 4s in ms
    delay_values, delay_errors = evaluate_delay(
        model=model,
        readout=readout,
        delays=delays,
        stimulus_duration=200.0,
        n_samples=50,
        device='cpu'
    )
    
    # Plot delay performance
    print("\nPlotting delay performance...")
    plt.figure(figsize=(10, 6))
    plt.plot(delay_values, delay_errors, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Delay (ms)', fontsize=12)
    plt.ylabel('Error (degrees)', fontsize=12)
    plt.title('Model Performance vs Delay\n(Stimulus: 200ms, Model frozen after 30 epochs)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(delay_values)
    
    # Add value labels on points
    for delay, error in zip(delay_values, delay_errors):
        plt.text(delay, error + max(delay_errors) * 0.05, f'{error:.2f}°', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")

