# Multi-Network RNN Model

This repository contains an implementation of a recurrent neural network (RNN) with multiple local networks connected via inhibition with Gaussian tuning.

## Architecture

- **12 local networks**, each containing **500 units**
- **80% excitatory units** (400 units) and **20% inhibitory units** (100 units) per network
- **Inter-network connectivity**: Networks are connected via inhibition with Gaussian tuning, where closer networks have stronger inhibitory connections

## Features

- Modular design with separate `LocalNetwork` and `MultiNetworkRNN` classes
- Configurable parameters for network size, connectivity, and dynamics
- Gaussian-tuned inter-network inhibition based on network proximity
- PyTorch implementation for efficient computation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from rnn_model import create_rnn_model

# Create the model
model = create_rnn_model(
    n_networks=12,
    n_units_per_network=500,
    excitatory_ratio=0.8,
    inhibition_strength=1.0,
    gaussian_sigma=2.0
)

# Reset network state
model.reset_state()

# Run forward pass
input_current = torch.randn(10, 500) * 0.1
output = model(input_current)  # Shape: [10, 500]

# Get network statistics
stats = model.get_network_statistics()
print(stats)
```

### Custom Parameters

You can customize various aspects of the network:

- `n_networks`: Number of local networks (default: 10)
- `n_units_per_network`: Units per network (default: 500)
- `excitatory_ratio`: Ratio of excitatory units (default: 0.8)
- `inhibition_strength`: Strength of inter-network inhibition (default: 1.0)
- `gaussian_sigma`: Standard deviation for Gaussian tuning (default: 2.0)
- `tau`: Time constant for dynamics (default: 0.1)
- `dt`: Integration time step (default: 0.01)

## Network Structure

### Local Networks

Each local network has:
- **Excitatory units**: 400 units (80%)
- **Inhibitory units**: 100 units (20%)
- **Local connectivity**: Sparse connections within each network following E-E, E-I, I-E, I-I patterns

### Inter-Network Connectivity

- Networks are connected via inhibitory connections
- Connection strength follows a Gaussian function: `W_ij = -strength * exp(-(i-j)^2 / (2*sigma^2))`
- Only inhibitory units participate in inter-network connections
- Closer networks (in index space) have stronger connections

## Example

See `rnn_model.py` for a complete example in the `__main__` block.

