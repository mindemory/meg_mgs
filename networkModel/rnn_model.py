"""
RNN Network Model with Multiple Local Networks

This module implements an RNN with 12 local networks, each containing 500 units
(80% excitatory, 20% inhibitory). Networks are connected via inhibition with
Gaussian tuning. Each network has a Gaussian receptive field for polar angles.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union


class LocalNetwork(nn.Module):
    """
    A single local network with excitatory and inhibitory units.
    
    Parameters:
    -----------
    n_units : int
        Total number of units in the network (default: 500)
    excitatory_ratio : float
        Ratio of excitatory units (default: 0.8)
    tau : float
        Time constant for dynamics (default: 0.1)
    dt : float
        Integration time step (default: 0.01)
    """
    
    def __init__(
        self,
        n_units: int = 500,
        excitatory_ratio: float = 0.8,
        tau: float = 0.1,
        dt: float = 0.01
    ):
        super(LocalNetwork, self).__init__()
        
        self.n_units = n_units
        self.n_excitatory = int(n_units * excitatory_ratio)
        self.n_inhibitory = n_units - self.n_excitatory
        self.tau = tau
        self.dt = dt
        
        # Initialize local connectivity weights
        # Excitatory to excitatory connections
        self.register_buffer('W_ee', self._init_local_weights(
            self.n_excitatory, self.n_excitatory, is_excitatory=True
        ))
        # Excitatory to inhibitory connections
        self.register_buffer('W_ei', self._init_local_weights(
            self.n_excitatory, self.n_inhibitory, is_excitatory=True
        ))
        # Inhibitory to excitatory connections
        self.register_buffer('W_ie', self._init_local_weights(
            self.n_inhibitory, self.n_excitatory, is_excitatory=False
        ))
        # Inhibitory to inhibitory connections
        self.register_buffer('W_ii', self._init_local_weights(
            self.n_inhibitory, self.n_inhibitory, is_excitatory=False
        ))
        
        # Initialize state
        self.register_buffer('r', torch.zeros(self.n_units))
        
    def _init_local_weights(
        self,
        n_pre: int,
        n_post: int,
        is_excitatory: bool,
        connection_prob: float = 0.1,
        weight_scale: float = 1.0
    ) -> torch.Tensor:
        """Initialize local connectivity weights."""
        # Create sparse connectivity mask
        mask = torch.rand(n_pre, n_post) < connection_prob
        
        # Initialize weights
        if is_excitatory:
            # Excitatory weights are positive
            weights = torch.randn(n_pre, n_post) * weight_scale * mask.float()
            weights = torch.clamp(weights, min=0.0)  # Ensure non-negative
        else:
            # Inhibitory weights are negative
            weights = -torch.abs(torch.randn(n_pre, n_post) * weight_scale) * mask.float()
        
        return weights
    
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the local network.
        
        Parameters:
        -----------
        input_current : torch.Tensor
            External input current (shape: [n_units])
            
        Returns:
        --------
        torch.Tensor
            Firing rates (shape: [n_units])
        """
        # Split into excitatory and inhibitory populations
        r_e = self.r[:self.n_excitatory]
        r_i = self.r[self.n_excitatory:]
        
        # Compute local recurrent inputs
        # Excitatory population receives: E->E, I->E
        input_e = (self.W_ee.T @ r_e + self.W_ie.T @ r_i + 
                   input_current[:self.n_excitatory])
        
        # Inhibitory population receives: E->I, I->I
        input_i = (self.W_ei.T @ r_e + self.W_ii.T @ r_i + 
                   input_current[self.n_excitatory:])
        
        # Combine inputs
        total_input = torch.cat([input_e, input_i])
        
        # Update firing rates using Euler integration
        # dr/dt = (-r + phi(I)) / tau
        # where phi is a nonlinearity (e.g., ReLU or tanh)
        phi_input = torch.relu(total_input)  # Simple ReLU nonlinearity
        dr = (-self.r + phi_input) / self.tau
        self.r = self.r + self.dt * dr
        
        return self.r


class MultiNetworkRNN(nn.Module):
    """
    RNN with multiple local networks connected via inhibition with Gaussian tuning.
    
    Parameters:
    -----------
    n_networks : int
        Number of local networks (default: 12)
    n_units_per_network : int
        Number of units per network (default: 500)
    excitatory_ratio : float
        Ratio of excitatory units (default: 0.8)
    inhibition_strength : float
        Strength of inter-network inhibition (default: 1.0)
    gaussian_sigma : float
        Standard deviation for Gaussian tuning of inter-network connections
        (default: 2.0)
    tau : float
        Time constant for dynamics (default: 0.1)
    dt : float
        Integration time step (default: 0.01)
    """
    
    def __init__(
        self,
        n_networks: int = 12,
        n_units_per_network: int = 500,
        excitatory_ratio: float = 0.8,
        inhibition_strength: float = 1.0,
        gaussian_sigma: float = 2.0,
        receptive_field_sigma: float = 30.0,  # Width of receptive field in degrees
        input_strength: float = 1.0,  # Scaling factor for input
        tau: float = 0.1,
        dt: float = 0.01
    ):
        super(MultiNetworkRNN, self).__init__()
        
        self.n_networks = n_networks
        self.n_units_per_network = n_units_per_network
        self.excitatory_ratio = excitatory_ratio
        self.n_excitatory = int(n_units_per_network * excitatory_ratio)
        self.n_inhibitory = n_units_per_network - self.n_excitatory
        self.inhibition_strength = inhibition_strength
        self.gaussian_sigma = gaussian_sigma
        self.receptive_field_sigma = receptive_field_sigma
        self.input_strength = input_strength
        self.tau = tau
        self.dt = dt
        
        # Define preferred angles for each network (equally spaced from 0 to 360)
        # Network 1 at 0°, Network 2 at 30°, etc.
        angle_step = 360.0 / n_networks
        self.preferred_angles = torch.tensor(
            [i * angle_step for i in range(n_networks)],
            dtype=torch.float32
        )
        self.register_buffer('preferred_angles_buffer', self.preferred_angles)
        
        # Create local networks
        self.local_networks = nn.ModuleList([
            LocalNetwork(
                n_units=n_units_per_network,
                excitatory_ratio=excitatory_ratio,
                tau=tau,
                dt=dt
            )
            for _ in range(n_networks)
        ])
        
        # Initialize inter-network inhibition weights with Gaussian tuning
        self.register_buffer('W_inter', self._init_inter_network_weights())
        
    def _init_inter_network_weights(self) -> torch.Tensor:
        """
        Initialize inter-network inhibition weights with Gaussian tuning.
        
        The Gaussian tuning means that networks closer together (in some
        feature space, e.g., network index) have stronger connections.
        """
        # Create network positions (e.g., on a 1D line or 2D grid)
        # For simplicity, we'll use network indices as positions
        network_positions = torch.arange(self.n_networks, dtype=torch.float32)
        
        # Compute pairwise distances
        pos_i = network_positions.unsqueeze(1)  # [n_networks, 1]
        pos_j = network_positions.unsqueeze(0)  # [1, n_networks]
        distances = pos_i - pos_j  # [n_networks, n_networks]
        
        # Gaussian tuning: W_ij = -strength * exp(-(i-j)^2 / (2*sigma^2))
        # Only inhibitory connections (negative weights)
        gaussian_weights = -self.inhibition_strength * torch.exp(
            -0.5 * (distances / self.gaussian_sigma) ** 2
        )
        
        # No self-connections (diagonal should be zero)
        gaussian_weights.fill_diagonal_(0.0)
        
        # Expand to full connectivity matrix
        # Shape: [n_networks, n_networks, n_inhibitory, n_inhibitory]
        # Each network's inhibitory units connect to other networks' inhibitory units
        W_inter = torch.zeros(
            self.n_networks,
            self.n_networks,
            self.n_inhibitory,
            self.n_inhibitory
        )
        
        for i in range(self.n_networks):
            for j in range(self.n_networks):
                if i != j:
                    # Create sparse connectivity between inhibitory units
                    connection_prob = 0.1
                    mask = torch.rand(self.n_inhibitory, self.n_inhibitory) < connection_prob
                    W_inter[i, j] = gaussian_weights[i, j] * mask.float()
        
        return W_inter
    
    def encode_angle(self, angle: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Encode a polar angle into a 12-dimensional vector using Gaussian receptive fields.
        
        Parameters:
        -----------
        angle : float or torch.Tensor
            Angle in degrees (0-360)
            
        Returns:
        --------
        torch.Tensor
            Encoded input vector of shape [n_networks]
        """
        if isinstance(angle, (int, float)):
            angle = torch.tensor(float(angle), dtype=torch.float32)
        
        # Ensure angle is in [0, 360)
        angle = angle % 360.0
        
        # Compute circular distance (handling wrap-around at 0/360)
        angles_diff = angle - self.preferred_angles_buffer
        # Handle circular distance: min(|diff|, 360 - |diff|)
        angles_diff = torch.minimum(
            torch.abs(angles_diff),
            360.0 - torch.abs(angles_diff)
        )
        
        # Gaussian receptive field encoding
        encoding = torch.exp(-0.5 * (angles_diff / self.receptive_field_sigma) ** 2)
        
        return encoding * self.input_strength
    
    def aggregate_output(self, unit_outputs: torch.Tensor) -> torch.Tensor:
        """
        Aggregate unit-level outputs to network-level outputs.
        
        Parameters:
        -----------
        unit_outputs : torch.Tensor
            Unit firing rates of shape [n_networks, n_units_per_network]
            
        Returns:
        --------
        torch.Tensor
            Network-level outputs of shape [n_networks]
        """
        # Use mean firing rate per network as the output
        return unit_outputs.mean(dim=1)
    
    def forward(
        self,
        input_current: Optional[Union[torch.Tensor, float]] = None,
        return_unit_outputs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all networks.
        
        Parameters:
        -----------
        input_current : torch.Tensor, float, or None
            Input can be:
            - A scalar angle (0-360 degrees) - will be encoded via receptive fields
            - A [n_networks] vector - will be broadcast to all units
            - A [n_networks, n_units_per_network] tensor - per-unit input
            - None - uses zero input
        return_unit_outputs : bool
            If True, returns [n_networks, n_units_per_network]
            If False, returns aggregated [n_networks] output
            
        Returns:
        --------
        torch.Tensor
            If return_unit_outputs=True: [n_networks, n_units_per_network]
            If return_unit_outputs=False: [n_networks] (aggregated)
        """
        # Handle different input types
        if input_current is None:
            # Get device from one of the buffers
            device = next(self.local_networks[0].buffers()).device
            network_input = torch.zeros(
                self.n_networks,
                device=device
            )
        elif isinstance(input_current, (int, float)):
            # Encode angle into network-level input
            network_input = self.encode_angle(input_current)
        elif isinstance(input_current, torch.Tensor):
            if input_current.dim() == 0:  # Scalar tensor
                network_input = self.encode_angle(input_current.item())
            elif input_current.shape == (self.n_networks,):
                # Network-level input [12]
                network_input = input_current
            elif input_current.shape == (self.n_networks, self.n_units_per_network):
                # Per-unit input - use directly
                unit_input = input_current
            else:
                raise ValueError(f"Unexpected input shape: {input_current.shape}")
        else:
            raise TypeError(f"Unexpected input type: {type(input_current)}")
        
        # If we have network-level input, broadcast to all units
        if 'unit_input' not in locals():
            device = network_input.device
            # Broadcast network input to all units (each unit in a network gets the same input)
            unit_input = network_input.unsqueeze(1).expand(
                self.n_networks, self.n_units_per_network
            )
        
        # Collect firing rates from all networks
        all_rates = []
        
        for i, local_net in enumerate(self.local_networks):
            # Get local input
            local_input = unit_input[i].clone()
            
            # Add inter-network inhibition
            # Each network receives inhibition from other networks' inhibitory units
            inter_inhibition = torch.zeros(self.n_units_per_network)
            
            for j in range(self.n_networks):
                if i != j:
                    # Get inhibitory firing rates from network j
                    r_j_inhibitory = local_net.r[self.n_excitatory:]
                    
                    # Compute inhibition: sum over inhibitory units of network j
                    inhibition_from_j = self.W_inter[i, j] @ r_j_inhibitory
                    
                    # Add to inhibitory units of network i
                    inter_inhibition[self.n_excitatory:] += inhibition_from_j
            
            # Add inter-network inhibition to local input
            local_input += inter_inhibition
            
            # Forward through local network
            rates = local_net(local_input)
            all_rates.append(rates)
        
        unit_outputs = torch.stack(all_rates)
        
        if return_unit_outputs:
            return unit_outputs
        else:
            # Aggregate to network-level output
            return self.aggregate_output(unit_outputs)
    
    def simulate_trial(
        self,
        angle: Union[float, torch.Tensor],
        stimulus_duration: float = 200.0,  # ms
        total_duration: float = 200.0,  # ms
        dt: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate a trial with stimulus presentation and readout.
        
        Parameters:
        -----------
        angle : float or torch.Tensor
            Stimulus angle in degrees (0-360)
        stimulus_duration : float
            Duration of stimulus presentation in ms (default: 200ms)
        total_duration : float
            Total simulation duration in ms (default: 200ms)
        dt : float, optional
            Time step in ms (default: uses self.dt * 1000)
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (time_points, outputs) where:
            - time_points: [n_steps] array of time points in ms
            - outputs: [n_steps, n_networks] array of network outputs
        """
        if dt is None:
            dt_ms = self.dt * 1000  # Convert to ms
        else:
            dt_ms = dt
        
        n_steps = int(total_duration / dt_ms)
        stimulus_steps = int(stimulus_duration / dt_ms)
        
        # Encode angle to network input
        network_input = self.encode_angle(angle)
        
        # Reset network state
        self.reset_state()
        
        # Simulate
        outputs = []
        time_points = []
        
        for step in range(n_steps):
            # Apply stimulus only during stimulus_duration
            if step < stimulus_steps:
                input_vec = network_input
            else:
                input_vec = torch.zeros_like(network_input)
            
            # Forward pass
            output = self.forward(input_vec, return_unit_outputs=False)
            outputs.append(output.clone())
            time_points.append(step * dt_ms)
        
        return torch.tensor(time_points), torch.stack(outputs)
    
    def reset_state(self):
        """Reset the state of all networks."""
        for local_net in self.local_networks:
            local_net.r.zero_()
    
    def get_network_statistics(self) -> dict:
        """Get statistics about the network architecture."""
        return {
            'n_networks': self.n_networks,
            'n_units_per_network': self.n_units_per_network,
            'n_excitatory': self.n_excitatory,
            'n_inhibitory': self.n_inhibitory,
            'excitatory_ratio': self.excitatory_ratio,
            'total_units': self.n_networks * self.n_units_per_network,
            'inter_network_connections': (self.n_networks * (self.n_networks - 1) * 
                                         self.n_inhibitory * self.n_inhibitory)
        }


def create_rnn_model(
    n_networks: int = 12,
    n_units_per_network: int = 500,
    excitatory_ratio: float = 0.8,
    inhibition_strength: float = 1.0,
    gaussian_sigma: float = 2.0,
    receptive_field_sigma: float = 30.0,
    input_strength: float = 1.0,
    device: str = 'cpu'
) -> MultiNetworkRNN:
    """
    Factory function to create a multi-network RNN model.
    
    Parameters:
    -----------
    n_networks : int
        Number of local networks
    n_units_per_network : int
        Number of units per network
    excitatory_ratio : float
        Ratio of excitatory units
    inhibition_strength : float
        Strength of inter-network inhibition
    gaussian_sigma : float
        Standard deviation for Gaussian tuning
    device : str
        Device to run the model on ('cpu' or 'cuda')
        
    Returns:
    --------
    MultiNetworkRNN
        The created RNN model
    """
    model = MultiNetworkRNN(
        n_networks=n_networks,
        n_units_per_network=n_units_per_network,
        excitatory_ratio=excitatory_ratio,
        inhibition_strength=inhibition_strength,
        gaussian_sigma=gaussian_sigma,
        receptive_field_sigma=receptive_field_sigma,
        input_strength=input_strength
    )
    return model.to(device)


if __name__ == "__main__":
    # Example usage
    print("Creating RNN model with 12 networks, 500 units each...")
    model = create_rnn_model(
        n_networks=12,
        n_units_per_network=500,
        excitatory_ratio=0.8,
        inhibition_strength=1.0,
        gaussian_sigma=2.0
    )
    
    # Print network statistics
    stats = model.get_network_statistics()
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nPreferred angles (degrees): {model.preferred_angles.numpy()}")
    
    # Test angle encoding
    print("\nTesting angle encoding...")
    test_angle = 45.0
    encoded = model.encode_angle(test_angle)
    print(f"Angle {test_angle}° encoded to: {encoded.numpy()}")
    
    # Test forward pass with angle input
    print("\nTesting forward pass with angle input...")
    model.reset_state()
    output = model(test_angle)
    print(f"Output shape: {output.shape}")
    print(f"Output (network responses): {output.numpy()}")
    
    # Test temporal simulation
    print("\nTesting temporal simulation (200ms stimulus, 200ms total)...")
    time_points, outputs = model.simulate_trial(angle=45.0, stimulus_duration=200.0, total_duration=200.0)
    print(f"Time points shape: {time_points.shape}")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Final output (at {time_points[-1]:.1f}ms): {outputs[-1].numpy()}")

