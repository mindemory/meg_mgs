"""
Visualization script for the Multi-Network RNN Model

This script creates visualizations of the network architecture including:
- Network structure overview
- Connectivity matrices
- Gaussian tuning curves
- Network graph representation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
from rnn_model import create_rnn_model


def visualize_network_architecture(model):
    """
    Create comprehensive visualizations of the network architecture.
    
    Parameters:
    -----------
    model : MultiNetworkRNN
        The RNN model to visualize
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Network Structure Overview (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    visualize_network_structure(ax1, model)
    
    # 2. Excitatory/Inhibitory Distribution (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    visualize_unit_distribution(ax2, model)
    
    # 3. Inter-network Gaussian Tuning (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    visualize_gaussian_tuning(ax3, model)
    
    # 4. Inter-network Connectivity Matrix (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    visualize_inter_network_connectivity(ax4, model)
    
    # 5. Local Network Connectivity (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    visualize_local_connectivity(ax5, model)
    
    # 6. Network Graph (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    visualize_network_graph(ax6, model)
    
    # 7. Connection Strength Distribution (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    visualize_connection_strength_distribution(ax7, model)
    
    # 8. Network Statistics Table (bottom middle-right)
    ax8 = fig.add_subplot(gs[2, 1:])
    visualize_statistics_table(ax8, model)
    
    plt.suptitle('Multi-Network RNN Architecture Visualization', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.show()


def visualize_network_structure(ax, model):
    """Visualize the overall network structure."""
    n_networks = model.n_networks
    n_units = model.n_units_per_network
    
    # Draw networks as rectangles
    network_width = 0.8
    network_height = n_units / 100  # Scale for visibility
    
    for i in range(n_networks):
        x = i
        y = 0
        
        # Draw network rectangle
        rect = mpatches.Rectangle(
            (x - network_width/2, y),
            network_width,
            network_height,
            linewidth=2,
            edgecolor='black',
            facecolor='lightblue',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add network label
        ax.text(x, y + network_height/2, f'N{i+1}', 
                ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Add unit count
        ax.text(x, y - network_height*0.2, f'{n_units} units',
                ha='center', va='top', fontsize=8)
    
    ax.set_xlim(-0.5, n_networks - 0.5)
    ax.set_ylim(-network_height*0.5, network_height*1.2)
    ax.set_xlabel('Network Index', fontsize=10)
    ax.set_ylabel('Units (scaled)', fontsize=10)
    ax.set_title('Network Structure\n(10 Networks, 500 Units Each)', fontsize=11, fontweight='bold')
    ax.set_xticks(range(n_networks))
    ax.set_xticklabels([f'N{i+1}' for i in range(n_networks)])
    ax.grid(True, alpha=0.3)


def visualize_unit_distribution(ax, model):
    """Visualize excitatory/inhibitory unit distribution."""
    n_excitatory = model.n_excitatory
    n_inhibitory = model.n_inhibitory
    
    categories = ['Excitatory\n(80%)', 'Inhibitory\n(20%)']
    counts = [n_excitatory, n_inhibitory]
    colors = ['#2ecc71', '#e74c3c']  # Green for excitatory, Red for inhibitory
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Number of Units', fontsize=10)
    ax.set_title('Unit Distribution per Network', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')


def visualize_gaussian_tuning(ax, model):
    """Visualize the Gaussian tuning function for inter-network connections."""
    n_networks = model.n_networks
    sigma = model.gaussian_sigma
    strength = model.inhibition_strength
    
    # Create distance array
    distances = np.arange(0, n_networks, 0.1)
    
    # Compute Gaussian tuning
    gaussian_weights = -strength * np.exp(-0.5 * (distances / sigma) ** 2)
    
    ax.plot(distances, gaussian_weights, 'b-', linewidth=2, label='Gaussian Tuning')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Mark network positions
    network_positions = np.arange(n_networks)
    for pos in network_positions:
        weight = -strength * np.exp(-0.5 * (pos / sigma) ** 2) if pos > 0 else 0
        ax.plot(pos, weight, 'ro', markersize=8, zorder=5)
    
    ax.set_xlabel('Network Distance', fontsize=10)
    ax.set_ylabel('Connection Strength', fontsize=10)
    ax.set_title(f'Inter-Network Gaussian Tuning\n(σ={sigma}, strength={strength})', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def visualize_inter_network_connectivity(ax, model):
    """Visualize the inter-network connectivity matrix."""
    n_networks = model.n_networks
    sigma = model.gaussian_sigma
    strength = model.inhibition_strength
    
    # Compute connectivity matrix
    network_positions = np.arange(n_networks)
    pos_i = network_positions[:, np.newaxis]
    pos_j = network_positions[np.newaxis, :]
    distances = pos_i - pos_j
    
    connectivity = -strength * np.exp(-0.5 * (distances / sigma) ** 2)
    np.fill_diagonal(connectivity, 0)  # No self-connections
    
    im = ax.imshow(connectivity, cmap='Reds', aspect='auto', origin='lower')
    ax.set_xlabel('Target Network', fontsize=10)
    ax.set_ylabel('Source Network', fontsize=10)
    ax.set_title('Inter-Network Connectivity Matrix\n(Inhibitory, Gaussian-tuned)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(range(n_networks))
    ax.set_yticks(range(n_networks))
    ax.set_xticklabels([f'N{i+1}' for i in range(n_networks)])
    ax.set_yticklabels([f'N{i+1}' for i in range(n_networks)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Inhibition Strength', fontsize=9)
    
    # Add text annotations for small matrices
    if n_networks <= 12:
        for i in range(n_networks):
            for j in range(n_networks):
                if i != j:
                    text = ax.text(j, i, f'{connectivity[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=7)


def visualize_local_connectivity(ax, model):
    """Visualize local network connectivity patterns."""
    # Get connectivity from first local network
    local_net = model.local_networks[0]
    
    # Create combined connectivity matrix
    n_excitatory = model.n_excitatory
    n_inhibitory = model.n_inhibitory
    n_total = n_excitatory + n_inhibitory
    
    connectivity = np.zeros((n_total, n_total))
    
    # Fill in the connectivity blocks
    connectivity[:n_excitatory, :n_excitatory] = local_net.W_ee.cpu().numpy()
    connectivity[:n_excitatory, n_excitatory:] = local_net.W_ei.cpu().numpy()
    connectivity[n_excitatory:, :n_excitatory] = local_net.W_ie.cpu().numpy()
    connectivity[n_excitatory:, n_excitatory:] = local_net.W_ii.cpu().numpy()
    
    # Sample for visualization if too large
    sample_size = min(200, n_total)
    if n_total > sample_size:
        step = n_total // sample_size
        connectivity = connectivity[::step, ::step]
        n_total = sample_size
    
    im = ax.imshow(connectivity, cmap='RdBu_r', aspect='auto', origin='lower',
                   vmin=-np.abs(connectivity).max(), vmax=np.abs(connectivity).max())
    
    # Add dividing lines
    if n_total == model.n_units_per_network:
        ax.axhline(y=n_excitatory-0.5, color='black', linewidth=2, linestyle='--')
        ax.axvline(x=n_excitatory-0.5, color='black', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Post-synaptic Unit', fontsize=10)
    ax.set_ylabel('Pre-synaptic Unit', fontsize=10)
    ax.set_title('Local Network Connectivity\n(E-E, E-I, I-E, I-I)', 
                 fontsize=11, fontweight='bold')
    
    # Add labels
    if n_total == model.n_units_per_network:
        ax.text(n_excitatory/2, -n_total*0.05, 'E', ha='center', fontweight='bold', fontsize=10)
        ax.text(n_excitatory + n_inhibitory/2, -n_total*0.05, 'I', ha='center', fontweight='bold', fontsize=10)
        ax.text(-n_total*0.05, n_excitatory/2, 'E', ha='center', va='center', fontweight='bold', fontsize=10, rotation=90)
        ax.text(-n_total*0.05, n_excitatory + n_inhibitory/2, 'I', ha='center', va='center', fontweight='bold', fontsize=10, rotation=90)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', fontsize=9)


def visualize_network_graph(ax, model):
    """Visualize network graph with connection strengths."""
    n_networks = model.n_networks
    sigma = model.gaussian_sigma
    strength = model.inhibition_strength
    
    # Compute positions in a circle
    angles = np.linspace(0, 2*np.pi, n_networks, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Compute connectivity
    network_positions = np.arange(n_networks)
    pos_i = network_positions[:, np.newaxis]
    pos_j = network_positions[np.newaxis, :]
    distances = np.abs(pos_i - pos_j)
    connectivity = -strength * np.exp(-0.5 * (distances / sigma) ** 2)
    np.fill_diagonal(connectivity, 0)
    
    # Draw connections
    max_weight = np.abs(connectivity).max()
    for i in range(n_networks):
        for j in range(n_networks):
            if i != j and connectivity[i, j] != 0:
                weight = np.abs(connectivity[i, j])
                alpha = weight / max_weight
                width = 0.5 + 2 * alpha
                ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                       'r-', alpha=alpha*0.5, linewidth=width, zorder=1)
    
    # Draw nodes
    for i in range(n_networks):
        ax.scatter(x_pos[i], y_pos[i], s=200, 
                  c='lightblue', edgecolors='black', linewidths=2, zorder=2)
        ax.text(x_pos[i], y_pos[i], f'N{i+1}', 
               ha='center', va='center', fontweight='bold', fontsize=9, zorder=3)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Network Graph\n(Line thickness = connection strength)', 
                 fontsize=11, fontweight='bold')


def visualize_connection_strength_distribution(ax, model):
    """Visualize the distribution of connection strengths."""
    # Collect all connection strengths
    all_weights = []
    
    # Local network weights
    local_net = model.local_networks[0]
    all_weights.extend(local_net.W_ee.cpu().numpy().flatten())
    all_weights.extend(local_net.W_ei.cpu().numpy().flatten())
    all_weights.extend(local_net.W_ie.cpu().numpy().flatten())
    all_weights.extend(local_net.W_ii.cpu().numpy().flatten())
    
    # Inter-network weights (sample)
    inter_weights = model.W_inter.cpu().numpy()
    all_weights.extend(inter_weights.flatten())
    
    all_weights = np.array(all_weights)
    all_weights = all_weights[all_weights != 0]  # Remove zero connections
    
    ax.hist(all_weights, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Connection Strength', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Connection Strength Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_weight = np.mean(all_weights)
    std_weight = np.std(all_weights)
    ax.text(0.02, 0.98, f'Mean: {mean_weight:.4f}\nStd: {std_weight:.4f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def visualize_statistics_table(ax, model):
    """Display network statistics in a table format."""
    ax.axis('off')
    
    stats = model.get_network_statistics()
    
    # Create table data
    table_data = [
        ['Parameter', 'Value'],
        ['Number of Networks', f"{stats['n_networks']}"],
        ['Units per Network', f"{stats['n_units_per_network']}"],
        ['Excitatory Units', f"{stats['n_excitatory']} ({stats['excitatory_ratio']*100:.0f}%)"],
        ['Inhibitory Units', f"{stats['n_inhibitory']} ({100-stats['excitatory_ratio']*100:.0f}%)"],
        ['Total Units', f"{stats['total_units']}"],
        ['Inter-network Connections', f"{stats['inter_network_connections']:,}"],
        ['Inhibition Strength', f"{model.inhibition_strength}"],
        ['Gaussian Sigma (σ)', f"{model.gaussian_sigma}"],
        ['Time Constant (τ)', f"{model.tau}"],
        ['Time Step (dt)', f"{model.dt}"],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Network Statistics', fontsize=12, fontweight='bold', pad=20)


if __name__ == "__main__":
    print("Creating RNN model...")
    model = create_rnn_model(
        n_networks=12,
        n_units_per_network=500,
        excitatory_ratio=0.8,
        inhibition_strength=1.0,
        gaussian_sigma=2.0
    )
    
    print("Generating visualizations...")
    visualize_network_architecture(model)
    print("Visualization complete!")

