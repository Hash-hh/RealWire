"""
Enhanced GNN architecture with more advanced layers and readout functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, GraphConv, 
    global_mean_pool, global_add_pool, global_max_pool,
    JumpingKnowledge, PNAConv, BatchNorm
)

class EnhancedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=4,
                 dropout=0.2, layer_type="pna", readout="combined",
                 use_batch_norm=True, task="regression", out_channels=1):
        """
        Advanced GNN with multiple layer types and readout functions
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Hidden dimensions
            num_layers: Number of message passing layers
            dropout: Dropout rate
            layer_type: 'gcn', 'gat', 'gin', 'graph', or 'pna'
            readout: 'mean', 'sum', 'max', or 'combined'
            use_batch_norm: Whether to use batch normalization
            task: 'regression' or 'classification'
            out_channels: Number of output channels (default 1 for regression)
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout = readout
        self.use_batch_norm = use_batch_norm
        self.task = task
        
        # Input embedding layer
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # Message passing layers
        self.convs = nn.ModuleList()
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            
        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_channels
            else:
                in_dim = hidden_channels
            
            if layer_type == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_channels))
            elif layer_type == "gat":
                # Multi-head attention with 8 heads
                self.convs.append(GATConv(in_dim, hidden_channels // 8, heads=8))
            elif layer_type == "gin":
                nn_layer = nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(nn_layer))
            elif layer_type == "graph":
                self.convs.append(GraphConv(in_dim, hidden_channels))
            elif layer_type == "pna":
                # PNA requires aggregators and scalers
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                
                self.convs.append(PNAConv(
                    in_dim, hidden_channels, 
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=None,  # Will be computed in forward pass
                    edge_dim=None,  # Can add edge features if needed
                    towers=4,  # Split channels into multiple towers
                    pre_layers=1,  # MLP layers before aggregation
                    post_layers=1,  # MLP layers after aggregation
                ))
            
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Jumping Knowledge with attention
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_channels, num_layers=num_layers)

        # Calculate the actual input dimension for the output layer
        if self.readout == "combined":
            # For combined readout (mean, sum, max), the dim is multiplied by 3
            jk_output_dim = hidden_channels * (num_layers+1)  # we are also including the original input layer
            readout_dim = jk_output_dim * 3
        else:
            # For single readout, no multiplication by 3
            jk_output_dim = hidden_channels * num_layers
            readout_dim = jk_output_dim

        # Update output layers with correct input dimension
        self.output_layers = nn.Sequential(
            nn.Linear(readout_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            output: Graph-level prediction
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Calculate node degrees for PNA if needed
        if any(isinstance(conv, PNAConv) for conv in self.convs):
            from torch_geometric.utils import degree
            deg = degree(edge_index[0], x.size(0)).long()
        
        # Input embedding
        x = self.node_encoder(x)
        
        # Save all intermediate representations
        xs = [x]
        
        # Message passing layers
        for i, conv in enumerate(self.convs):
            if isinstance(conv, PNAConv):
                x = conv(xs[-1], edge_index, deg=deg)
            else:
                x = conv(xs[-1], edge_index)
                
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
                
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        
        # Apply JK-connection
        x = self.jk(xs)
        
        # Readout function
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        else:  # combined
            x_mean = global_mean_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_sum, x_max], dim=1)
        
        # Output layers
        output = self.output_layers(x)
        
        # Apply activation function for classification
        if self.task == "classification":
            output = torch.sigmoid(output)
            
        return output
