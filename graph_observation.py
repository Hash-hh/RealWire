"""
Advanced graph feature extraction for RL observations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool

class GraphObservationExtractor(nn.Module):
    """
    Extract a fixed-size embedding from a graph for RL observation
    """
    def __init__(self, in_channels, hidden_channels=64, embedding_size=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.embedding_layer = nn.Linear(hidden_channels * 3, embedding_size)
        
        # Node-level properties extractors
        self.node_degree_embedding = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8)
        )
        
        # Graph-level property extractors
        self.graph_stats_embedding = nn.Sequential(
            nn.Linear(5, 16),  # 5 graph-level statistics
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Final projection
        self.final_projection = nn.Linear(embedding_size + 8 + 16, embedding_size)
        
    def forward(self, x, edge_index, batch=None):
        """
        Extract a fixed-size representation from graph
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Optional batch assignment [num_nodes]
            
        Returns:
            observation: A fixed-size tensor for RL [embedding_size]
        """
        # If batch is None (single graph), create dummy batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Calculate node degrees
        from torch_geometric.utils import degree
        node_degrees = degree(edge_index[0], x.size(0), dtype=torch.float)
        
        # Graph structure embedding
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        # Multiple pooling strategies
        pooled_mean = global_mean_pool(h, batch)
        pooled_sum = global_add_pool(h, batch)
        pooled_max = global_max_pool(h, batch)
        pooled = torch.cat([pooled_mean, pooled_sum, pooled_max], dim=1)
        
        # Graph structure embedding
        graph_embedding = F.relu(self.embedding_layer(pooled))
        
        # Node degree information
        degree_embedding = self.node_degree_embedding(node_degrees.unsqueeze(-1))
        degree_embedding = global_mean_pool(degree_embedding, batch)
        
        # Calculate graph-level statistics
        from torch_geometric.utils import density, segregation, homophily
        
        # Graph size (normalized by dataset max size for stability)
        max_nodes = 100  # Adjust based on your dataset
        graph_size = torch.tensor([x.size(0) / max_nodes], device=x.device)
        
        # Edge density
        edge_count = edge_index.size(1) / 2  # Assuming undirected
        max_edges = x.size(0) * (x.size(0) - 1) / 2
        edge_density = torch.tensor([edge_count / max_edges], device=x.device)
        
        # Average clustering coefficient (approximate for efficiency)
        # In practice you'd use nx.average_clustering or networkx methods
        # Here we just use edge density as a proxy
        avg_clustering = edge_density
        
        # Average path length approximation
        # For efficiency, we use a heuristic based on graph diameter and density
        # In practice you might use nx.average_shortest_path_length
        avg_path_length = torch.tensor([1.0 / (edge_density + 1e-6)], device=x.device)
        
        # Number of connected components (simplified)
        # In practice, you'd use networkx or union-find algorithm
        # Here we use a dummy value
        connected_components = torch.tensor([1.0], device=x.device)
        
        graph_stats = torch.cat([
            graph_size, 
            edge_density,
            avg_clustering, 
            avg_path_length, 
            connected_components
        ], dim=0).unsqueeze(0)
        
        graph_stats_embedding = self.graph_stats_embedding(graph_stats)
        
        # Combine all embeddings
        combined = torch.cat([
            graph_embedding,
            degree_embedding,
            graph_stats_embedding
        ], dim=1)
        
        # Final projection to desired embedding size
        observation = self.final_projection(combined)
        
        return observation.view(-1)  # Flatten to 1D tensor