"""
Advanced edge operations for graph rewiring.
"""
import torch
import numpy as np
from enum import Enum

class EdgeOperation(Enum):
    ADD = 0
    REMOVE = 1
    REWIRE = 2  # remove one edge, add another

class EdgeOperator:
    def __init__(self, strategy="degree_based"):
        """
        strategy options:
        - "random": Random edge candidates
        - "degree_based": Favor high/low degree nodes
        - "feature_similarity": Connect similar nodes
        - "structural": Based on graph topology (e.g., transitivity)
        """
        self.strategy = strategy
        
    def generate_candidates(self, data, max_candidates=50):
        """Generate candidate edge operations based on strategy"""
        num_nodes = data.x.size(0)
        edge_index = data.edge_index
        candidates = []
        
        if self.strategy == "random":
            # Random edge additions
            for _ in range(max_candidates // 3):
                i, j = np.random.randint(0, num_nodes, size=2)
                if i != j:  # Avoid self-loops
                    candidates.append((EdgeOperation.ADD, (i, j)))
            
            # Random edge removals
            existing_edges = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                                for i in range(edge_index.size(1))])
            for edge in list(existing_edges)[:max_candidates // 3]:
                candidates.append((EdgeOperation.REMOVE, edge))
                
            # Random edge rewires
            for _ in range(max_candidates // 3):
                if len(existing_edges) > 0:
                    old_edge = list(existing_edges)[np.random.randint(len(existing_edges))]
                    i, j = np.random.randint(0, num_nodes, size=2)
                    if i != j and (i, j) not in existing_edges:
                        candidates.append((EdgeOperation.REWIRE, (old_edge, (i, j))))
        
        elif self.strategy == "degree_based":
            # Calculate node degrees
            degrees = torch.zeros(num_nodes, dtype=torch.long)
            for i in range(edge_index.size(1)):
                degrees[edge_index[0, i]] += 1
            
            # Connect high-degree to low-degree nodes
            sorted_nodes = torch.argsort(degrees)
            high_degree = sorted_nodes[-max_candidates//10:]
            low_degree = sorted_nodes[:max_candidates//10]
            
            existing_edges = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                                for i in range(edge_index.size(1))])
            
            for high in high_degree:
                for low in low_degree:
                    high, low = high.item(), low.item()
                    if high != low and (high, low) not in existing_edges:
                        candidates.append((EdgeOperation.ADD, (high, low)))
                        if len(candidates) >= max_candidates:
                            break
                if len(candidates) >= max_candidates:
                    break
        
        elif self.strategy == "feature_similarity":
            # Connect nodes with similar features
            node_features = data.x.cpu().numpy()
            # Use cosine similarity for feature comparison
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(node_features)
            
            existing_edges = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                                for i in range(edge_index.size(1))])
            
            # Get top similar pairs that aren't already connected
            pairs = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if (i, j) not in existing_edges:
                        pairs.append((i, j, similarity[i, j]))
            
            # Sort by similarity (highest first)
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Take top pairs as candidates
            for i, j, _ in pairs[:max_candidates]:
                candidates.append((EdgeOperation.ADD, (i, j)))
        
        return candidates

    def apply_operation(self, data, operation, edge_pair):
        """Apply the edge operation to the graph"""
        edge_index = data.edge_index.cpu()
        edge_list = edge_index.t().tolist()
        
        if operation == EdgeOperation.ADD:
            i, j = edge_pair
            if [i, j] not in edge_list:
                edge_list.append([i, j])
            if [j, i] not in edge_list:  # For undirected graphs
                edge_list.append([j, i])
        
        elif operation == EdgeOperation.REMOVE:
            i, j = edge_pair
            if [i, j] in edge_list:
                edge_list.remove([i, j])
            if [j, i] in edge_list:  # For undirected graphs
                edge_list.remove([j, i])
        
        elif operation == EdgeOperation.REWIRE:
            old_edge, new_edge = edge_pair
            i, j = old_edge
            if [i, j] in edge_list:
                edge_list.remove([i, j])
            if [j, i] in edge_list:  # For undirected graphs
                edge_list.remove([j, i])
            
            i, j = new_edge
            if [i, j] not in edge_list:
                edge_list.append([i, j])
            if [j, i] not in edge_list:  # For undirected graphs
                edge_list.append([j, i])
        
        new_edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        data.edge_index = new_edge_index.to(data.x.device)
        return data
