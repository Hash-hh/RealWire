import torch
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform


class FeatureTransform(BaseTransform):
    """Transform node and edge features to float type and optionally one-hot encode them."""

    def __init__(self, one_hot=True, num_atom_types=28, num_bond_types=4):
        """
        Args:
            one_hot (bool): Whether to one-hot encode features or just convert to float
            num_atom_types (int): Number of atom types for one-hot encoding
            num_bond_types (int): Number of bond types for one-hot encoding
        """
        self.one_hot = one_hot
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types

    def __call__(self, data):
        # Convert node features to float or one-hot
        if data.x is not None:
            if self.one_hot:
                # One-hot encode node features
                x_one_hot = F.one_hot(data.x.view(-1), self.num_atom_types).float()
                data.x = x_one_hot
            else:
                # Simply convert to float
                data.x = data.x.float()

        # Convert edge attributes to float or one-hot if they exist
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if self.one_hot:
                # One-hot encode edge features
                edge_attr_one_hot = F.one_hot(data.edge_attr.view(-1), self.num_bond_types).float()
                data.edge_attr = edge_attr_one_hot
            else:
                # Simply convert to float
                data.edge_attr = data.edge_attr.float()

        return data
