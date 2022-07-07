from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from typing import Optional
import numpy as np
import torch


class CustomData(Data):
    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        smiles: str = None,
    ):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
        )
        self.smiles = smiles


class GraphData:
    """GraphData class

    This data class is almost same as `torch_geometric.data.Data
    <https://pytorch-geometric.readthedocs.io/en/\
        latest/modules/data.html#torch_geometric.data.Data>`_.

    Attributes
    ----------
    node_features: np.ndarray
      Node feature matrix with shape [num_nodes, num_node_features]
    edge_index: np.ndarray, dtype int
      Graph connectivity in COO format with shape [2, num_edges]
    edge_features: np.ndarray, optional (default None)
      Edge feature matrix with shape [num_edges, num_edge_features]
    node_pos_features: np.ndarray, optional (default None)
      Node position matrix with shape [num_nodes, num_dimensions].
    num_nodes: int
      The number of nodes in the graph
    num_node_features: int
      The number of features per node in the graph
    num_edges: int
      The number of edges in the graph
    num_edges_features: int, optional (default None)
      The number of features per edge in the graph

    Examples
    --------
    >>> import numpy as np
    >>> node_features = np.random.rand(5, 10)
    >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    >>> graph = GraphData(node_features=node_features, edge_index=edge_index)
    """

    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
        node_pos_features: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        node_features: np.ndarray
          Node feature matrix with shape [num_nodes, num_node_features]
        edge_index: np.ndarray, dtype int
          Graph connectivity in COO format with shape [2, num_edges]
        edge_features: np.ndarray, optional (default None)
          Edge feature matrix with shape [num_edges, num_edge_features]
        node_pos_features: np.ndarray, optional (default None)
          Node position matrix with shape [num_nodes, num_dimensions].
        """
        # validate params
        if isinstance(node_features, np.ndarray) is False:
            raise ValueError("node_features must be np.ndarray.")

        if isinstance(edge_index, np.ndarray) is False:
            raise ValueError("edge_index must be np.ndarray.")
        elif issubclass(edge_index.dtype.type, np.integer) is False:
            raise ValueError("edge_index.dtype must contains integers.")
        elif edge_index.shape[0] != 2:
            raise ValueError("The shape of edge_index is [2, num_edges].")
        elif np.max(edge_index) >= len(node_features):
            raise ValueError("edge_index contains the invalid node number.")

        if edge_features is not None:
            if isinstance(edge_features, np.ndarray) is False:
                raise ValueError("edge_features must be np.ndarray or None.")
            elif edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    "The first dimension of edge_features must be the \
                          same as the second dimension of edge_index."
                )

        if node_pos_features is not None:
            if isinstance(node_pos_features, np.ndarray) is False:
                raise ValueError("node_pos_features must be np.ndarray or None.")
            elif node_pos_features.shape[0] != node_features.shape[0]:
                raise ValueError(
                    "The length of node_pos_features must be the same as the \
                          length of node_features."
                )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_pos_features = node_pos_features
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = edge_index.shape[1]
        if self.edge_features is not None:
            self.num_edge_features = self.edge_features.shape[1]

    def to_pyg_graph(self):
        """Convert to PyTorch Geometric graph data instance

        Returns
        -------
        torch_geometric.data.Data
          Graph data for PyTorch Geometric

        Note
        ----
        This method requires PyTorch Geometric to be installed.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ModuleNotFoundError:
            raise ImportError(
                "This function requires PyTorch Geometric to be installed."
            )

        edge_features = self.edge_features
        if edge_features is not None:
            edge_features = torch.from_numpy(self.edge_features).float()
        node_pos_features = self.node_pos_features
        if node_pos_features is not None:
            node_pos_features = torch.from_numpy(self.node_pos_features).float()

        return Data(
            x=torch.from_numpy(self.node_features).float(),
            edge_index=torch.from_numpy(self.edge_index).long(),
            edge_attr=edge_features,
            pos=node_pos_features,
        )


class CustomGraphData(GraphData):
    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
        node_pos_features: Optional[np.ndarray] = None,
    ):
        super().__init__(
            node_features,
            edge_index,
            edge_features,
            node_pos_features,
        )

    def to_pyg_graph(self):
        """Convert to PyTorch Geometric graph data instance

        Returns
        -------
        torch_geometric.data.Data
        Graph data for PyTorch Geometric

        Note
        ----
        This method requires PyTorch Geometric to be installed.
        """

        edge_features = self.edge_features
        if edge_features is not None:
            edge_features = torch.from_numpy(self.edge_features).float()
        node_pos_features = self.node_pos_features
        if node_pos_features is not None:
            node_pos_features = torch.from_numpy(self.node_pos_features).float()
        return CustomData(
            x=torch.from_numpy(self.node_features).float(),
            edge_index=torch.from_numpy(self.edge_index).long(),
            edge_attr=edge_features,
            pos=node_pos_features,
        )
