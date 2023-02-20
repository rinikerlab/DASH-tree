from typing import List, Optional, Tuple

from torch import Tensor, is_tensor
from torch.nn import Module
from torch_geometric.nn import GNNExplainer

# Above line does not work anymore fix 10.01.2023
# from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from serenityff.charge.gnn.utils import CustomData


class FixedGNNExplainer(GNNExplainer):
    def subgraph(self, node_idx: int, x: Tensor, edge_index: Tensor, **kwargs):
        """
        Overloading the torch_geometric.nn.models.Explainer function. It is
        already fixed in the github with commit 7526c8b but not yet updated
        in the version distributed by conda. This will be unecessary from
        the next pyg conda release on.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (Tensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        Returns:
            (Tensor, Tensor, LongTensor, LongTensor, LongTensor, dict)
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx,
            self.num_hops,
            edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            flow=self._flow(),
        )

        x = x[subset]
        kwargs_new = {}
        for key, value in kwargs.items():
            if is_tensor(value) and value.size(0) == num_nodes:
                kwargs_new[key] = value[subset]
            elif is_tensor(value) and value.size(0) == num_edges:
                kwargs_new[key] = value[edge_mask]
            else:
                kwargs_new[key] = value  # TODO: this is not in PGExplainer
        return x, edge_index, mapping, edge_mask, subset, kwargs_new


class Explainer:
    """
    Class holding a pytorch_geometric GNNExplainer object
    and some settings.
    """

    def __init__(
        self,
        model: Module,
        epochs: Optional[int] = 2000,
        verbose: Optional[bool] = False,
    ) -> None:
        """
        Args:
            model (Module): model to be
            epochs (Optional[int], optional): _description_. Defaults to 2000.
            verbose (Optional[bool], optional): _description_. Defaults to False.
        """
        self.gnn_explainer = FixedGNNExplainer(
            model=model,
            epochs=epochs,
            log=verbose,
            return_type="regression",
            feat_mask_type="scalar",
        )
        self.gnnverbose = verbose

    @property
    def gnn_explainer(self) -> FixedGNNExplainer:
        return self._gnn_explainer

    @gnn_explainer.setter
    def gnn_explainer(self, value) -> None:
        if not isinstance(value, FixedGNNExplainer):
            raise TypeError("explainer needs to be of type GNNExplainer")
        else:
            self._gnn_explainer = value

    @property
    def gnnverbose(self) -> bool:
        return self._gnnverbose

    @gnnverbose.setter
    def gnnverbose(self, value: bool) -> None:
        self._gnnverbose = value
        self._gnn_explainer.log = value

    def _explain(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        parser to the GNNExplainers' explain node Function.
        Give all args also needed in the models forward function.

        Returns:
            Tuple[Tensor, Tensor]: Node and Edge Attentions for node in graph.
        """
        return self.gnn_explainer.explain_node(*args, **kwargs)

    def _explain_atom(
        self,
        node_idx: int,
        graph: CustomData,
    ) -> Tuple[Tensor, Tensor]:
        """
        Explains a given node in an Graph object.

        Args:
            node_idx (int): Idx of node to be explained.
            data (CustomData): Graph containing the node to be explained.

        Returns:
            Tuple[Tensor, Tensor]: Node and Edge Attentions for node in graph.
        """
        node, edge = self._explain(
            node_idx,
            graph.x,
            graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=graph.batch,
            molecule_charge=graph.molecule_charge,
        )
        return node, edge

    def explain_molecule(
        self,
        graph: CustomData,
    ) -> Tuple[List[Tensor]]:
        """
        Explains all Atom Predictions in a Molecule

        Args:
            data (CustomData): Graph representing the molecule.

        Returns:
            Tuple[List[Tensor]]: Lists of Node and Edge Attentions.
        """
        nodes: List[Tensor] = []
        edges: List[Tensor] = []
        for i in range(graph.num_nodes):
            node, edge = self._explain_atom(node_idx=i, graph=graph)
            nodes.append(node)
            edges.append(edge)
        return nodes, edges
