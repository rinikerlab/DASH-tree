from torch_geometric.nn import AttentiveFP
from torch.nn import functional as F
from torch import nn
from typing import Optional
import torch


class ChargeCorrectedNodeWiseAttentiveFP(AttentiveFP):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            edge_dim,
            num_layers,
            num_timesteps,
            dropout,
        )

        self._adalin1 = nn.Linear(hidden_channels, hidden_channels)
        self._adalin2 = nn.Linear(hidden_channels, hidden_channels)
        self._adalin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch, molecule_charge):
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # MLP decoding
        x = self._adalin1(x).relu_()
        x = self._adalin2(x).relu_()
        x = self._adalin3(x)

        for b in torch.unique(batch):
            batch_location = torch.where(batch == b, True, False)
            mol_charge = molecule_charge[b]
            charge_correction = mol_charge / len(x[batch_location])
            x[batch_location] = x[batch_location] - torch.mean(x[batch_location]) + charge_correction
        return x
