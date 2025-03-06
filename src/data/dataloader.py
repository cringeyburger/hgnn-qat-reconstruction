import random
from pathlib import Path
from typing import List, Dict, Iterable, Tuple, Optional, Any

import torch
import scipy as sp
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torch_geometric.data import Data
import frnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Adapted from https://github.com/ryanliu30/HierarchicalGNN/tree/main


def load_dataset_paths(input_dir: str, datatype_names: Iterable[str]) -> List[Path]:
    all_events = [
        file
        for name in datatype_names
        for file in (Path(input_dir) / name).glob("*")
        if file.is_file()
    ]
    random.seed(42)
    random.shuffle(all_events)
    return all_events


class TrackMLDataset(Dataset):
    def __init__(
        self, dirs: List[str], hparams: Dict[str, Any], stage="train", device="cpu"
    ):
        super().__init__()
        self.dirs = dirs
        self.num = len(dirs)
        self.device = device
        self.stage = stage
        self.hparams = hparams

    def __getitem__(self, key: int) -> Data:

        # load the event
        event = torch.load(self.dirs[key], map_location=torch.device(self.device))
        if "1GeV" in f"{self.dirs[key]}":
            event = Data.from_dict(event.__dict__)  # handle older PyG data format

        # the MASK tensor filter out hits from event
        mask = event.pid == event.pid if self.hparams["noise"] else event.pid != 0
        # If using noise then only filter out those with nan PID
        # If not using noise then filter out those with PID 0, which represent that they are noise

        if self.hparams["hard_ptcut"] > 0:
            mask = mask & (
                event.pt > self.hparams["hard_ptcut"]
            )  # Hard background cut in pT
        if self.hparams["remove_isolated"]:
            node_mask = torch.zeros(event.pid.shape).bool()
            node_mask[event.edge_index.unique()] = torch.ones(
                1
            ).bool()  # Keep only those nodes with edges attached to it
            mask = mask & node_mask

        # Set the pT of noise hits to be 0
        event.pt = torch.where(event.pid == 0, torch.tensor(0.0), event.pt)

        # Provide inverse mask to invert the change when necessary
        # (e.g. track evaluation with not modified files)
        inverse_mask = torch.zeros(len(event.pid)).long()
        inverse_mask[mask] = torch.arange(mask.sum())
        event.inverse_mask = torch.arange(len(mask))[mask]

        # Compute number of hits (nhits) of each particle
        _, inverse, counts = event.pid.unique(return_inverse=True, return_counts=True)
        event.nhits = counts[inverse]

        if self.hparams["primary"]:
            event.signal_mask = torch.logical_and(
                event.nhits >= self.hparams["n_hits"], event.primary == 1
            )
        else:
            event.signal_mask = event.nhits >= self.hparams["n_hits"]

        # Randomly remove edges if needed
        if self.hparams.get("edge_dropping_ratio", 0) != 0:
            edge_mask = (
                torch.rand(event.edge_index.shape[1])
                >= self.hparams["edge_dropping_ratio"]
            )
            event.edge_index = event.edge_index[:, edge_mask]
            event.y, event.y_pid = event.y[edge_mask], event.y_pid[edge_mask]

        for i in ["y", "y_pid"]:
            graph_mask = mask[event.edge_index].all(0)
            event[i] = event[i][graph_mask]

        for i in ["modulewise_true_edges", "signal_true_edges", "edge_index"]:
            event[i] = event[i][:, mask[event[i]].all(0)]
            event[i] = inverse_mask[event[i]]

        for i in ["x", "cell_data", "pid", "hid", "pt", "signal_mask"]:
            event[i] = event[i][mask]

        if self.hparams["primary"]:
            event.primary = event.primary[mask]

        event.dir = self.dirs[key]

        return event

    def __len__(self) -> int:
        return self.num


def graph_intersection(
    pred_graph: torch.Tensor,
    truth_graph: torch.Tensor,
    using_weights: bool = False,
    weights_bidir: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    l1 = pred_graph.cpu().numpy() if torch.is_tensor(pred_graph) else pred_graph
    l2 = truth_graph.cpu().numpy() if torch.is_tensor(truth_graph) else truth_graph

    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1, e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()
        del weights_list, l2
        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.stack(
        [torch.from_numpy(e_intersection.row), torch.from_numpy(e_intersection.col)]
    ).long()

    y = torch.from_numpy(e_intersection.data > 0)
    del e_intersection

    return (new_pred_graph, y, new_weights) if using_weights else (new_pred_graph, y)


def get_activation(activation_name: Optional[str]) -> Optional[nn.Module]:
    if activation_name is None:
        return None
    try:
        return getattr(nn, activation_name)()
    except AttributeError as exc:
        raise ValueError(f"Unsupported activation function: {activation_name}") from exc


def make_layer(
    in_size: int, out_size: int, activation: Optional[nn.Module], use_layer_norm: bool
) -> List[nn.Module]:
    layer = [nn.Linear(in_size, out_size)]
    if use_layer_norm:
        layer.append(nn.LayerNorm(out_size))
    if activation is not None:
        layer.append(activation)
    return layer


def make_mlp(
    input_size: int,
    hidden_size: int,
    output_size: int,
    hidden_layers: int,
    hidden_activation: str = "GELU",
    output_activation: str = "GELU",
    layer_norm: bool = False,
) -> nn.Sequential:
    """Construct an MLP with specified fully-connected layers."""

    hidden_activation = get_activation(hidden_activation)
    output_activation = get_activation(output_activation)

    layers = []
    sizes = [input_size] + [hidden_size] * (hidden_layers - 1) + [output_size]
    # Hidden layers
    for i in range(hidden_layers - 1):
        layers.extend(make_layer(sizes[i], sizes[i + 1], hidden_activation, layer_norm))
    layers.extend(make_layer(sizes[-2], sizes[-1], output_activation, layer_norm))

    return nn.Sequential(*layers)


def find_neighbors(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    r_max: float = 1.0,
    k_max: int = 10,
) -> torch.Tensor:
    embedding1 = (
        embedding1.clone()
        .detach()
        .reshape((1, embedding1.shape[0], embedding1.shape[1]))
    )
    embedding2 = (
        embedding2.clone()
        .detach()
        .reshape((1, embedding2.shape[0], embedding2.shape[1]))
    )

    _, idxs, _, _ = frnn.frnn_grid_points(
        points1=embedding1,
        points2=embedding2,
        lengths1=None,
        lengths2=None,
        K=k_max,
        r=r_max,
    )
    return idxs.squeeze(0)


def frnn_graph(embeddings: torch.Tensor, r: float, k: int) -> torch.Tensor:
    idxs = find_neighbors(embeddings, embeddings, r_max=r, k_max=k)

    positive_idxs = idxs.squeeze() >= 0
    ind = (
        torch.arange(idxs.shape[0], device=positive_idxs.device)
        .unsqueeze(1)
        .expand(idxs.shape)
    )
    edges = torch.stack([ind[positive_idxs], idxs[positive_idxs]], dim=0)
    return edges
