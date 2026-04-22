import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def build_graph(sample):
    vars_data = sample["variables"]
    cons_data = sample["constraints"]
    edges_data = sample["edges"]

    n_vars = len(vars_data)
    n_cons = len(cons_data)

    N = n_vars + n_cons

    node_features = []

    # ----- variable nodes -----

    for v in vars_data:

        is_frac = abs(v["solution"] - round(v["solution"])) > 1e-6

        feats = [
            1.0,
            0.0,

            float(v["is_integer"]),
            float(is_frac),

            v["obj_coef"],
            v["solution"],

            v["lb"],
            v["ub"],

            v.get("reduced_cost", 0.0),
            v.get("pseudo_up", 0.0),
            v.get("pseudo_down", 0.0),
        ]

        node_features.append(feats)

    # ----- constraint nodes -----

    for c in cons_data:

        feats = [
            0.0,
            1.0,

            0.0,
            0.0,

            c["rhs"],
            c.get("slack", 0.0),

            c.get("dual", 0.0),
            c.get("nnz", 0),

            0.0,
            0.0,
            0.0,
        ]

        node_features.append(feats)

    node_fts = torch.tensor(node_features, dtype=torch.float)

    # ----- edges -----

    edge_index = []
    edge_features = []

    for e in edges_data:

        v = e["var_index"]
        c = e["con_index"]

        coef = e["value"]

        edge_index.append([v, n_vars + c])
        edge_index.append([n_vars + c, v])

        edge_features.append([
            coef,
            abs(coef),
            np.sign(coef),
        ])

        edge_features.append([
            coef,
            abs(coef),
            np.sign(coef),
        ])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_fts = torch.tensor(edge_features, dtype=torch.float)

    # ----- label -----

    y = torch.tensor(sample["branching_variable"])

    return Data(
        node_fts=node_fts,
        edge_fts=edge_fts,
        edge_index=edge_index,
        y=y,
        n_vars=torch.tensor(n_vars),
    )

def create_milp_evolve_loader(dataset, batch_size):

    graphs = []

    for sample in dataset:

        g = build_graph(sample)

        graphs.append(g)

    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=True,
    )