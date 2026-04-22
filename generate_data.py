import math

import networkx as nx
import numpy as np
import torch
import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from configs import base_config

from milp_task import bnb_trace, random_feasible_milp

class ProblemInstance:
    def __init__(self, adj, start, weighted, randomness):
        self.adj = np.copy(adj)
        self.start = start
        self.weighted = weighted
        self.randomness = np.copy(randomness)
        self.edge_index = np.stack(np.nonzero(adj + np.eye(adj.shape[0])))

        self.out_nodes = [[] for _ in range(adj.shape[0])]
        for x, y in self.edge_index[:, self.edge_index[0] != self.edge_index[1]].T:
            self.out_nodes[x].append(y)
        random_pos = np.random.uniform(0.0, 1.0, (adj.shape[0],))
        self.pos = random_pos[np.argsort(random_pos)]


def push_states(
    node_states, edge_states, scalars, cur_step_nodes, cur_step_edges, cur_step_scalars
):
    node_states.append(np.stack(cur_step_nodes, axis=-1))
    edge_states.append(np.stack(cur_step_edges, axis=-1))
    scalars.append(np.stack(cur_step_scalars, axis=-1))

def milp(instance, max_steps=None):
    states, heuristic_y, branch_y = bnb_trace(instance, max_steps=max_steps)

    node_fts_list = []
    edge_fts_list = []
    scalars_list = []

    for node_fts, edge_fts, scalars in states:
        node_fts_list.append(node_fts)
        edge_fts_list.append(edge_fts)
        scalars_list.append(scalars)

    if len(node_fts_list) == 0:
        raise RuntimeError("MILP trace is empty")

    if max_steps is None:
        max_steps = len(node_fts_list)

    last_node = node_fts_list[-1]
    last_edge = edge_fts_list[-1]
    last_scalar = scalars_list[-1]

    while len(node_fts_list) < max_steps:
        node_fts_list.append(last_node.copy())
        edge_fts_list.append(last_edge.copy())
        scalars_list.append(last_scalar.copy())

    node_fts_list = node_fts_list[:max_steps]
    edge_fts_list = edge_fts_list[:max_steps]
    scalars_list = scalars_list[:max_steps]

    return (
        np.stack(node_fts_list),
        np.stack(edge_fts_list),
        np.stack(scalars_list),
    )

def bfs(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    visited = np.zeros(n, dtype=np.int32)
    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    cur_scalars = instance.pos[instance.edge_index[0]]

    visited[instance.start] = 1

    push_states(
        node_states,
        edge_states,
        scalars,
        (visited,),
        (pointers, self_loops),
        (cur_scalars,),
    )

    layer = [instance.start]

    while layer:
        next_layer = []
        layer.sort()
        for node in layer:
            for out in instance.out_nodes[node]:
                if visited[out] == 0:
                    visited[out] = 1
                    next_layer.append(out)
                    assert pointers[out][out] == 1
                    pointers[out][out] = 0
                    pointers[out][node] = 1
        layer = next_layer
        push_states(
            node_states,
            edge_states,
            scalars,
            (visited,),
            (pointers, self_loops),
            (cur_scalars,),
        )

    while len(node_states) < n:
        push_states(
            node_states,
            edge_states,
            scalars,
            (visited,),
            (pointers, self_loops),
            (cur_scalars,),
        )
    return np.array(node_states), np.array(edge_states), np.array(scalars)


def dfs(instance: ProblemInstance):
    n = instance.adj.shape[0]

    node_states = []
    edge_states = []
    scalars = []

    not_in_the_stack = np.ones(n, dtype=np.int32)
    top_of_the_stack = np.zeros(n, dtype=np.int32)
    in_the_stack = np.zeros(n, dtype=np.int32)
    pre_end = np.zeros(n, dtype=np.int32)

    pointers = np.eye(n, dtype=np.int32)
    stack_update = np.zeros((n, n), dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    cur_scalars = instance.pos[instance.edge_index[0]]

    top_of_the_stack[instance.start] = 1
    not_in_the_stack[instance.start] = 0

    push_states(
        node_states,
        edge_states,
        scalars,
        (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
        (pointers, stack_update, self_loops),
        (cur_scalars,),
    )

    def rec_dfs(current_node, prev_node=-1):
        assert top_of_the_stack[current_node] == 1
        assert prev_node == -1 or in_the_stack[prev_node] == 1
        for out in instance.out_nodes[current_node]:
            if not_in_the_stack[out]:
                in_the_stack[current_node] = 1

                stack_update[current_node][out] = 1
                push_states(
                    node_states,
                    edge_states,
                    scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops),
                    (cur_scalars,),
                )
                stack_update[current_node][out] = 0

                top_of_the_stack[current_node] = 0
                top_of_the_stack[out] = 1

                not_in_the_stack[out] = 0
                pointers[out][current_node] = 1
                pointers[out][out] = 0

                stack_update[out][current_node] = 1
                push_states(
                    node_states,
                    edge_states,
                    scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops),
                    (cur_scalars,),
                )
                stack_update[out][current_node] = 0

                rec_dfs(out, current_node)

                top_of_the_stack[current_node] = 1
                top_of_the_stack[out] = 0
                in_the_stack[current_node] = 0
                pre_end[out] = 0

                stack_update[current_node][out] = 1
                push_states(
                    node_states,
                    edge_states,
                    scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops),
                    (cur_scalars,),
                )
                stack_update[current_node][out] = 0

        pre_end[current_node] = 1

        stack_update[current_node][current_node] = 1
        push_states(
            node_states,
            edge_states,
            scalars,
            (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
            (pointers, stack_update, self_loops),
            (cur_scalars,),
        )
        stack_update[current_node][current_node] = 0

    rec_dfs(instance.start)

    return np.array(node_states), np.array(edge_states), np.array(scalars)


def mst(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    in_queue = np.zeros(n, dtype=np.int32)
    in_tree = np.zeros(n, dtype=np.int32)

    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    node_scalars = instance.pos

    def compute_current_scalars(node_scalars):
        scalars = instance.adj[instance.edge_index[0], instance.edge_index[1]]
        scalars[instance.edge_index[0] == instance.edge_index[1]] = node_scalars
        return scalars

    in_queue[instance.start] = 1
    node_scalars[instance.start] = 0.0

    push_states(
        node_states,
        edge_states,
        scalars,
        (in_queue, in_tree),
        (pointers, self_loops),
        (compute_current_scalars(node_scalars),),
    )

    for _ in range(1, n):
        node = np.argsort(node_scalars + (1.0 - in_queue) * 1e3)[0]
        assert in_queue[node] == 1
        in_tree[node] = 1
        in_queue[node] = 0

        for out in instance.out_nodes[node]:
            if in_tree[out] == 0 and (
                in_queue[out] == 0 or instance.adj[node][out] < node_scalars[out]
            ):
                pointers[out] = np.zeros(n, dtype=np.int32)
                pointers[out][node] = 1
                node_scalars[out] = instance.adj[node][out]
                in_queue[out] = 1

        push_states(
            node_states,
            edge_states,
            scalars,
            (in_queue, in_tree),
            (pointers, self_loops),
            (compute_current_scalars(node_scalars),),
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)


def dijkstra(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    in_queue = np.zeros(n, dtype=np.int32)
    in_tree = np.zeros(n, dtype=np.int32)

    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    node_scalars = instance.pos

    def compute_current_scalars(node_scalars):
        scalars = instance.adj[instance.edge_index[0], instance.edge_index[1]]
        scalars[instance.edge_index[0] == instance.edge_index[1]] = node_scalars
        return scalars

    in_queue[instance.start] = 1
    node_scalars[instance.start] = 0

    push_states(
        node_states,
        edge_states,
        scalars,
        (in_queue, in_tree),
        (pointers, self_loops),
        (compute_current_scalars(node_scalars),),
    )

    for _ in range(1, n):
        node = np.argsort(node_scalars + (1.0 - in_queue) * 1e3)[0]
        assert in_queue[node] == 1

        in_tree[node] = 1
        in_queue[node] = 0

        for out in instance.out_nodes[node]:
            if in_tree[out] == 0 and (
                in_queue[out] == 0
                or node_scalars[node] + instance.adj[node][out] < node_scalars[out]
            ):
                pointers[out] = np.zeros(n, dtype=np.int32)
                pointers[out][node] = 1
                node_scalars[out] = node_scalars[node] + instance.adj[node][out]
                in_queue[out] = 1

        push_states(
            node_states,
            edge_states,
            scalars,
            (in_queue, in_tree),
            (pointers, self_loops),
            (compute_current_scalars(node_scalars),),
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)


def mis(instance: ProblemInstance):
    n = instance.adj.shape[0]

    node_states = []
    edge_states = []
    scalars = []

    alive = np.ones(n, dtype=np.int32)
    in_mis = np.zeros(n, dtype=np.int32)

    self_loops = np.eye(n, dtype=np.int32)

    def compute_current_scalars():
        random_numbers = instance.randomness[len(node_states) // 2]
        return random_numbers[instance.edge_index[0]]

    push_states(
        node_states,
        edge_states,
        scalars,
        (in_mis, alive),
        (self_loops,),
        (compute_current_scalars(),),
    )
    while np.any(alive):
        random_numbers = instance.randomness[len(node_states) // 2]

        for node in filter(lambda x: alive[x], range(n)):
            if random_numbers[node] < random_numbers[
                np.logical_and(instance.adj[node], alive)
            ].min(initial=1.0):
                in_mis[node] = 1
            else:
                in_mis[node] = 0

        push_states(
            node_states,
            edge_states,
            scalars,
            (in_mis, alive),
            (self_loops,),
            (compute_current_scalars(),),
        )

        new_alive = np.copy(alive)
        for node in filter(lambda x: alive[x], range(n)):
            if in_mis[node] or np.any(in_mis[instance.adj[node].astype(bool)]):
                new_alive[node] = 0
            else:
                new_alive[node] = 1

        alive = new_alive
        push_states(
            node_states,
            edge_states,
            scalars,
            (in_mis, alive),
            (self_loops,),
            (compute_current_scalars(),),
        )

    while len(node_states) < n:
        push_states(
            node_states,
            edge_states,
            scalars,
            (in_mis, alive),
            (self_loops,),
            (compute_current_scalars(),),
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)


def er_probabilities(n):
    base = math.log(n) / n
    return (base, base * 3)


class ErdosRenyiGraphSampler:
    def __init__(self, config: base_config.Config):
        self.weighted = config.edge_weights
        self.generate_random_numbers = config.generate_random_numbers

    def __call__(self, num_nodes):
        p_segment = er_probabilities(num_nodes)
        p = p_segment[0] + np.random.rand() * (p_segment[1] - p_segment[0])

        random_numbers = None
        start = np.random.randint(0, num_nodes)

        while True:
            adj = np.triu(np.random.binomial(1, p, size=(num_nodes, num_nodes)), k=1)
            adj += adj.T

            if self.weighted:
                w = np.triu(np.random.uniform(0.0, 1.0, (num_nodes, num_nodes)), k=1)
                w *= adj
                adj = w + w.T
            if self.generate_random_numbers:
                random_numbers = np.random.rand(
                    num_nodes, num_nodes
                )  # steps count bounded by num_nodes
            instance = ProblemInstance(adj, start, self.weighted, random_numbers)

            is_connected = np.all(bfs(instance)[0][-1, :, 0] == 1)
            if not is_connected:
                continue

            return instance


MASK = 0
NODE_POINTER = 1
EDGE_MASK_ONE = 2
NODE_MASK_ONE = 3

SPEC = {}
SPEC["bfs"] = ((MASK,), (NODE_POINTER, NODE_POINTER))
SPEC["dfs"] = (
    (MASK, NODE_MASK_ONE, MASK, MASK),
    (NODE_POINTER, EDGE_MASK_ONE, NODE_POINTER),
)
SPEC["mst"] = ((MASK, MASK), (NODE_POINTER, NODE_POINTER))
SPEC["dijkstra"] = ((MASK, MASK), (NODE_POINTER, NODE_POINTER))
SPEC["mis"] = ((MASK, MASK, MASK, MASK), (NODE_POINTER,))  # MASK
SPEC["milp"] = ((MASK, NODE_POINTER, MASK, MASK, MASK, MASK, MASK, MASK), (MASK,))

ALGORITHMS = {"bfs": bfs, "dfs": dfs, "mst": mst, "dijkstra": dijkstra, "mis": mis, "milp": milp}


def _maybe_rebase_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index

    max_idx = int(edge_index.max().item())
    min_idx = int(edge_index.min().item())

    # Some generators may emit 1-based indices. Convert to 0-based when unambiguous.
    if min_idx == 1 and max_idx == num_nodes:
        return edge_index - 1
    return edge_index


def _normalize_edge_features(edge_fts, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge_fts = torch.as_tensor(edge_fts)

    # Normalize edge feature layout to [T, E, K].
    if edge_fts.ndim == 4:
        # Dense layouts: [T, N, N, K] or [T, K, N, N].
        if edge_fts.shape[1] == edge_fts.shape[2]:
            num_nodes = int(edge_fts.shape[1])
            edge_index = _maybe_rebase_edge_index(edge_index, num_nodes)
            edge_fts = edge_fts[:, edge_index[0], edge_index[1]]
        elif edge_fts.shape[2] == edge_fts.shape[3]:
            num_nodes = int(edge_fts.shape[2])
            edge_index = _maybe_rebase_edge_index(edge_index, num_nodes)
            edge_fts = edge_fts.permute(0, 2, 3, 1)[:, edge_index[0], edge_index[1]]
        else:
            raise ValueError(f"Unexpected dense edge_fts shape {tuple(edge_fts.shape)}")
    elif edge_fts.ndim == 3:
        # Dense layout without channel dim: [T, N, N].
        if edge_fts.shape[1] == edge_fts.shape[2]:
            num_nodes = int(edge_fts.shape[1])
            edge_index = _maybe_rebase_edge_index(edge_index, num_nodes)
            edge_fts = edge_fts[:, edge_index[0], edge_index[1]].unsqueeze(-1)
        else:
            # Sparse layouts: [T, E, K] or [T, K, E].
            num_edges = edge_index.shape[1]
            if edge_fts.shape[1] == num_edges:
                pass
            elif edge_fts.shape[2] == num_edges:
                edge_fts = edge_fts.permute(0, 2, 1)
            else:
                raise ValueError(
                    f"Unexpected sparse edge_fts shape {tuple(edge_fts.shape)} for num_edges={num_edges}"
                )
    else:
        raise ValueError(f"Unexpected edge_fts ndim={edge_fts.ndim}")

    return edge_fts, edge_index


def _normalize_scalars(scalars, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scalars = torch.as_tensor(scalars)
    num_edges = edge_index.shape[1]

    # Normalize scalar layout to [T, E, S].
    if scalars.ndim == 4 and scalars.shape[1] == scalars.shape[2]:
        # Dense per-node-pair scalars [T, N, N, S].
        num_nodes = int(scalars.shape[1])
        edge_index = _maybe_rebase_edge_index(edge_index, num_nodes)
        scalars = scalars[:, edge_index[0], edge_index[1]]
    elif scalars.ndim == 3 and scalars.shape[1] == scalars.shape[2]:
        # Dense per-node-pair scalars [T, N, N].
        num_nodes = int(scalars.shape[1])
        edge_index = _maybe_rebase_edge_index(edge_index, num_nodes)
        scalars = scalars[:, edge_index[0], edge_index[1]].unsqueeze(-1)
    elif scalars.ndim == 3 and scalars.shape[1] == num_edges:
        # Already [T, E, S].
        pass
    elif scalars.ndim == 2 and scalars.shape[1] == num_edges:
        # [T, E] -> [T, E, 1].
        scalars = scalars.unsqueeze(-1)
    else:
        raise ValueError(f"Unexpected scalars shape {tuple(scalars.shape)} for num_edges={num_edges}")

    return scalars, edge_index

def create_dataloader(config: base_config.Config, split: str, seed: int, device=None):
    np.random.seed(seed)

    datapoints = []
    sampler = ErdosRenyiGraphSampler(config)

    for _ in tqdm.tqdm(
        range(config.num_samples[split]), f"Generate samples for {split}"
    ):
        if config.algorithm == "milp":
            instance = random_feasible_milp(
                n_vars=config.problem_size[split],
                n_cons=config.milp_num_constraints,
                int_ratio=config.milp_int_ratio,
                seed=np.random.randint(0, 1_000_000),
            )

            node_fts, edge_fts, scalars = ALGORITHMS["milp"](
                instance,
                max_steps=config.milp_max_steps,
            )
            edge_index = torch.as_tensor(instance.edge_index, dtype=torch.long).contiguous()
        else:
            instance = sampler(config.problem_size[split])
            node_fts, edge_fts, scalars = ALGORITHMS[config.algorithm](instance)
            edge_index = torch.as_tensor(instance.edge_index, dtype=torch.long).contiguous()

        node_fts = torch.transpose(torch.as_tensor(node_fts), 0, 1)

        edge_fts, edge_index = _normalize_edge_features(edge_fts, edge_index)
        scalars, edge_index = _normalize_scalars(scalars, edge_index)

        edge_fts = torch.transpose(edge_fts, 0, 1)
        scalars = torch.transpose(scalars, 0, 1)

        output_fts = edge_fts if config.output_type == "pointer" else node_fts
        y = output_fts[:, -1, config.output_idx].clone().detach()

        datapoints.append(
            Data(
                node_fts=node_fts,
                edge_fts=edge_fts,
                scalars=scalars,
                edge_index=edge_index,
                y=y,
            )
        )
    return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True)


if __name__ == "__main__":
    from configs import base_config

    config = base_config.read_config("configs/mst.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = create_dataloader(config, "val", seed=1232, device=device)
    for batch in data:
        print(batch.node_fts[:, -1:, 0].sum() / 32)
        break