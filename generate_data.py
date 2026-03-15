import math

import networkx as nx
import numpy as np
import torch
import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from configs import base_config


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
        
def sat(instance_or_clauses):
    clauses = np.asarray(instance_or_clauses, dtype=np.int64)
    randomness = np.random.rand(3 * clauses.shape[0], 3 * clauses.shape[0])
    start = 0
    if clauses.ndim != 2 or clauses.shape[1] != 3:
        raise ValueError("clauses must have shape [num_clauses, 3]")
    if clauses.size == 0:
        raise ValueError("clauses must contain at least one clause")
    if np.any(clauses == 0):
        raise ValueError("literals must be non-zero integers")

    num_clauses = clauses.shape[0]
    num_nodes = 3 * num_clauses
    sat_adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    flattened_literals = clauses.reshape(-1)

    for clause_idx in range(num_clauses):
        base = 3 * clause_idx
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = base + i, base + j
                sat_adj[u][v] = 1
                sat_adj[v][u] = 1

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if flattened_literals[i] == -flattened_literals[j]:
                sat_adj[i][j] = 1
                sat_adj[j][i] = 1

    reduced_instance = ProblemInstance(sat_adj, start, weighted=False, randomness=randomness)
    outputs = mis(reduced_instance)
    return *outputs, reduced_instance



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

class SatClauseSampler:
    def __init__(self, config: base_config.Config):
        self.generate_random_numbers = config.generate_random_numbers

    def __call__(self, num_nodes):
        num_clauses = max(1, num_nodes // 3)
        num_vars = max(1, num_nodes)

        clauses = np.random.randint(1, num_vars + 1, size=(num_clauses, 3))
        signs = np.where(np.random.rand(num_clauses, 3) < 0.5, 1, -1)
        return clauses * signs

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
SPEC["sat"] = ((MASK, MASK, MASK, MASK), (NODE_POINTER,))

ALGORITHMS = {"bfs": bfs, "dfs": dfs, "mst": mst, "dijkstra": dijkstra, "mis": mis, 'sat': sat}


def create_dataloader(config: base_config.Config, split: str, seed: int, device):
    np.random.seed(seed)

    datapoints = []
    if config.algorithm == "sat":
        sampler = SatClauseSampler(config)
    else:
        sampler = ErdosRenyiGraphSampler(config)

    for _ in tqdm.tqdm(
        range(config.num_samples[split]), f"Generate samples for {split}"
    ):
        if config.algorithm == "sat":
            clauses = sampler(config.problem_size[split])
            node_fts, edge_fts, scalars, instance = sat(clauses)
        else:
            instance = sampler(config.problem_size[split])
            node_fts, edge_fts, scalars = ALGORITHMS[config.algorithm](instance)

        edge_index = torch.tensor(instance.edge_index).contiguous()

        node_fts = torch.transpose(torch.tensor(node_fts), 0, 1)
        edge_fts = torch.transpose(
            torch.tensor(edge_fts)[:, edge_index[0], edge_index[1]], 0, 1
        )
        scalars = torch.transpose(torch.tensor(scalars), 0, 1)

        output_fts = edge_fts if config.output_type == "pointer" else node_fts
        y = output_fts[:, -1, config.output_idx].clone().detach()

        data_kwargs = {
            "node_fts": node_fts,
            "edge_fts": edge_fts,
            "scalars": scalars,
            "edge_index": edge_index,
            "y": y,
        }

        datapoints.append(Data(**data_kwargs).to(device))

    
    return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True)


if __name__ == "__main__":
    from configs import base_config

    config = base_config.read_config("configs/mst.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = create_dataloader(config, "val", seed=1232, device=device)
    for batch in data:
        print(batch.node_fts[:, -1:, 0].sum() / 32)
        break
