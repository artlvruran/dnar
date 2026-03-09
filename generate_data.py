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

class SATProblemInstance(ProblemInstance):
    def __init__(self, adj, clauses, num_vars, solution):
        super().__init__(adj=adj, start=0, weighted=True, randomness=np.zeros((1, 1)))
        self.clauses = clauses
        self.num_vars = num_vars
        self.num_clauses = len(clauses)
        self.solution = np.array(solution, dtype=np.int32)

def push_states(
    node_states, edge_states, scalars, cur_step_nodes, cur_step_edges, cur_step_scalars
):
    node_states.append(np.stack(cur_step_nodes, axis=-1))
    edge_states.append(np.stack(cur_step_edges, axis=-1))
    scalars.append(np.stack(cur_step_scalars, axis=-1))

def _unit_propagate(clauses, assignments, trace, snapshots):
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            clause_satisfied = False
            unassigned = []
            for var, is_positive in clause:
                if assignments[var] == -1:
                    unassigned.append((var, is_positive))
                    continue

                literal_value = assignments[var] == (1 if is_positive else 0)
                if literal_value:
                    clause_satisfied = True
                    break

            if clause_satisfied:
                continue
            if len(unassigned) == 0:
                return False
            if len(unassigned) == 1:
                unit_var, unit_positive = unassigned[0]
                unit_value = 1 if unit_positive else 0
                if assignments[unit_var] == -1:
                    assignments[unit_var] = unit_value
                    trace.append((unit_var, unit_value, 1))
                    snapshots.append(np.copy(assignments))
                    changed = True
                elif assignments[unit_var] != unit_value:
                    return False
    return True

def _dpll(clauses, assignments, trace, snapshots):
    if not _unit_propagate(clauses, assignments, trace, snapshots):
        return None

    if np.all(assignments != -1):
        return np.copy(assignments), trace, snapshots

    branch_var = np.where(assignments == -1)[0][0]

    for value in (1, 0):
        next_assignments = np.copy(assignments)
        next_trace = trace.copy()
        next_snapshots = snapshots.copy()

        next_assignments[branch_var] = value
        next_trace.append((branch_var, value, 0))
        next_snapshots.append(np.copy(next_assignments))

        result = _dpll(clauses, next_assignments, next_trace, next_snapshots)
        if result is not None:
            return result

    return None

def _compute_clause_satisfied_from_assignments(instance: SATProblemInstance, assignments):
    clause_satisfied = np.zeros(instance.adj.shape[0], dtype=np.int32)
    for clause_idx, clause in enumerate(instance.clauses):
        clause_node = instance.num_vars + clause_idx
        satisfied = 0
        for clause_var, is_positive in clause:
            current_value = assignments[clause_var]
            if current_value == -1:
                continue
            if current_value == (1 if is_positive else 0):
                satisfied = 1
                break
        clause_satisfied[clause_node] = satisfied
    return clause_satisfied

def sat(instance: SATProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    is_variable = np.zeros(n, dtype=np.int32)
    is_variable[: instance.num_vars] = 1

    assigned_flag = np.zeros(n, dtype=np.int32)
    clause_satisfied = np.zeros(n, dtype=np.int32)

    self_loops = np.eye(n, dtype=np.int32)

    def compute_sender_scalars(assignments_vars):
        rng = np.random.RandomState(0)
        sender = np.zeros(n, dtype=np.float32)
        for v in range(instance.num_vars):
            if assignments_vars[v] == -1:
                sender[v] = float(rng.rand())
            else:
                sender[v] = float(assignments_vars[v])
        for ci, clause in enumerate(instance.clauses):
            clause_node = instance.num_vars + ci
            satisfied = 0
            num_unassigned = 0
            for var, is_positive in clause:
                val = assignments_vars[var]
                if val == -1:
                    num_unassigned += 1
                    continue
                if val == (1 if is_positive else 0):
                    satisfied = 1
                    break
            sender[clause_node] = 0.0 if satisfied else float(num_unassigned)
        return sender

    init_assignments = np.full(instance.num_vars, -1, dtype=np.int32)
    cur_sender = compute_sender_scalars(init_assignments)
    cur_scalars = cur_sender[instance.edge_index[0]]

    push_states(
        node_states,
        edge_states,
        scalars,
        (is_variable, assigned_flag, clause_satisfied),
        (self_loops,),
        (cur_scalars,),
    )

    initial_assignments = np.full(instance.num_vars, -1, dtype=np.int32)
    solved = _dpll(instance.clauses, initial_assignments, trace=[], snapshots=[])
    assert solved is not None
    solution, trace, snapshots = solved

    for assignment_snapshot in snapshots:
        assigned_mask = (assignment_snapshot != -1).astype(np.int32)
        assigned_flag = np.zeros(n, dtype=np.int32)
        assigned_flag[: instance.num_vars] = assigned_mask

        clause_satisfied = _compute_clause_satisfied_from_assignments(instance, assignment_snapshot)

        cur_sender = compute_sender_scalars(assignment_snapshot)
        cur_scalars = cur_sender[instance.edge_index[0]]

        push_states(
            node_states,
            edge_states,
            scalars,
            (is_variable, assigned_flag, clause_satisfied),
            (self_loops,),
            (cur_scalars,),
        )

    min_steps = max(n, len(trace) + 1)
    final_sender = compute_sender_scalars(solution)
    final_scalars = final_sender[instance.edge_index[0]]
    final_assigned_mask = (solution != -1).astype(np.int32)
    final_assigned_flag = np.zeros(n, dtype=np.int32)
    final_assigned_flag[: instance.num_vars] = final_assigned_mask
    final_clause_satisfied = _compute_clause_satisfied_from_assignments(instance, solution)

    while len(node_states) < min_steps:
        push_states(
            node_states,
            edge_states,
            scalars,
            (is_variable, final_assigned_flag, final_clause_satisfied),
            (self_loops,),
            (final_scalars,),
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)


def sat_instance_from_cnf(cnf_clauses, num_vars):
    clauses = []
    for clause in cnf_clauses:
        parsed_clause = []
        for literal in clause:
            assert literal != 0
            var = abs(int(literal)) - 1
            assert 0 <= var < num_vars
            is_positive = literal > 0
            parsed_clause.append((var, is_positive))
        clauses.append(parsed_clause)

    assignments = np.full(num_vars, -1, dtype=np.int32)
    solved = _dpll(clauses, assignments, trace=[], snapshots=[])
    if solved is None:
        raise ValueError('Provided CNF formula is unsatisfiable.')
    solved_assignment, _, _ = solved

    num_nodes = num_vars + len(clauses)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for clause_idx, clause in enumerate(clauses):
        clause_node = num_vars + clause_idx
        for var, is_positive in clause:
            polarity = 1.0 if is_positive else -1.0
            adj[var, clause_node] = polarity
            adj[clause_node, var] = polarity
            
    np.fill_diagonal(adj, 1.0)

    return SATProblemInstance(
        adj=adj, clauses=clauses, num_vars=num_vars, solution=solved_assignment
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


class SATGraphSampler:
    def __init__(self, config: base_config.Config):
        self.clause_ratio = config.sat_clause_ratio
        self.clause_width = config.sat_clause_width

    def __call__(self, num_vars):
        assert self.clause_width >= 2
        num_clauses = max(1, int(self.clause_ratio * num_vars))

        while True:
            planted_solution = np.random.binomial(1, 0.5, size=(num_vars,))
            clauses = []
            for _ in range(num_clauses):
                vars_in_clause = np.random.choice(
                    num_vars, size=min(self.clause_width, num_vars), replace=False
                )

                while True:
                    signs = np.random.binomial(1, 0.5, size=(len(vars_in_clause),))
                    clause_is_satisfied = np.any(
                        planted_solution[vars_in_clause] == signs
                    )
                    if clause_is_satisfied:
                        break

                clause = [
                    (int(var), bool(sign))
                    for var, sign in zip(vars_in_clause, signs, strict=True)
                ]
                clauses.append(clause)

            assignments = np.full(num_vars, -1, dtype=np.int32)
            solved = _dpll(clauses, assignments, trace=[], snapshots=[])

            if solved is None:
                continue

            solved_assignment, _, _ = solved

            num_nodes = num_vars + num_clauses
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            for clause_idx, clause in enumerate(clauses):
                clause_node = num_vars + clause_idx
                for var, is_positive in clause:
                    polarity = 1.0 if is_positive else -1.0
                    adj[var, clause_node] = polarity
                    adj[clause_node, var] = polarity

            return SATProblemInstance(
                adj=adj, clauses=clauses, num_vars=num_vars, solution=solved_assignment
            )


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
SPEC["sat"] = ((MASK, MASK, MASK), (MASK,))

ALGORITHMS = {"bfs": bfs, "dfs": dfs, "mst": mst, "dijkstra": dijkstra, "mis": mis, 'sat': sat}


def create_dataloader(config: base_config.Config, split: str, seed: int, device):
    np.random.seed(seed)

    datapoints = []
    if config.algorithm == "sat":
        sampler = SATGraphSampler(config)
    else:
        sampler = ErdosRenyiGraphSampler(config)

    for _ in tqdm.tqdm(
        range(config.num_samples[split]), f"Generate samples for {split}"
    ):
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

        datapoints.append(
            Data(
                node_fts=node_fts,
                edge_fts=edge_fts,
                scalars=scalars,
                edge_index=edge_index,
                y=y,
            ).to(device)
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
