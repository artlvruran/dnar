import math

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.utils import scatter

from configs import base_config
from generate_data import EDGE_MASK_ONE, MASK, NODE_MASK_ONE, NODE_POINTER, SPEC
from utils import from_binary_states, gumbel_softmax, node_pointer_loss, temp_by_step
from mamba_ssm import Mamba


def _edge_scalar_column(scalars):
    if not torch.is_tensor(scalars):
        scalars = torch.tensor(scalars)
    if scalars.dim() == 1:
        return scalars.unsqueeze(-1)
    if scalars.dim() == 2 and scalars.size(-1) == 1:
        return scalars
    return scalars.reshape(scalars.shape[0], -1)[:, :1]


def _sequence_order_key(group_ids, secondary):
    group_ids = group_ids.long()
    secondary = secondary.long()
    scale = int(secondary.max().item()) + 1 if secondary.numel() else 1
    return group_ids * scale + secondary


def _apply_sequence_block(x, group_ids, secondary, block):
    if x.numel() == 0:
        return x

    order = torch.argsort(_sequence_order_key(group_ids, secondary), stable=True)
    x_sorted = x[order]
    groups_sorted = group_ids[order]
    counts = torch.bincount(groups_sorted, minlength=int(groups_sorted.max().item()) + 1)

    lengths = counts[counts > 0].tolist()
    seqs = []
    start = 0
    for length in lengths:
        seqs.append(x_sorted[start : start + length])
        start += length

    if len(seqs) == 1:
        y_sorted = block(seqs[0])
    else:
        padded = pad_sequence(seqs, batch_first=True)
        y_padded = block(padded)
        y_sorted = torch.cat(
            [y_padded[i, :length] for i, length in enumerate(lengths)], dim=0
        )

    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=x.device)
    return y_sorted[inv]


class StatesEncoder(torch.nn.Module):
    def __init__(self, h, num_binary_states):
        super().__init__()
        self.emb = torch.nn.Embedding(2**num_binary_states, h)

    def forward(self, states):
        return self.emb(from_binary_states(states))


class SelectBest(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.emb = torch.nn.Embedding(2 ** (config.num_node_states + 1), config.h)

    def forward(self, binary_states, scalars, index):
        states = 2 * from_binary_states(binary_states)
        group_with_reciever = torch.cat(
            [torch.unsqueeze(states, -1), torch.unsqueeze(index, -1)], dim=1
        )
        _, group_index = torch.unique(
            group_with_reciever, sorted=False, return_inverse=True, dim=0
        )

        best_in_group = gumbel_softmax(
            -scalars.squeeze(), group_index, tau=0.0, use_noise=False
        )

        state_with_best = states + best_in_group
        return self.emb(state_with_best.long())


class _TinyMambaBlock(torch.nn.Module):
    def __init__(self, h, d_state=4, d_conv=1, expand=1):
        super().__init__()
        self.norm = torch.nn.LayerNorm(h)
        self.mamba = Mamba(
            d_model=h,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ).float()

    def forward(self, x):
        if x.numel() == 0:
            return x

        orig_dtype = x.dtype
        x = x.float()

        z = F.layer_norm(
            x,
            (x.size(-1),),
            self.norm.weight.float(),
            self.norm.bias.float(),
            self.norm.eps,
        )

        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze = True

        y = self.mamba(z.to(torch.float32))

        if squeeze:
            y = y.squeeze(0)

        return (x + y).to(orig_dtype)


class MambaMessagePassing(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h
        self.h = h

        self.edge_states_encoder = StatesEncoder(h, config.num_edge_states)
        self.node_states_encoder = StatesEncoder(h, config.num_node_states)
        self.static_fts_encoder = StatesEncoder(h, 2)

        self.select_best_virtual = SelectBest(config)

        self.node_scalar_proj = Linear(1, h, bias=False)
        self.edge_in = Linear(5 * h, h, bias=False)

        self.node_mamba = _TinyMambaBlock(
            h,
            d_state=getattr(config, "mamba_d_state", 4),
            d_conv=getattr(config, "mamba_d_conv", 1),
            expand=getattr(config, "mamba_expand", 1),
        )

    def _node_order(self, batch, num_nodes, device):
        deg = torch.bincount(batch.edge_index[0], minlength=num_nodes) + torch.bincount(
            batch.edge_index[1], minlength=num_nodes
        )
        return deg.to(device) * (num_nodes + 1) + torch.arange(num_nodes, device=device)

    def select_best_from_virtual(self, node_states, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        return self.select_best_virtual(node_states, node_scalars, batch.batch)

    def compute_static_fts(self, scalars, batch):
        scalars = _edge_scalar_column(scalars).float()
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        sender_s = node_scalars[batch.edge_index[0]]
        reciever_s = node_scalars[batch.edge_index[1]]

        rlx = scalars < reciever_s
        rlx_d = sender_s + scalars < reciever_s

        fts = torch.cat([rlx, rlx_d], dim=-1).long()
        return self.static_fts_encoder(fts)

    def forward(self, node_states, edge_states, scalars, batch, training_step):
        scalars = _edge_scalar_column(scalars).float()

        node_tokens = self.select_best_from_virtual(node_states, scalars, batch)
        node_order = self._node_order(batch, node_tokens.size(0), node_tokens.device)
        node_ctx = _apply_sequence_block(node_tokens, batch.batch, node_order, self.node_mamba)

        edge_emb = self.edge_states_encoder(edge_states)
        static_fts = self.compute_static_fts(scalars, batch)

        edge_tokens = self.edge_in(
            torch.cat(
                [
                    node_ctx[batch.edge_index[0]],
                    node_ctx[batch.edge_index[1]],
                    edge_emb,
                    edge_emb[batch.batched_reverse_idx],
                    static_fts,
                ],
                dim=-1,
            )
        )

        node_ctx = node_ctx + scatter(
            edge_tokens,
            index=batch.edge_index[1],
            dim=0,
            dim_size=node_ctx.size(0),
            reduce="sum",
        )
        return node_ctx, edge_tokens


class ScalarUpdater(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h

        self.node_states_encoder = StatesEncoder(config.h, config.num_node_states)
        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)

        self.combine_fts = Linear(2 * h, h)

        self.keep_proj = Linear(h, 2)
        self.push_proj = Linear(h, 2)
        self.push_node_proj = Linear(h, 2)
        self.increment_proj = Linear(h, 2)

        self.scalars_only_as_input = config.generate_random_numbers
        self.temp = (
            config.processor_upper_t,
            config.processor_lower_t,
            config.num_iterations,
            config.temp_on_eval,
        )
        self.use_noise = config.use_noise

    def forward(
        self,
        node_states,
        edge_states,
        scalars,
        batch,
        training_step,
        processor_step,
        teacher_force,
    ):
        if self.scalars_only_as_input:
            return batch.scalars[:, processor_step], 0.0

        dtype = self.combine_fts.weight.dtype
        node_fts = self.node_states_encoder(node_states).to(dtype)
        edge_fts = self.edge_states_encoder(edge_states).to(dtype)

        fts = self.combine_fts(
            torch.cat(
                [edge_fts[batch.batched_reverse_idx], node_fts[batch.edge_index[0]]],
                dim=1,
            )
        )
        index = torch.repeat_interleave(torch.arange(fts.shape[0]).to(fts.device), 2)

        increment = self.compute_increment(fts, index, training_step)
        push = self.compute_push(fts, scalars.view(-1), batch, index, training_step)
        keep = self.compute_keep(fts, scalars.view(-1), index, training_step)

        new_scalars = torch.unsqueeze(increment + keep + push, -1)

        loss = (
            ((batch.scalars[:, processor_step] - new_scalars) ** 2).mean()
            if training_step != -1
            else 0.0
        )

        if teacher_force:
            new_scalars = batch.scalars[:, processor_step]

        return new_scalars, loss

    def compute_increment(self, fts, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        logits = self.increment_proj(fts).view(-1)
        increment = gumbel_softmax(logits, index=index, tau=tau, use_noise=use_noise)[
            ::2
        ]
        return 1.0 * increment

    def compute_push(self, fts, scalars, batch, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        push_without_node_logits = self.push_proj(fts).view(-1)
        push_without_node = gumbel_softmax(
            push_without_node_logits, index=index, tau=tau, use_noise=use_noise
        )[::2]

        push_with_node_logits = self.push_node_proj(fts).view(-1)
        push_with_node = gumbel_softmax(
            push_with_node_logits, index=index, tau=tau, use_noise=use_noise
        )[::2]

        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        scalars_without_node = scalars - node_scalars[batch.edge_index[1]]
        scalars_with_node = scalars_without_node + node_scalars[batch.edge_index[0]]

        edge_push_without_node = scatter(
            push_without_node * scalars_without_node, batch.edge_index[1], reduce="sum"
        )
        edge_push_with_node = scatter(
            push_with_node * scalars_with_node, batch.edge_index[1], reduce="sum"
        )

        accumulated_node = edge_push_without_node + edge_push_with_node
        edge_push = torch.zeros_like(scalars)
        edge_push[batch.edge_index[0] == batch.edge_index[1]] = accumulated_node
        return edge_push

    def compute_keep(self, fts, scalars, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        logits = self.keep_proj(fts).view(-1)
        keep = gumbel_softmax(logits, index=index, tau=tau, use_noise=use_noise)[::2]
        return scalars * keep


class StatesBottleneck(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.node_projections = ModuleList(
            [Linear(h, 1) for _ in range(config.num_node_states)]
        )
        self.edge_projections = ModuleList(
            [Linear(h, 1) for _ in range(config.num_edge_states)]
        )
        self.spec = SPEC[config.algorithm]

    def forward(
        self, node_fts, edge_fts, batch, training_step, processor_step, teacher_force
    ):
        dtype = self.node_projections[0].weight.dtype
        node_fts = node_fts.to(dtype)
        edge_fts = edge_fts.to(dtype)

        states = []

        loss = 0.0

        for group in range(2):
            fts = node_fts if group == 0 else edge_fts
            stacked_fts = []

            projections = self.node_projections if group == 0 else self.edge_projections
            hints = (
                batch.node_fts[:, processor_step]
                if group == 0
                else batch.edge_fts[:, processor_step]
            )

            for idx, projection in enumerate(projections):
                logits = projection(fts).squeeze()
                gt = hints[:, idx].to(dtype)

                if training_step != -1:
                    if self.spec[group][idx] != MASK:
                        index = batch.batch if group == 0 else batch.edge_index[0]
                        weight = 1
                        if self.spec[group][idx] == EDGE_MASK_ONE:
                            index = batch.batch[batch.edge_index[0]]
                            num_nodes = (batch.batch == 0).sum()
                            weight = num_nodes
                        ce_loss = weight * node_pointer_loss(logits, gt, index)
                    else:
                        ce_loss = binary_cross_entropy_with_logits(logits, gt)

                    loss += ce_loss

                if not teacher_force:
                    if self.spec[group][idx] != MASK:
                        index = batch.batch if group == 0 else batch.edge_index[0]
                        if self.spec[group][idx] == EDGE_MASK_ONE:
                            index = batch.batch[batch.edge_index[0]]
                        pred = gumbel_softmax(
                            logits, index=index, tau=0.0, use_noise=False
                        )
                    else:
                        pred = (logits > 0.0).to(dtype)
                else:
                    pred = gt
                stacked_fts.append(torch.unsqueeze(pred, -1))
            states.append(torch.cat(stacked_fts, -1))

        return *states, loss


class DiscreteProcessor(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h
        self.message_passing = MambaMessagePassing(config)

        self.node_ffn = Sequential(Linear(h, h), ReLU(), Linear(h, h), ReLU())
        self.edge_ffn = Sequential(Linear(2 * h, h), ReLU(), Linear(h, h), ReLU())

        self.states_bottleneck = StatesBottleneck(config)
        self.scalar_update = ScalarUpdater(config)

    def forward(
        self,
        node_states,
        edge_states,
        scalars,
        batch,
        training_step,
        processor_step,
        teacher_force,
    ):
        node_fts, edge_fts = self.message_passing(
            node_states, edge_states, scalars, batch, training_step
        )
        node_fts, edge_fts = self.ffn(node_fts, edge_fts, batch)

        node_states, edge_states, states_loss = self.states_bottleneck(
            node_fts, edge_fts, batch, training_step, processor_step, teacher_force
        )
        out_scalars, scalars_loss = self.scalar_update(
            node_states,
            edge_states,
            scalars,
            batch,
            training_step,
            processor_step,
            teacher_force,
        )

        loss = scalars_loss + states_loss

        return node_states, edge_states, out_scalars, loss

    def ffn(self, node_fts, edge_fts, batch):
        dtype = self.node_ffn[0].weight.dtype
        node_fts = node_fts.to(dtype)
        edge_fts = edge_fts.to(dtype)

        node_fts = node_fts + self.node_ffn(node_fts)
        edge_fts_with_reversed = torch.cat(
            [edge_fts, edge_fts[batch.batched_reverse_idx]], dim=1
        )

        edge_fts = edge_fts + self.edge_ffn(edge_fts_with_reversed)
        return node_fts, edge_fts