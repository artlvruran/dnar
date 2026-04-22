import argparse
import torch

from configs import base_config
from generate_data import create_dataloader
from models import Dnar


def load_model(config, checkpoint_path, device):
    model = Dnar(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def infer_one(model, graph):
    pred, _ = model(graph)

    # Подстройте маску, если у вас другой layout признаков.
    # Здесь предполагается, что node_fts[:, 0] == 1 для допустимых variable nodes.
    variable_mask = graph.node_fts[:, 0] > 0.5

    scores = pred.squeeze(-1) if pred.ndim > 1 else pred
    scores = scores[variable_mask]

    var_nodes = torch.where(variable_mask)[0]
    chosen = var_nodes[torch.argmax(scores)].item()
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = base_config.read_config(args.config_path)
    model = load_model(config, args.checkpoint, device)
    loader = create_dataloader(config, args.split, seed=args.seed, device=device)

    batch = next(iter(loader))
    graph = batch.to_data_list()[0]
    chosen_var = infer_one(model, graph)

    print({"chosen_variable": chosen_var})