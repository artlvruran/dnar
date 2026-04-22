import argparse
import torch

import utils
from configs import base_config
from generate_data import create_dataloader
from models import Dnar


def load_model(config, checkpoint_path, device):
    model = Dnar(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_printoptions(precision=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = base_config.read_config(args.config_path)

    model = load_model(config, args.checkpoint, device)
    sampler = create_dataloader(config, args.split, seed=args.seed, device=device)

    with torch.no_grad():
        scores = utils.evaluate(model, sampler, utils.METRICS[config.output_type])

    print(scores)