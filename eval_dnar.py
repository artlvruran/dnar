import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch

import utils
from configs import base_config
from generate_data import create_dataloader
from models import Dnar


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "dijkstra.yaml"
MODEL_PATH = ROOT / "dijkstra_47_last_dnar_attn"   # файл модели лежит рядом со скриптом
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def main():
    config = base_config.read_config(str(CONFIG_PATH))

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
    torch.autograd.set_detect_anomaly(False)

    split = "val"
    config.num_samples[split] = 1  # если нужен один сэмпл на валидацию

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Dnar(config)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    results = []

    for num_nodes in range(1000, 10001, 1000):
        print(f"\nEvaluating for num_nodes_max: {num_nodes}")
        config.problem_size[split] = num_nodes
        config.num_samples['val'] = 1

        dataloader = create_dataloader(config, split, seed=40, device=device)

        with torch.no_grad():
            start_time = time.time()
            scores = utils.evaluate(model, dataloader, utils.METRICS[config.output_type])
            end_time = time.time()

        row = {
            "num_nodes": num_nodes,
            "validation_time": end_time - start_time,
            "scores": to_jsonable(scores),
        }
        results.append(row)

        print(f"Validation time: {row['validation_time']:.4f} seconds")
        print(f"Scores: {scores}")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(CONFIG_PATH),
        "model_path": str(MODEL_PATH),
        "split": split,
        "device": str(device),
        "results": results,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)

    print("\n--- Final Results ---")
    for r in results:
        print(f"Num Nodes: {r['num_nodes']}, Time: {r['validation_time']:.4f}, Scores: {r['scores']}")
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()