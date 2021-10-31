import os
import argparse
import ray
from hyperpyyaml import load_hyperpyyaml
from ray import tune
from tqdm import tqdm
from log_utils import save_gifs


def evaluate(config, checkpoint_path, n_runs, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    assert os.path.isfile(checkpoint_path), 'checkpoint not found'

    # Tuning enviroment
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env(config['env_name'], lambda cfg: config['env_class'](**cfg))

    agent = config['agent'](config['agent_config'])
    agent.restore(checkpoint_path)

    for i in tqdm(range(n_runs)):
        gif_path = os.path.join(output_folder, f"run-{i + 1}")
        save_gifs(agent, config, gif_path=gif_path, log_wandb=False)


if __name__ == "__main__":
    # Reading config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help='path to HyperPyYaml config')
    parser.add_argument("--checkpoint", type=str, required=True,
                        help='path to agent checkpoint')
    parser.add_argument("--n-runs", type=int, required=True,
                        help='number of model runs')
    parser.add_argument("--output", type=str, required=True,
                        help='path to output folder')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = load_hyperpyyaml(f)

    evaluate(cfg, args.checkpoint, args.n_runs, args.output)

