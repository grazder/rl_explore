import os
import argparse
import ray
import wandb
from hyperpyyaml import load_hyperpyyaml
from ray import tune
from log_utils import log_metrics, save_gifs


def train(config):
    # Directory check
    for dir in [config['checkpoint_save_dir'], config['gifs_save_dir']]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Tuning enviroment
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env(config['env_name'], lambda cfg: config['env_class'](**cfg))

    agent = config['agent'](config['agent_config'])

    for iter_number in range(config['n_iters']):
        result = agent.train()
        _ = agent.save(config['checkpoint_save_dir'])

        # logging
        log_metrics(iter_number, result)

        if (iter_number + 1) % 5 == 0:
            save_gifs(agent, config, iter_number)


if __name__ == "__main__":
    # Reading config
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, required=True,
                        help='path to HyperPyYaml config')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = load_hyperpyyaml(f)

    wandb.init(project="prod-rl-hw", config=cfg)
    train(cfg)
