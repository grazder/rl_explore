import argparse
import os
import ray
from hyperpyyaml import load_hyperpyyaml
from ray import tune
from PIL import Image


def save_log(agent, config, iter_number):
    env = config['env_class'](**config['env_config'])
    obs = env.reset()
    Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save('tmp.png')

    frames = []

    for _ in range(config.eval_steps):
        action = agent.compute_single_action(obs)

        frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
        frames.append(frame)

        obs, reward, done, info = env.step(action)
        if done:
            break

    gif_path = os.path.join(config['gifs_save_dir'], f"out{iter_number + 1}.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)


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
        file_name = agent.save(config['checkpoint_save_dir'])

        # logging
        if (iter_number + 1) % 5 == 0:
            save_log(agent, config, iter_number)


if __name__ == "__main__":
    # Reading config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help='path to hyperpyyaml config')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = load_hyperpyyaml(f)

    train(cfg)
