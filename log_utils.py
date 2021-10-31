import os
import wandb
from PIL import Image


def save_gifs(agent, config, iter_number: int) -> None:
    env = config['env_class'](**config['env_config'])
    obs = env.reset()
    Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save('tmp.png')

    frames = []

    for _ in range(config['eval_steps']):
        action = agent.compute_single_action(obs)

        frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
        frames.append(frame)

        obs, reward, done, info = env.step(action)
        if done:
            break

    gif_path = os.path.join(config['gifs_save_dir'], f"out{iter_number + 1}.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)
    wandb.log({"video": wandb.Video(gif_path, fps=30, format="gif")})


def log_metrics(iter_number, result):
    metrics_dict = {
        "iter": iter_number,
        "reward_min": result["episode_reward_min"],
        "reward_mean": result["episode_reward_mean"],
        "reward_max": result["episode_reward_max"],
        "entropy": result["info"]["learner"]["default_policy"]["learner_stats"]["entropy"],
        "episode_len_mean": result["episode_len_mean"]
    }

    wandb.log(metrics_dict)
