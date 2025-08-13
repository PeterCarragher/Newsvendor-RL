import ray.rllib.agents.ppo as ppo
# Add callbacks for custom metrics
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from typing import TYPE_CHECKING, Dict, Optional
from ray.rllib.evaluation.episode import Episode
from or_gym.envs.classic_or.newsvendor import NewsvendorEnv, NewsvendorDiscreteEnv
from scipy.stats import kurtosis
import numpy as np

class InventoryCallbacks(DefaultCallbacks):

    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: any,
                         episode: Episode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:

        env = base_env.get_sub_environments()[0]
        episode.custom_metrics["starting_inventory_episode"] = env.inv_on_hand

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: Episode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:
        env = base_env.get_sub_environments()[0]
        episode.custom_metrics.setdefault("running_inventory", []).append(env.inv_on_hand)
        episode.custom_metrics.setdefault("running_order", []).append(env.last_order)
        episode.custom_metrics.setdefault("running_remaining_inventory", []).append(env.max_inventory - env.inv_on_hand)
        episode.custom_metrics.setdefault("running_leading_orders", []).append(env.currently_ordered)
        episode.custom_metrics.setdefault("running_demand", []).append(env.last_demand)
        episode.custom_metrics.setdefault("running_sold", []).append(env.last_sold)
        episode.custom_metrics.setdefault("running_short", []).append(env.last_short)
        episode.custom_metrics.setdefault("running_excess", []).append(env.last_excess)

        # print('Stats: ', env.inv_on_hand, env.last_demand, env.last_order, env.last_sold, env.currently_ordered)

        # print('remaining: ', env.max_inventory - order - inventory)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: any,
                       episode: Episode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:

        env = base_env.get_sub_environments()[0]
        episode.custom_metrics["final_inventory_episode"] = env.inv_on_hand
        episode.custom_metrics["final_remaining_inventory_episode"] = env.max_inventory - env.inv_on_hand

        items = [(x, y) for x, y in episode.custom_metrics.items() if 'running' in x]

        for key, values in items:
            idx = int(len(values)*0.95) - 1
            episode.custom_metrics[f"{key}_timestep_mean_episode"] = np.mean(values)
            # episode.custom_metrics[f"{key}_timestep_kurt_episode"] = kurtosis(values)
            episode.custom_metrics[f"{key}_final_timestep_mean_episode"] = np.mean(values[idx:])
            # episode.custom_metrics[f"{key}_final_timestep_kurt_episode"] = kurtosis(values[idx:])

debug = False
sample_cpu = 6
sample_gpu = 0

rl_config = ppo.DEFAULT_CONFIG.copy()
config = {
        # 'env': env_name,
        # 'env_config':env_config,
        'framework': 'torch',

        # 'gamma': 0.9936809332376452,
        # 'lambda': 0.9517171675473532,
        # 'kl_target': 0.010117093480119358,
        # 'clip_param': 0.20425701146213993,
        # 'vf_loss_coeff': 0.3503035138680095,
        'clip_rewards': True,
        'vf_clip_param': 1000000000,
        # 'entropy_coeff': 0.0004158966184268587,
        # 'train_batch_size': 1024,
        # 'rollout_fragment_length': 128,
        'num_workers': sample_cpu - 1,
        'num_envs_per_worker': 1,
        'num_gpus': sample_gpu,
        'log_level': 'INFO' if not debug else 'DEBUG',
        "ignore_worker_failures": True,
        "gamma": 1,
        "kl_coeff": 1.0,
        "num_sgd_iter": 5,
        "lr": 0.001,
        "sgd_minibatch_size": 3276,
        "train_batch_size": 32000,
        "model": {
            "fcnet_hiddens": [64, 32],
        },
        "use_gae": False,
        # "num_workers": (self.num_cpus - 1),
        # "num_gpus": self.num_gpus,
        "batch_mode": "complete_episodes",
        'observation_filter': 'MeanStdFilter',
        # 'horizon': 500,
        # 'model': {
        #     'vf_share_layers': "false",
        #     # 'use_lstm': "true",
        #     # 'max_seq_len': 13,
        #     'fcnet_hiddens': [256, 256],
        #     'fcnet_activation': "elu",
            # 'lstm_cell_size': 256,
            # 'lstm_use_prev_action': "true",
            # 'lstm_use_prev_reward': "true",
        # },
        # '_disable_execution_plan_api': True,
        # 'multiagent': default_multiagent,
        'callbacks': InventoryCallbacks
    }
rl_config.update(config)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
 
# Unpack values from each iteration

def plot_results(results):
    rewards = np.hstack([i['hist_stats']['episode_reward'] 
        for i in results])
    pol_loss = [
        i['info']['learner']['default_policy']['learner_stats']['policy_loss'] 
        for i in results]
    vf_loss = [
        i['info']['learner']['default_policy']['learner_stats']['vf_loss']
        for i in results]
    
    p = 100
    mean_rewards = np.array([np.mean(rewards[i-p:i+1]) 
                    if i >= p else np.mean(rewards[:i+1]) 
                    for i, _ in enumerate(rewards)])
    std_rewards = np.array([np.std(rewards[i-p:i+1])
                if i >= p else np.std(rewards[:i+1])
                for i, _ in enumerate(rewards)])
    
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax0 = fig.add_subplot(gs[:, :-2])
    ax0.fill_between(np.arange(len(mean_rewards)), 
                    mean_rewards - std_rewards, 
                    mean_rewards + std_rewards,
                    label='Standard Deviation', alpha=0.3)
    ax0.plot(mean_rewards, label='Mean Rewards')
    ax0.set_ylabel('Rewards')
    ax0.set_xlabel('Episode')
    ax0.set_title('Training Rewards')
    ax0.legend()
    
    ax1 = fig.add_subplot(gs[0, 2:])
    ax1.plot(pol_loss)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_title('Policy Loss')
    
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.plot(vf_loss)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Value Function Loss')
    
    plt.show()