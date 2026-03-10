"""
train.py
--------
Train the drone racing RL agent using Stable-Baselines3 PPO.

Run:
    python training/train.py --config training/configs/ppo_config.yaml

The trained policy is saved to:
    control/models/rl_policy.zip
"""

import argparse
import yaml
import os
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from envs.drone_racing_env import DroneRacingEnv
from envs.reward_functions  import CurriculumManager


class CurriculumCallback(BaseCallback):
    """
    SB3 callback that advances curriculum based on success rate.
    Updates environment config when stage changes.
    """

    def __init__(self, curriculum: CurriculumManager,
                       eval_env,
                       verbose: int = 1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.eval_env   = eval_env
        self.n_episodes = 0

    def _on_step(self) -> bool:
        # Check for episode ends in training envs
        for info in self.locals.get('infos', []):
            if 'gates_passed' in info:
                n_gates = self.training_env.get_attr('n_gates')[0]
                self.curriculum.record_episode(
                    info['gates_passed'], n_gates
                )
                self.n_episodes += 1

                if self.curriculum.maybe_advance():
                    new_config = self.curriculum.get_config()
                    # Update all parallel envs
                    for attr, val in new_config.items():
                        self.training_env.set_attr(attr, val)
                    if self.verbose:
                        print(f'  Updated env config: {new_config}')
        return True


class TensorboardCallback(BaseCallback):
    """Log custom metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards  = []
        self.episode_speeds   = []
        self.episode_gates    = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'total_reward' in info:
                self.episode_rewards.append(info['total_reward'])
                self.episode_speeds.append(info.get('speed', 0))
                self.episode_gates.append(info.get('gates_passed', 0))

        if len(self.episode_rewards) >= 10:
            self.logger.record('racing/mean_reward',
                               np.mean(self.episode_rewards))
            self.logger.record('racing/mean_speed',
                               np.mean(self.episode_speeds))
            self.logger.record('racing/mean_gates_passed',
                               np.mean(self.episode_gates))
            self.episode_rewards.clear()
            self.episode_speeds.clear()
            self.episode_gates.clear()
        return True


def train(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    log_dir    = cfg['logging']['log_dir']
    model_dir  = cfg['logging']['model_dir']
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get('environment', {})

    # ── Create parallel training environments ──────────────────────────────
    n_envs = cfg['training'].get('n_envs', 8)
    print(f'Creating {n_envs} parallel training environments...')

    train_env = make_vec_env(
        DroneRacingEnv,
        n_envs=n_envs,
        env_kwargs={'config': env_cfg},
        vec_env_cls=SubprocVecEnv
    )

    # Normalize observations and rewards — critical for PPO stability
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    # ── Evaluation environment (not normalized separately) ─────────────────
    eval_env = make_vec_env(
        DroneRacingEnv,
        n_envs=1,
        env_kwargs={'config': {**env_cfg, 'randomize_gates': False}}
    )

    # ── Curriculum manager ─────────────────────────────────────────────────
    curriculum = CurriculumManager(
        success_threshold=cfg['curriculum'].get('success_threshold', 0.75),
        window_size=cfg['curriculum'].get('window_size', 100)
    )

    # ── Build PPO model ────────────────────────────────────────────────────
    ppo_cfg = cfg['ppo']
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=ppo_cfg.get('learning_rate', 3e-4),
        n_steps=ppo_cfg.get('n_steps', 2048),
        batch_size=ppo_cfg.get('batch_size', 256),
        n_epochs=ppo_cfg.get('n_epochs', 10),
        gamma=ppo_cfg.get('gamma', 0.99),
        gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
        clip_range=ppo_cfg.get('clip_range', 0.2),
        ent_coef=ppo_cfg.get('ent_coef', 0.005),
        vf_coef=ppo_cfg.get('vf_coef', 0.5),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],   # policy network
                vf=[256, 256, 128]    # value network
            ),
            activation_fn=__import__('torch.nn', fromlist=['Tanh']).Tanh
        ),
        tensorboard_log=log_dir,
        verbose=1,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
    )

    print(f'PPO model created. Device: {model.device}')
    print(f'Policy network: {model.policy}')

    # ── Load checkpoint if resuming ────────────────────────────────────────
    resume_path = cfg['training'].get('resume_from', None)
    if resume_path and os.path.exists(resume_path):
        print(f'Resuming from: {resume_path}')
        model = PPO.load(resume_path, env=train_env)

    # ── Callbacks ──────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg['logging'].get('checkpoint_freq', 50000),
        save_path=model_dir,
        name_prefix='drone_racing_ppo'
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=cfg['logging'].get('eval_freq', 10000),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1
    )

    curriculum_cb  = CurriculumCallback(curriculum, eval_env)
    tensorboard_cb = TensorboardCallback()

    # ── Train ──────────────────────────────────────────────────────────────
    total_steps = cfg['training'].get('total_timesteps', 5_000_000)
    print(f'\n🚀 Training for {total_steps:,} timesteps...')

    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, eval_cb, curriculum_cb, tensorboard_cb],
        progress_bar=True
    )

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(model_dir, 'rl_policy_final')
    model.save(final_path)
    train_env.save(os.path.join(model_dir, 'vec_normalize.pkl'))

    print(f'\n✅ Training complete!')
    print(f'   Model saved: {final_path}.zip')
    print(f'   Copy to:     drone_racing_ws/src/control/models/rl_policy.zip')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ppo_config.yaml')
    args = parser.parse_args()
    train(args.config)