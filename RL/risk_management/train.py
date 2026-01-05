"""
SAC Training Script for Risk Management

This script trains the SAC agent on the TradingRiskEnv environment
to learn risk-adjusted position sizing.

Usage:
    # Synthetic data (default)
    python -m RL.risk_management.train --episodes 10000 --device cuda

    # Historical data from Yahoo Finance
    python -m RL.risk_management.train --episodes 10000 --use-historical --start-date 2020-01-01

    # With Weights & Biases logging:
    python -m RL.risk_management.train --episodes 10000 --wandb --wandb-project sac-risk

The trained model can be loaded into the QuantConnect algorithm
for live risk management.
"""

import argparse
import os
import json
import time
from datetime import datetime
from collections import deque
from typing import Dict, Optional, List

import numpy as np
import torch

from .sac_agent import SACAgent, create_sac_agent
from .environment import TradingRiskEnv, VectorizedTradingEnv
from .reward_functions import RiskRewardConfig
from .data_loader import (
    HistoricalDataLoader,
    HistoricalEpisodeEnv,
    download_and_prepare_data,
)
from .qc_data_loader import load_hybrid_episodes, QCHistoricalEpisodeEnv

# Weights & Biases integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingLogger:
    """Logger for training metrics with optional W&B integration."""

    def __init__(
        self,
        log_dir: str = 'logs/sac_training',
        use_wandb: bool = False,
        wandb_project: str = 'stock-split-regression-sac',
        wandb_run_name: Optional[str] = None,
        wandb_entity: str = 'bzhou1018-uc-berkeley-electrical-engineering-computer-sc',
        wandb_config: Optional[Dict] = None,
    ):
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.json')

        self.metrics: List[Dict] = []
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Initialize Weights & Biases
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or f"sac_run_{timestamp}",
                config=wandb_config or {},
                save_code=True,
            )
            print(f"W&B initialized: {wandb.run.url}")

    def log_episode(self, episode: int, reward: float, length: int,
                    metrics: Optional[Dict] = None):
        """Log episode results."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)

        entry = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'avg_reward_100': avg_reward,
            'avg_length_100': avg_length,
            'timestamp': datetime.now().isoformat(),
        }

        if metrics:
            entry.update(metrics)

        self.metrics.append(entry)

        # Log to W&B
        if self.use_wandb:
            wandb_log = {
                'episode': episode,
                'reward': reward,
                'episode_length': length,
                'avg_reward_100': avg_reward,
                'avg_length_100': avg_length,
            }
            if metrics:
                wandb_log.update({
                    'critic_loss': metrics.get('critic_loss', 0),
                    'actor_loss': metrics.get('actor_loss', 0),
                    'alpha': metrics.get('alpha', 0),
                    'entropy': metrics.get('entropy', 0),
                    'q_value': metrics.get('q_value', 0),
                })
            wandb.log(wandb_log, step=episode)

    def log_eval(self, episode: int, eval_reward: float, is_best: bool = False, train_reward: float = None):
        """Log evaluation results and train/eval gap for overfitting detection."""
        if self.use_wandb:
            log_dict = {
                'eval_reward': eval_reward,
                'is_best': is_best,
            }
            # Log train/eval gap - positive gap may indicate overfitting
            if train_reward is not None:
                log_dict['train_eval_gap'] = train_reward - eval_reward
            wandb.log(log_dict, step=episode)

    def log_hyperparameters(self, config: Dict):
        """Log hyperparameters to W&B."""
        if self.use_wandb:
            wandb.config.update(config)

    def save(self):
        """Save metrics to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def finish(self):
        """Finish W&B run."""
        if self.use_wandb:
            wandb.finish()

    def print_progress(self, episode: int, total_episodes: int):
        """Print training progress."""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0

        print(f"Episode {episode}/{total_episodes} | "
              f"Avg Reward (100): {avg_reward:.4f} | "
              f"Avg Length (100): {avg_length:.1f}")


def train_sac(
    num_episodes: int = 10000,
    max_steps: int = 10,
    batch_size: int = 256,
    updates_per_step: int = 1,
    warmup_episodes: int = 100,
    eval_frequency: int = 100,
    save_frequency: int = 1000,
    device: str = 'auto',
    log_dir: str = 'logs/sac_training',
    checkpoint_dir: str = 'checkpoints',
    seed: Optional[int] = None,
    reward_config: Optional[RiskRewardConfig] = None,
    verbose: bool = True,
    # SAC hyperparameters
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    tau: float = 0.005,
    # Weights & Biases options
    use_wandb: bool = False,
    wandb_project: str = 'stock-split-regression-sac',
    wandb_run_name: Optional[str] = None,
    wandb_entity: str = 'bzhou1018-uc-berkeley-electrical-engineering-computer-sc',
    # Optional environment (for historical data training)
    env=None,
    eval_env=None,
    # Training data mode for logging
    data_mode: str = 'synthetic',
    num_training_episodes: Optional[int] = None,
) -> SACAgent:
    """
    Train the SAC agent for risk management.

    Args:
        num_episodes: Total training episodes
        max_steps: Maximum steps per episode
        batch_size: Batch size for training
        updates_per_step: Number of gradient updates per environment step
        warmup_episodes: Episodes before starting training
        eval_frequency: Episodes between evaluations
        save_frequency: Episodes between checkpoints
        device: Device to use ('auto', 'cuda', 'cpu')
        log_dir: Directory for training logs
        checkpoint_dir: Directory for model checkpoints
        seed: Random seed for reproducibility
        reward_config: Reward function configuration
        verbose: Print progress during training
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        wandb_entity: W&B entity (username or team)
        env: Pre-created environment (for historical data training)
        eval_env: Separate environment for evaluation (optional)

    Returns:
        Trained SACAgent
    """
    # Set seeds
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create or use provided environment
    if env is None:
        env = TradingRiskEnv(
            reward_config=reward_config,
            max_steps=max_steps,
            synthetic_mode=True,
        )

    # Use provided eval_env or fallback to main env
    if eval_env is None:
        eval_env = env

    # Create agent
    agent = create_sac_agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        tau=tau,
    )

    # Hyperparameters for logging
    hyperparams = {
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'batch_size': batch_size,
        'updates_per_step': updates_per_step,
        'warmup_episodes': warmup_episodes,
        'eval_frequency': eval_frequency,
        'gamma': agent.gamma,
        'tau': agent.tau,
        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'state_dim': env.state_dim,
        'action_dim': env.action_dim,
        'device': agent.device,
        'seed': seed,
        'data_mode': data_mode,
        'num_training_episodes': num_training_episodes,
    }

    # Create logger with W&B integration
    logger = TrainingLogger(
        log_dir=log_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        wandb_config=hyperparams,
    )

    # Training stats
    total_steps = 0
    best_avg_reward = float('-inf')
    training_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print("SAC Risk Management Training")
        print(f"{'='*60}")
        print(f"Device: {agent.device}")
        print(f"Data mode: {data_mode}")
        if num_training_episodes:
            print(f"Training data episodes: {num_training_episodes}")
        print(f"Training iterations: {num_episodes}")
        print(f"Max steps per episode: {max_steps}")
        print(f"Batch size: {batch_size}")
        print(f"Warmup episodes: {warmup_episodes}")
        print(f"W&B logging: {'enabled' if use_wandb and WANDB_AVAILABLE else 'disabled'}")
        print(f"{'='*60}\n")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_metrics = {}

        done = False
        while not done:
            # Select action
            if episode < warmup_episodes:
                # Random exploration during warmup
                action = np.random.uniform(-1, 1, size=(1,))
            else:
                action = agent.select_action(state, evaluate=False)

            # Environment step
            next_state, reward, done, info = env.step(action[0] if len(action.shape) > 0 else action)

            # Store transition
            agent.store_transition(
                state,
                action if isinstance(action, np.ndarray) else np.array([action]),
                reward,
                next_state,
                done
            )

            # Update agent
            if episode >= warmup_episodes:
                for _ in range(updates_per_step):
                    update_metrics = agent.update()
                    if update_metrics:
                        episode_metrics = update_metrics

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

        # Log episode
        logger.log_episode(episode, episode_reward, episode_length, episode_metrics)

        # Print progress
        if verbose and episode % 100 == 0:
            logger.print_progress(episode, num_episodes)

            if episode_metrics:
                print(f"  Critic Loss: {episode_metrics.get('critic_loss', 0):.4f} | "
                      f"Alpha: {episode_metrics.get('alpha', 0):.4f} | "
                      f"Entropy: {episode_metrics.get('entropy', 0):.4f}")

        # Evaluation
        if episode % eval_frequency == 0:
            eval_reward = evaluate_agent(agent, eval_env, num_episodes=10)
            is_best = eval_reward > best_avg_reward

            if verbose:
                print(f"  Eval Reward (10 eps): {eval_reward:.4f}")

            # Log evaluation to W&B (include train avg for overfitting detection)
            train_avg = np.mean(logger.episode_rewards) if logger.episode_rewards else 0
            logger.log_eval(episode, eval_reward, is_best=is_best, train_reward=train_avg)

            # Save best model
            if is_best:
                best_avg_reward = eval_reward
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                agent.save(best_model_path)
                if verbose:
                    print(f"  [*] New best model saved!")

                # Best model logged at end of training via artifact

        # Regular checkpoints
        if episode % save_frequency == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth'))
            logger.save()

    # Final save
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    agent.save(final_model_path)
    logger.save()

    # Log models to W&B as artifacts
    if use_wandb and WANDB_AVAILABLE:
        artifact = wandb.Artifact(
            name=f'sac-risk-model-{wandb.run.id}',
            type='model',
            description='SAC risk management model checkpoints'
        )
        artifact.add_file(best_model_path)
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)

        wandb.log({
            'final_alpha': agent.alpha.item(),
            'total_steps': total_steps,
            'best_eval_reward': best_avg_reward,
        })

    # Training summary
    training_time = time.time() - training_start
    if verbose:
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}")
        print(f"Total time: {training_time/3600:.2f} hours")
        print(f"Total steps: {total_steps}")
        print(f"Best eval reward: {best_avg_reward:.4f}")
        print(f"Final alpha: {agent.alpha.item():.4f}")
        print(f"{'='*60}")

    # Finish W&B run
    logger.finish()

    return agent


def train_hybrid(
    pretrain_episodes: int = 5000,
    finetune_episodes: int = 1000,
    max_steps: int = 10,
    batch_size: int = 256,
    warmup_episodes: int = 100,
    eval_frequency: int = 100,
    save_frequency: int = 1000,
    device: str = 'auto',
    log_dir: str = 'logs/sac_hybrid',
    checkpoint_dir: str = 'checkpoints',
    seed: Optional[int] = None,
    verbose: bool = True,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    finetune_lr_factor: float = 0.1,
    tau: float = 0.005,
    use_wandb: bool = False,
    wandb_project: str = 'stock-split-regression-sac',
    wandb_run_name: Optional[str] = None,
    wandb_entity: str = 'bzhou1018-uc-berkeley-electrical-engineering-computer-sc',
    reward_type: str = 'simple',
    raw_data_file: str = 'research/data/price_trajectories.pkl',
    strategy_data_file: str = 'research/data/strategy_results.pkl',
    include_catastrophic: bool = True,
    catastrophic_ratio: float = 0.3,
) -> SACAgent:
    """
    Hybrid training: pre-train on raw data, fine-tune on strategy data.

    This approach addresses the limited strategy data problem:
    1. Pre-train on 30K+ raw price trajectories (general risk patterns)
    2. Fine-tune on 56 strategy trades (strategy-specific behavior)

    Args:
        pretrain_episodes: Training iterations for pre-training phase
        finetune_episodes: Training iterations for fine-tuning phase
        finetune_lr_factor: Learning rate multiplier for fine-tuning (0.1 = 10x smaller)
        Other args: Same as train_sac

    Returns:
        Trained SACAgent
    """
    # Load hybrid data
    data = load_hybrid_episodes(
        raw_data_file=raw_data_file,
        strategy_data_file=strategy_data_file,
        include_catastrophic_synthetic=include_catastrophic,
        catastrophic_ratio=catastrophic_ratio,
    )

    pretrain_train, pretrain_eval = data['pretrain']
    finetune_train, finetune_eval = data['finetune']

    if len(pretrain_train) == 0:
        raise ValueError("No pre-training data found. Run export first.")

    # Create pre-training environment
    pretrain_env = QCHistoricalEpisodeEnv(pretrain_train, reward_type=reward_type)
    pretrain_eval_env = QCHistoricalEpisodeEnv(pretrain_eval, reward_type=reward_type)

    # ===== PHASE 1: PRE-TRAINING =====
    if verbose:
        print("\n" + "="*60)
        print("PHASE 1: PRE-TRAINING on Raw Price Data")
        print("="*60)
        print(f"  Training episodes: {len(pretrain_train):,}")
        print(f"  Iterations: {pretrain_episodes}")
        print(f"  Learning rate: {lr_actor}")

    agent = train_sac(
        num_episodes=pretrain_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        warmup_episodes=warmup_episodes,
        eval_frequency=eval_frequency,
        save_frequency=save_frequency,
        device=device,
        log_dir=os.path.join(log_dir, 'pretrain'),
        checkpoint_dir=os.path.join(checkpoint_dir, 'pretrain'),
        seed=seed,
        verbose=verbose,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        tau=tau,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=f"{wandb_run_name}_pretrain" if wandb_run_name else None,
        wandb_entity=wandb_entity,
        env=pretrain_env,
        eval_env=pretrain_eval_env,
        data_mode='hybrid_pretrain',
        num_training_episodes=len(pretrain_train),
    )

    # Save pre-trained model
    pretrain_path = os.path.join(checkpoint_dir, 'pretrained_model.pth')
    agent.save(pretrain_path)
    if verbose:
        print(f"\n  Pre-trained model saved: {pretrain_path}")

    # ===== PHASE 2: FINE-TUNING =====
    if len(finetune_train) == 0:
        if verbose:
            print("\n  No fine-tuning data available. Skipping Phase 2.")
        return agent

    if verbose:
        print("\n" + "="*60)
        print("PHASE 2: FINE-TUNING on Strategy Data")
        print("="*60)
        print(f"  Training episodes: {len(finetune_train)}")
        print(f"  Iterations: {finetune_episodes}")
        print(f"  Learning rate: {lr_actor * finetune_lr_factor} (reduced {finetune_lr_factor}x)")

    # Create fine-tuning environment
    finetune_env = QCHistoricalEpisodeEnv(finetune_train, reward_type=reward_type)
    finetune_eval_env = QCHistoricalEpisodeEnv(finetune_eval, reward_type=reward_type) if finetune_eval else finetune_env

    # Reduce learning rate for fine-tuning
    finetune_lr = lr_actor * finetune_lr_factor

    # Update agent's learning rates
    for param_group in agent.actor_optim.param_groups:
        param_group['lr'] = finetune_lr
    for param_group in agent.critic_optim.param_groups:
        param_group['lr'] = finetune_lr

    # Continue training on strategy data
    # Note: We're continuing with the same agent, not creating a new one
    os.makedirs(os.path.join(log_dir, 'finetune'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'finetune'), exist_ok=True)

    # Fine-tuning loop
    best_eval_reward = float('-inf')
    for episode in range(1, finetune_episodes + 1):
        state = finetune_env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = finetune_env.step(action[0])
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward

        # Periodic evaluation
        if episode % eval_frequency == 0:
            eval_reward = evaluate_agent(agent, finetune_eval_env, num_episodes=min(10, len(finetune_eval)))
            if verbose:
                print(f"  Episode {episode}/{finetune_episodes} | "
                      f"Train: {episode_reward:.4f} | Eval: {eval_reward:.4f}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(checkpoint_dir, 'finetune', 'best_finetuned.pth'))
                if verbose:
                    print(f"    [*] New best fine-tuned model!")

    # Save final fine-tuned model
    final_path = os.path.join(checkpoint_dir, 'finetuned_model.pth')
    agent.save(final_path)

    if verbose:
        print("\n" + "="*60)
        print("HYBRID TRAINING COMPLETE")
        print("="*60)
        print(f"  Pre-trained model: {pretrain_path}")
        print(f"  Fine-tuned model: {final_path}")
        print(f"  Best eval reward: {best_eval_reward:.4f}")
        print("="*60)

    return agent


def evaluate_agent(
    agent: SACAgent,
    env: TradingRiskEnv,
    num_episodes: int = 10,
) -> float:
    """
    Evaluate agent performance.

    Args:
        agent: SAC agent to evaluate
        env: Environment to evaluate in
        num_episodes: Number of evaluation episodes

    Returns:
        Average reward across episodes
    """
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _ = env.step(action[0] if len(action.shape) > 0 else action)
            episode_reward += reward

        rewards.append(episode_reward)

    return np.mean(rewards)


def load_and_evaluate(
    checkpoint_path: str,
    num_episodes: int = 100,
    device: str = 'auto',
) -> Dict:
    """
    Load a trained model and evaluate it.

    Args:
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of evaluation episodes
        device: Device to use

    Returns:
        Dictionary of evaluation metrics
    """
    # Create environment
    env = TradingRiskEnv(synthetic_mode=True)

    # Create and load agent
    agent = create_sac_agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
    )
    agent.load(checkpoint_path)

    # Collect metrics
    rewards = []
    lengths = []
    margin_calls = 0
    position_changes = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        prev_action = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            action_val = action[0] if len(action.shape) > 0 else action

            state, reward, done, info = env.step(action_val)

            episode_reward += reward
            episode_length += 1
            position_changes.append(abs(action_val - prev_action))
            prev_action = action_val

            if info.get('margin_call'):
                margin_calls += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'margin_call_rate': margin_calls / num_episodes,
        'avg_position_change': np.mean(position_changes),
        'alpha': agent.alpha.item(),
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train SAC Risk Management Agent')

    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=10,
                        help='Maximum steps per episode')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Actor learning rate')
    parser.add_argument('--lr-critic', type=float, default=3e-4,
                        help='Critic learning rate')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient for target networks')
    parser.add_argument('--warmup', type=int, default=100,
                        help='Warmup episodes before training')
    parser.add_argument('--reward-type', type=str, default='simple',
                        choices=['simple', 'pnl_only', 'sac', 'sparse', 'dense', 'curriculum', 'risk_management'],
                        help='Reward function type (default: simple, recommended: risk_management)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--log-dir', type=str, default='logs/sac_training',
                        help='Directory for training logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--eval-freq', type=int, default=100,
                        help='Episodes between evaluations')
    parser.add_argument('--save-freq', type=int, default=1000,
                        help='Episodes between checkpoints')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    # Weights & Biases options
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='stock-split-regression-sac',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not specified)')
    parser.add_argument('--wandb-entity', type=str,
                        default='bzhou1018-uc-berkeley-electrical-engineering-computer-sc',
                        help='W&B entity (username or team)')

    # Historical data options (yfinance)
    parser.add_argument('--use-historical', action='store_true',
                        help='Use historical stock data from yfinance (limited to 2 years hourly)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for historical data (default: today)')
    parser.add_argument('--data-cache', type=str, default='data/price_cache',
                        help='Directory to cache downloaded price data')
    parser.add_argument('--include-catastrophic', action='store_true', default=False,
                        help='Include synthetic catastrophic scenarios in training (default: False)')
    parser.add_argument('--catastrophic-ratio', type=float, default=0.5,
                        help='Ratio of catastrophic to normal episodes (default: 0.5)')

    # QuantConnect data options (higher quality, recommended)
    parser.add_argument('--use-qc-data', action='store_true',
                        help='Use QuantConnect exported data (higher quality, recommended)')
    parser.add_argument('--qc-episodes-file', type=str, default='research/data/price_trajectories.pkl',
                        help='Path to QC exported episodes pickle file')
    parser.add_argument('--qc-metadata-file', type=str, default='research/data/export_metadata.json',
                        help='Path to QC export metadata JSON file')

    # Hybrid training (recommended for limited strategy data)
    parser.add_argument('--hybrid', action='store_true',
                        help='Use hybrid training: pre-train on raw data, fine-tune on strategy data')
    parser.add_argument('--pretrain-episodes', type=int, default=5000,
                        help='Number of pre-training iterations (default: 5000)')
    parser.add_argument('--finetune-episodes', type=int, default=1000,
                        help='Number of fine-tuning iterations (default: 1000)')
    parser.add_argument('--finetune-lr-factor', type=float, default=0.1,
                        help='Learning rate multiplier for fine-tuning (default: 0.1)')
    parser.add_argument('--raw-data-file', type=str, default='research/data/price_trajectories.pkl',
                        help='Path to raw price trajectories for pre-training')
    parser.add_argument('--strategy-data-file', type=str, default='research/data/strategy_results.pkl',
                        help='Path to strategy results for fine-tuning')

    # Evaluation mode
    parser.add_argument('--eval', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes')

    args = parser.parse_args()

    if args.eval:
        # Evaluation mode
        print(f"Evaluating model: {args.eval}")
        metrics = load_and_evaluate(
            args.eval,
            num_episodes=args.eval_episodes,
            device=args.device,
        )
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"  Mean Length: {metrics['mean_length']:.2f}")
        print(f"  Margin Call Rate: {metrics['margin_call_rate']:.2%}")
        print(f"  Avg Position Change: {metrics['avg_position_change']:.4f}")
        print(f"  Final Alpha: {metrics['alpha']:.4f}")
    else:
        # Check W&B availability
        if args.wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            print("Continuing without W&B logging...")

        # ===== HYBRID TRAINING MODE (RECOMMENDED) =====
        if args.hybrid:
            print("\n" + "="*60)
            print("HYBRID TRAINING MODE")
            print("="*60)
            print("Pre-training on raw price data, fine-tuning on strategy data")
            print("="*60)

            agent = train_hybrid(
                pretrain_episodes=args.pretrain_episodes,
                finetune_episodes=args.finetune_episodes,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                warmup_episodes=args.warmup,
                eval_frequency=args.eval_freq,
                save_frequency=args.save_freq,
                device=args.device,
                log_dir=args.log_dir,
                checkpoint_dir=args.checkpoint_dir,
                seed=args.seed,
                verbose=not args.quiet,
                lr_actor=args.lr_actor,
                lr_critic=args.lr_critic,
                finetune_lr_factor=args.finetune_lr_factor,
                tau=args.tau,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name,
                wandb_entity=args.wandb_entity,
                reward_type=args.reward_type,
                raw_data_file=args.raw_data_file,
                strategy_data_file=args.strategy_data_file,
                include_catastrophic=args.include_catastrophic,
                catastrophic_ratio=args.catastrophic_ratio,
            )
            return

        # Create environment based on mode
        env = None
        eval_env = None
        episodes = None  # Track for logging

        # ===== OPTION 1: QuantConnect exported data (RECOMMENDED) =====
        if args.use_qc_data:
            print("\n" + "="*60)
            print("QuantConnect Data Training Mode (RECOMMENDED)")
            print("="*60)

            try:
                from .qc_data_loader import load_qc_episodes, QCHistoricalEpisodeEnv

                train_episodes, eval_episodes = load_qc_episodes(
                    episodes_file=args.qc_episodes_file,
                    metadata_file=args.qc_metadata_file,
                    train_ratio=0.8,
                    include_catastrophic_synthetic=args.include_catastrophic,
                    catastrophic_ratio=args.catastrophic_ratio,
                )

                if len(train_episodes) == 0:
                    print("Error: No training episodes loaded.")
                    return

                print(f"  Reward function: {args.reward_type}")
                env = QCHistoricalEpisodeEnv(train_episodes, reward_type=args.reward_type)
                eval_env = QCHistoricalEpisodeEnv(eval_episodes, reward_type=args.reward_type)
                episodes = train_episodes + eval_episodes
                # Episode counts already printed by load_qc_episodes()
                print("="*60 + "\n")

            except FileNotFoundError as e:
                print(f"\nError: {e}")
                print("\nTo generate QC data:")
                print("  1. Open QuantConnect Research notebook")
                print("  2. Copy and run: research/export_training_data.py")
                print("  3. Download price_trajectories.pkl to research/")
                print("\nAlternatively, use --use-historical for yfinance data (lower quality)")
                return

        # ===== OPTION 2: yfinance historical data =====
        elif args.use_historical:
            print("\n" + "="*60)
            print("Historical Data Training Mode")
            print("="*60)

            # Create data loader
            loader = HistoricalDataLoader(cache_dir=args.data_cache)

            # Download historical data
            print(f"\nDownloading data from {args.start_date} to {args.end_date or 'today'}...")
            loader.download_data(
                start_date=args.start_date,
                end_date=args.end_date,
                interval='1h',
            )

            # Create training episodes
            print("\nCreating training episodes...")
            episodes = loader.create_training_episodes(
                episodes_per_symbol=100,
            )

            # Optionally add catastrophic scenarios (only to training set)
            catastrophic_episodes = []
            if args.include_catastrophic:
                num_catastrophic = int(len(episodes) * args.catastrophic_ratio)
                print(f"\nAdding {num_catastrophic} catastrophic scenarios to training set...")
                catastrophic_episodes = loader.create_catastrophic_scenarios(
                    num_scenarios=num_catastrophic,
                )

            if len(episodes) == 0:
                print("Error: No episodes created. Check data download.")
                return

            # ===== TIME-BASED TRAIN/TEST SPLIT (PREVENTS DATA LEAKAGE) =====
            # Sort episodes by entry date
            episodes_sorted = sorted(episodes, key=lambda e: e.entry_date)

            # Use 80% oldest data for training, 20% newest for evaluation
            train_ratio = 0.8
            split_idx = int(len(episodes_sorted) * train_ratio)

            train_episodes = episodes_sorted[:split_idx]
            eval_episodes = episodes_sorted[split_idx:]

            # Add catastrophic scenarios ONLY to training set
            if catastrophic_episodes:
                train_episodes = train_episodes + catastrophic_episodes

            print(f"\n  TIME-BASED SPLIT (preventing data leakage):")
            print(f"    Training episodes: {len(train_episodes)} (oldest 80% + catastrophic)")
            print(f"    Evaluation episodes: {len(eval_episodes)} (newest 20%)")
            if len(train_episodes) > 0 and len(eval_episodes) > 0:
                train_dates = [e.entry_date for e in train_episodes if not e.symbol.startswith('CATASTROPHIC')]
                eval_dates = [e.entry_date for e in eval_episodes]
                if train_dates and eval_dates:
                    print(f"    Training date range: {min(train_dates).date()} to {max(train_dates).date()}")
                    print(f"    Evaluation date range: {min(eval_dates).date()} to {max(eval_dates).date()}")

            # Create environments with SEPARATE data
            env = HistoricalEpisodeEnv(train_episodes)
            eval_env = HistoricalEpisodeEnv(eval_episodes)

            print(f"\nEnvironments created:")
            print(f"  Training: {len(train_episodes)} episodes")
            print(f"  Evaluation: {len(eval_episodes)} episodes (no overlap with training)")
            print("="*60 + "\n")

        # Determine data mode for logging
        if args.use_qc_data:
            data_mode = 'quantconnect'
        elif args.use_historical:
            data_mode = 'yfinance'
        else:
            data_mode = 'synthetic'
        num_data_episodes = len(episodes) if episodes else None

        # Training mode
        agent = train_sac(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            warmup_episodes=args.warmup,
            device=args.device,
            seed=args.seed,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            eval_frequency=args.eval_freq,
            save_frequency=args.save_freq,
            verbose=not args.quiet,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            tau=args.tau,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_entity=args.wandb_entity,
            env=env,
            eval_env=eval_env,
            data_mode=data_mode,
            num_training_episodes=num_data_episodes,
        )


if __name__ == "__main__":
    main()
