"""
Main entry point for the RL-based GNN rewiring pipeline.

This script:
1. Loads configuration from a YAML file
2. Sets up the dataset, GNN model, and observation extractor
3. Creates the environment with edge operations
4. Optionally sets up curriculum learning
5. Initializes and trains the PPO agent
6. Evaluates the final performance
"""

import os
import yaml
import argparse
import numpy as np
import random
from datetime import datetime
from ml_collections import ConfigDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ZINC, QM9, TUDataset, MoleculeNet
from torch_geometric.loader import DataLoader
from pretransforms import FeatureTransform

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

# Import our custom modules
from edge_operations import EdgeOperator, EdgeOperation
from graph_observation import GraphObservationExtractor
from multi_step_env import MultiStepGraphEnv
from enhanced_gnn import EnhancedGNN
from curriculum import CurriculumScheduler, CurriculumCallback


def load_config(yaml_path):
    """Load YAML config file into ConfigDict"""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert to ConfigDict
    config = ConfigDict()

    # Add main sections
    for section in config_dict:
        if isinstance(config_dict[section], dict):
            config[section] = ConfigDict(config_dict[section])
        else:
            config[section] = config_dict[section]

    return config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(config):
    """Load the dataset based on config"""
    dataset_config = config.dataset
    name = dataset_config.name

    # Create feature transform
    use_one_hot = dataset_config.get('use_one_hot', True)
    transform = FeatureTransform(
        one_hot=use_one_hot,
        num_atom_types=dataset_config.get('num_atom_types', 28),
        num_bond_types=dataset_config.get('num_bond_types', 4)
    )

    if name == "ZINC":
        dataset = ZINC(
            root=dataset_config.root_dir,
            subset=dataset_config.get('subset', True),
            split=dataset_config.get('split', 'train'),
            # transform=transform  # Apply as transform (executes on every access)
            pre_transform=transform  # Uncomment if you want it applied during preprocessing
        )
    elif name == "QM9":
        dataset = QM9(
            root=dataset_config.root_dir,
            transform=None
        )
    elif name == "TUDataset":
        dataset = TUDataset(
            root=dataset_config.root_dir,
            name=dataset_config.tu_name
        )
    elif name == "MoleculeNet":
        dataset = MoleculeNet(
            root=dataset_config.root_dir,
            name=dataset_config.moleculenet_name
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"Loaded {len(dataset)} graphs from {name} dataset")

    # If we need to limit the dataset size for development
    if dataset_config.get('max_samples', -1) > 0:
        limit = min(len(dataset), dataset_config.max_samples)
        dataset = dataset[:limit]
        print(f"Limited dataset to {limit} samples")

    return dataset


def create_gnn(config, dataset):
    """Create and optionally pretrain the GNN model"""
    model_config = config.gnn_model
    device = torch.device("cuda" if torch.cuda.is_available() and not config.general.get("force_cpu", False) else "cpu")

    # Determine input and output dimensions from dataset
    in_channels = dataset[0].x.size(1)

    # Create model
    model = EnhancedGNN(
        in_channels=in_channels,
        hidden_channels=model_config.hidden_channels,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        layer_type=model_config.layer_type,
        readout=model_config.readout,
        use_batch_norm=model_config.use_batch_norm,
        task=model_config.task,
        out_channels=model_config.out_channels
    )

    model = model.to(device)
    print(f"Created GNN model with {model_config.layer_type} layers")

    # Optionally pretrain the GNN
    if model_config.pretrain_epochs > 0:
        print(f"Pretraining GNN for {model_config.pretrain_epochs} epochs")
        loader = DataLoader(dataset, batch_size=model_config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)

        model.train()
        for epoch in range(model_config.pretrain_epochs):
            total_loss = 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()

                if hasattr(data, 'batch') and data.batch is not None:
                    batch = data.batch
                else:
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

                pred = model(data.x, data.edge_index, batch)

                # Handle different regression targets based on dataset
                if config.dataset.name == "ZINC":
                    target = data.y
                elif config.dataset.name == "QM9":
                    # QM9 has multiple targets, select the one specified in config
                    target_idx = model_config.get('target_idx', 0)
                    target = data.y[:, target_idx]
                else:
                    target = data.y

                loss = F.mse_loss(pred, target.unsqueeze(dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{model_config.pretrain_epochs}, Loss: {total_loss / len(loader):.4f}")

    return model, device


def create_observation_extractor(config, dataset):
    """Create the observation extractor for RL state representation"""
    obs_config = config.observation
    device = torch.device("cuda" if torch.cuda.is_available() and not config.general.get("force_cpu", False) else "cpu")

    # Determine input dimensions from dataset
    in_channels = dataset[0].x.size(1)

    # Create observation extractor
    obs_extractor = GraphObservationExtractor(
        in_channels=in_channels,
        hidden_channels=obs_config.hidden_channels,
        embedding_size=obs_config.embedding_size
    ).to(device)

    print(f"Created observation extractor with embedding size {obs_config.embedding_size}")
    return obs_extractor, device


def create_environment(config, dataset, gnn_model, obs_extractor, edge_operator, device):
    """Create the RL environment"""
    env_config = config.environment

    env = MultiStepGraphEnv(
        dataset=dataset,
        gnn_model=gnn_model,
        obs_extractor=obs_extractor,
        edge_operator=edge_operator,
        config=env_config,
        device=device,
        reward_shaping=env_config.get('reward_shaping', True)
    )

    # Run a quick check to ensure the environment works
    check_env(env, warn=True)
    print("Environment check passed")

    return env


def setup_logging(config):
    """Set up logging and checkpoint directories"""
    log_dir = os.path.join(config.general.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # Save the config for reproducibility
    config_path = os.path.join(log_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f)

    print(f"Log directory created at {log_dir}")
    return log_dir


def evaluate_model(env, model, num_episodes=10):
    """Evaluate the RL model's performance"""
    rewards = []
    mses = []
    improvements = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        initial_mse = env.initial_mse

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        # Record metrics
        final_mse = info["mse"]
        improvement = info["improvement"]

        rewards.append(episode_reward)
        mses.append(final_mse)
        improvements.append(improvement)

    # Calculate average metrics
    avg_reward = sum(rewards) / len(rewards)
    avg_mse = sum(mses) / len(mses)
    avg_improvement = sum(improvements) / len(improvements)
    avg_relative_improvement = sum([i["relative_improvement"] for i in improvements]) / len(improvements)

    return {
        "avg_reward": avg_reward,
        "avg_mse": avg_mse,
        "avg_improvement": avg_improvement,
        "avg_relative_improvement": avg_relative_improvement,
        "all_rewards": rewards,
        "all_mses": mses,
        "all_improvements": improvements,
    }


def main(config_path=None):
    """
    Main entry point for the RL-GNN Graph Rewiring pipeline.

    Args:
        config_path: Optional path to the YAML config file.
                     If None, will try to get from command line arguments.
    """
    # If config_path is not provided, try to get it from command line arguments
    if config_path is None:
        parser = argparse.ArgumentParser(description="RL-GNN Graph Rewiring")
        parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
        args = parser.parse_args()
        config_path = args.config

    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded:", config)

    # Set random seed
    set_seed(config.general.seed)

    # Set up logging
    log_dir = setup_logging(config)

    # Load dataset
    dataset = load_dataset(config)

    # Create GNN model
    gnn_model, device = create_gnn(config, dataset)

    # Create observation extractor
    obs_extractor, _ = create_observation_extractor(config, dataset)

    # Create edge operator
    edge_operator = EdgeOperator(strategy=config.edge_operator.strategy)
    print(f"Created edge operator with strategy: {config.edge_operator.strategy}")

    # Create environment
    env = create_environment(config, dataset, gnn_model, obs_extractor, edge_operator, device)

    # Set up curriculum learning if enabled
    if config.general.get('use_curriculum', False):
        print("Setting up curriculum learning")
        curriculum_scheduler = CurriculumScheduler(
            env=env,
            config=config.curriculum,
            difficulty_metrics=config.curriculum.difficulty_metrics
        )

        # Update environment with initial curriculum
        curriculum_scheduler.update_env_dataset()

        # Create curriculum callback
        curriculum_callback = CurriculumCallback(
            curriculum_scheduler=curriculum_scheduler,
            check_freq=config.curriculum.check_freq
        )
    else:
        curriculum_callback = None

    # Create vectorized environment
    if config.training.num_envs > 1:
        # Create multiple environments
        def make_env_fn():
            return create_environment(config, dataset, gnn_model, obs_extractor, edge_operator, device)

        env = make_vec_env(make_env_fn, n_envs=config.training.num_envs)
    else:
        # Use just one environment
        env = DummyVecEnv([lambda: env])  # Wrap single env


    # Setup callbacks
    callbacks = []

    # Add evaluation callback
    if config.training.get('use_eval_callback', True):
        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(log_dir, 'best_model'),
            log_path=os.path.join(log_dir, 'eval_results'),
            eval_freq=config.training.eval_freq,
            n_eval_episodes=config.training.n_eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)

    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.checkpoint_freq,
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix="ppo_model"
    )
    callbacks.append(checkpoint_callback)

    # Add curriculum callback if enabled
    if curriculum_callback is not None:
        callbacks.append(curriculum_callback)

    # Create RL model (PPO)
    model = PPO(
        policy=config.ppo.policy_type,
        env=env,
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        clip_range_vf=config.ppo.get('clip_range_vf', None),
        normalize_advantage=config.ppo.get('normalize_advantage', True),
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        use_sde=config.ppo.get('use_sde', False),
        sde_sample_freq=config.ppo.get('sde_sample_freq', -1),
        target_kl=config.ppo.get('target_kl', None),
        tensorboard_log=os.path.join(log_dir, 'tensorboard'),
        policy_kwargs=dict(
            net_arch=config.ppo.policy_network
        ),
        verbose=1,
        seed=config.general.seed,
        device=device
    )

    print("Starting training...")
    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=callbacks
    )

    # Save the final model
    model.save(os.path.join(log_dir, 'final_model'))

    print("Training complete!")

    # Evaluate the model
    print("Evaluating final model performance...")
    # We need to create a new non-vectorized env for evaluation
    eval_env = create_environment(config, dataset, gnn_model, obs_extractor, edge_operator, device)
    eval_results = evaluate_model(eval_env, model, num_episodes=config.training.n_eval_episodes)

    print(f"Evaluation complete!")
    print(f"Average reward: {eval_results['avg_reward']:.4f}")
    print(f"Average MSE: {eval_results['avg_mse']:.4f}")
    print(f"Average improvement: {eval_results['avg_improvement']:.4f}")
    print(f"Average relative improvement: {eval_results['avg_relative_improvement']:.4f}")

    # Save evaluation results
    eval_path = os.path.join(log_dir, "final_eval_results.yaml")
    with open(eval_path, "w") as f:
        yaml.dump(eval_results, f)

    print(f"Results saved to {eval_path}")

    return model, eval_results


if __name__ == "__main__":
    main(config_path='configs/gnn_rewire.yaml')
