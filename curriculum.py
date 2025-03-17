"""
Curriculum learning to gradually increase difficulty of graph rewiring.
"""
import numpy as np
import torch
import gym
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumScheduler:
    def __init__(self, env, config, difficulty_metrics=['graph_size', 'edge_density']):
        """
        Curriculum scheduler that gradually increases task difficulty.

        Args:
            env: The graph environment
            config: Configuration object
            difficulty_metrics: What metrics to use for difficulty
        """
        self.env = env
        self.config = config
        self.difficulty_metrics = difficulty_metrics
        self.current_stage = 0
        self.max_stages = config.curriculum_stages
        self.success_threshold = config.curriculum_success_threshold
        self.window_size = config.curriculum_window_size

        # Performance tracking
        self.performance_history = []

        # Dataset sorting by difficulty
        self.sorted_indices = self._sort_dataset_by_difficulty()

    def _sort_dataset_by_difficulty(self):
        """Sort dataset examples by difficulty"""
        dataset = self.env.dataset
        difficulties = []

        for i, data in enumerate(dataset):
            difficulty_score = 0

            if 'graph_size' in self.difficulty_metrics:
                # Larger graphs are harder
                difficulty_score += data.x.size(0) / 100

            if 'edge_density' in self.difficulty_metrics:
                # Denser graphs might be harder to modify effectively
                edge_count = data.edge_index.size(1) / 2  # Assuming undirected
                max_edges = data.x.size(0) * (data.x.size(0) - 1) / 2
                if max_edges > 0:
                    edge_density = edge_count / max_edges
                    difficulty_score += edge_density

            if 'feature_dim' in self.difficulty_metrics:
                # More complex node features might be harder
                difficulty_score += data.x.size(1) / 10

            difficulties.append((i, difficulty_score))

        # Sort by difficulty score
        difficulties.sort(key=lambda x: x[1])
        return [idx for idx, _ in difficulties]

    def update_curriculum(self):
        """Check if we should advance to the next curriculum stage"""
        if len(self.performance_history) < self.window_size:
            return False

        # Check if average performance in recent episodes meets threshold
        recent_performance = self.performance_history[-self.window_size:]
        avg_improvement = sum(recent_performance) / len(recent_performance)

        if avg_improvement >= self.success_threshold and self.current_stage < self.max_stages - 1:
            self.current_stage += 1
            print(f"Curriculum advancing to stage {self.current_stage}/{self.max_stages-1}")
            return True

        return False

    def get_current_dataset(self):
        """Get the subset of the dataset for the current curriculum stage"""
        # Calculate what portion of the dataset to use based on current stage
        if self.max_stages <= 1:
            return self.env.dataset

        proportion = (self.current_stage + 1) / self.max_stages
        num_samples = max(1, int(proportion * len(self.sorted_indices)))

        # Get indices for the current stage
        current_indices = self.sorted_indices[:num_samples]

        # Create a subset of the dataset
        return [self.env.dataset[i] for i in current_indices]

    def update_env_dataset(self):
        """Update the environment with the current curriculum stage's dataset"""
        current_dataset = self.get_current_dataset()
        self.env.dataset = current_dataset
        self.env.current_graph_idx = 0  # Reset graph index

        # Additional curriculum parameters can be adjusted here
        if hasattr(self.config, 'curriculum_step_increase'):
            # Gradually increase max steps per episode
            base_steps = self.config.max_steps_per_episode_base
            step_increase = self.config.curriculum_step_increase
            self.env.config.max_steps_per_episode = base_steps + self.current_stage * step_increase

        if hasattr(self.config, 'curriculum_candidates_increase'):
            # Gradually increase action space size
            base_candidates = self.config.max_candidates_base
            candidate_increase = self.config.curriculum_candidates_increase
            new_candidates = base_candidates + self.current_stage * candidate_increase
            self.env.max_candidates = min(new_candidates, self.config.max_candidates_max)
            # Update action space
            self.env.action_space = gym.spaces.Discrete(self.env.max_candidates)

        return current_dataset

    def record_performance(self, relative_improvement):
        """Record the performance for curriculum advancement"""
        self.performance_history.append(relative_improvement)
        # Keep only recent history
        if len(self.performance_history) > self.window_size * 2:
            self.performance_history = self.performance_history[-self.window_size*2:]

    def get_curriculum_info(self):
        """Return information about current curriculum state"""
        return {
            "stage": self.current_stage,
            "max_stages": self.max_stages,
            "dataset_size": len(self.get_current_dataset()),
            "total_dataset_size": len(self.env.dataset),
            "recent_performance": np.mean(self.performance_history[-self.window_size:])
                                 if len(self.performance_history) >= self.window_size else 0
        }

class CurriculumCallback(BaseCallback):
    """
    Callback for Stable-Baselines3 to handle curriculum learning
    """
    def __init__(self, curriculum_scheduler, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.curriculum_scheduler = curriculum_scheduler
        self.check_freq = check_freq

    def _init_callback(self):
        # Called when the callback is initialized
        pass

    def _on_step(self):
        """Called at each step of the training"""
        # Check if we should update curriculum
        if self.n_calls % self.check_freq == 0:
            if self.curriculum_scheduler.update_curriculum():
                # Update environment with new curriculum
                self.curriculum_scheduler.update_env_dataset()

                # Print curriculum info
                curr_info = self.curriculum_scheduler.get_curriculum_info()
                print(f"Curriculum updated: Stage {curr_info['stage']}/{curr_info['max_stages']-1}, "
                      f"Dataset size: {curr_info['dataset_size']}/{curr_info['total_dataset_size']}, "
                      f"Recent performance: {curr_info['recent_performance']:.4f}")

        # Always return True to continue training
        return True
