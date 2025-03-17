"""
Enhanced environment that supports multi-step edge modifications
with adaptive rewards and episode termination.
"""
import gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv

class MultiStepGraphEnv(gym.Env):
    """
    Environment that allows multiple edge modifications per episode.
    Supports more complex action space and observation space.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, dataset, gnn_model, obs_extractor, edge_operator, 
                 config, device='cpu', reward_shaping=True):
        super(MultiStepGraphEnv, self).__init__()
        self.dataset = dataset
        self.config = config
        self.device = device
        self.reward_shaping = reward_shaping
        
        # Store models
        self.gnn_model = gnn_model.to(device)
        self.obs_extractor = obs_extractor.to(device)
        self.edge_operator = edge_operator
        
        # Environment state
        self.current_graph_idx = 0
        self.current_data = None
        self.original_data = None
        self.step_count = 0
        self.best_mse = float('inf')
        self.initial_mse = float('inf')
        
        # History tracking
        self.history = []
        
        # Set up observation space: output size of the obs_extractor
        obs_size = self.obs_extractor.final_projection.out_features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Action space: action_type (add, remove, rewire) + which candidate edge
        # For this example, we'll use discrete actions for simplicity
        self.max_candidates = config.max_candidates
        self.action_space = gym.spaces.Discrete(self.max_candidates)
        
        # Prepare for first reset
        self._setup_new_episode()
        
    def _setup_new_episode(self):
        """
        Initialize a new episode with a fresh graph from the dataset.
        """
        # Select a graph from the dataset
        if self.current_graph_idx >= len(self.dataset):
            self.current_graph_idx = 0
        
        # Deep copy the graph to avoid modifying the original
        self.original_data = self.dataset[self.current_graph_idx].clone()
        self.current_data = self.original_data.clone().to(self.device)
        self.current_graph_idx += 1
        
        # Reset episode state
        self.step_count = 0
        self.candidates = self.edge_operator.generate_candidates(
            self.current_data, max_candidates=self.max_candidates
        )
        
        # Calculate initial performance
        with torch.no_grad():
            initial_pred = self.gnn_model(
                self.current_data.x, self.current_data.edge_index, self.current_data.batch
            )
            self.initial_mse = F.mse_loss(initial_pred, self.current_data.y).item()
            self.best_mse = self.initial_mse
        
        self.history = []
        
    def _compute_reward(self, current_mse):
        """
        Compute reward based on improvement in MSE.
        Can use different reward shaping strategies.
        """
        if not self.reward_shaping:
            # Simple reward: negative MSE
            return -current_mse
        
        # Shaped reward based on improvement from previous best
        improvement = self.best_mse - current_mse
        
        # Update best MSE if improved
        if current_mse < self.best_mse:
            self.best_mse = current_mse
        
        # Reward consists of:
        # 1. Base reward for improvement over previous best
        # 2. Bonus for absolute improvement over initial MSE
        # 3. Small penalty for each step to encourage efficiency
        
        base_reward = np.clip(improvement * 10, -2, 2)  # Scale improvement, limit negative rewards
        
        # Bonus for beating initial MSE (relative improvement)
        relative_improvement = (self.initial_mse - current_mse) / self.initial_mse
        improvement_bonus = np.clip(relative_improvement * 5, 0, 1) 
        
        # Small penalty per step
        step_penalty = -0.01
        
        total_reward = base_reward + improvement_bonus + step_penalty
        
        return total_reward
        
    def reset(self):
        """
        Reset the environment at the end of an episode and return the initial observation.
        """
        self._setup_new_episode()
        
        # Extract observation
        with torch.no_grad():
            obs = self.obs_extractor(
                self.current_data.x, 
                self.current_data.edge_index,
                self.current_data.batch
            ).cpu().numpy().astype(np.float32)
            
        return obs
    
    def step(self, action):
        """
        Take a step in the environment by applying the chosen edge operation.
        """
        # Validate action
        if action < 0 or action >= len(self.candidates):
            # Invalid action
            reward = -1.0
            obs = self._build_observation()
            return obs, reward, True, {"error": "Invalid action"}
        
        # Apply the operation
        operation, edge_data = self.candidates[action]
        self.current_data = self.edge_operator.apply_operation(
            self.current_data, operation, edge_data
        )
        
        # Update step count and possibly generate new candidates
        self.step_count += 1
        
        # Generate new candidates based on the modified graph
        self.candidates = self.edge_operator.generate_candidates(
            self.current_data, max_candidates=self.max_candidates
        )
        
        # Optionally fine-tune GNN based on the modified graph
        if self.config.fine_tune_steps > 0 and not self.config.freeze_gnn:
            optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=self.config.lr_gnn)
            
            for _ in range(self.config.fine_tune_steps):
                optimizer.zero_grad()
                pred = self.gnn_model(
                    self.current_data.x, 
                    self.current_data.edge_index,
                    self.current_data.batch
                )
                loss = F.mse_loss(pred, self.current_data.y)
                loss.backward()
                optimizer.step()
        
        # Evaluate performance
        with torch.no_grad():
            pred = self.gnn_model(
                self.current_data.x, 
                self.current_data.edge_index,
                self.current_data.batch
            )
            current_mse = F.mse_loss(pred, self.current_data.y).item()
        
        # Calculate reward
        reward = self._compute_reward(current_mse)
        
        # Check if episode should end
        done = (self.step_count >= self.config.max_steps_per_episode)
        
        # Track history
        self.history.append({
            "step": self.step_count,
            "operation": operation.name,
            "edge_data": edge_data,
            "mse": current_mse,
            "reward": reward
        })
        
        # Extract observation
        obs = self._build_observation()
        
        # Info dictionary
        info = {
            "mse": current_mse,
            "initial_mse": self.initial_mse,
            "best_mse": self.best_mse,
            "improvement": self.initial_mse - current_mse,
            "relative_improvement": (self.initial_mse - current_mse) / self.initial_mse,
            "operation": operation.name,
            "step": self.step_count
        }
        
        return obs, reward, done, info
    
    def _build_observation(self):
        """
        Build the observation vector using the observation extractor.
        """
        with torch.no_grad():
            obs = self.obs_extractor(
                self.current_data.x, 
                self.current_data.edge_index,
                self.current_data.batch
            ).cpu().numpy().astype(np.float32)
            
        return obs
        
    def render(self, mode='human'):
        """
        Render the current state of the environment.
        """
        if mode == 'human':
            if len(self.history) > 0:
                last_step = self.history[-1]
                print(f"Step {last_step['step']}: {last_step['operation']} edge {last_step['edge_data']}")
                print(f"MSE: {last_step['mse']:.4f}, Reward: {last_step['reward']:.4f}")
        
        elif mode == 'rgb_array':
            # For advanced visualization you could implement:
            # 1. Networkx visualization of the graph
            # 2. Highlighting modified edges
            # 3. Performance plots
            # But this requires additional code not included here
            pass