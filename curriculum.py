"""
Curriculum learning to gradually increase difficulty of graph rewiring.
"""

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
            self.