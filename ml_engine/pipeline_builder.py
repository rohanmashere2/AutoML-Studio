"""
Visual Pipeline Builder — DAG-based pipeline block execution engine.
"""

import time
import numpy as np
import pandas as pd
from sklearn.base import clone


# ── Block Registry ──────────────────────────────────────

BLOCK_REGISTRY = {
    'load_csv': {
        'name': 'Load CSV', 'category': 'Data', 'icon': '📂',
        'inputs': [], 'outputs': ['data'],
        'params': {'file_path': 'str'}
    },
    'missing_values': {
        'name': 'Handle Missing', 'category': 'Clean', 'icon': '🩹',
        'inputs': ['data'], 'outputs': ['data'],
        'params': {'strategy': ['mean', 'median', 'mode', 'drop']}
    },
    'encoding': {
        'name': 'Encode Categories', 'category': 'Transform', 'icon': '🔤',
        'inputs': ['data'], 'outputs': ['data'],
        'params': {'method': ['label', 'onehot', 'ordinal']}
    },
    'scaling': {
        'name': 'Scale Features', 'category': 'Transform', 'icon': '📏',
        'inputs': ['data'], 'outputs': ['data'],
        'params': {'method': ['standard', 'minmax', 'robust']}
    },
    'pca': {
        'name': 'PCA Reduction', 'category': 'Transform', 'icon': '🔬',
        'inputs': ['data'], 'outputs': ['data'],
        'params': {'n_components': 'int', 'variance_threshold': 'float'}
    },
    'feature_selection': {
        'name': 'Feature Selection', 'category': 'Transform', 'icon': '🎯',
        'inputs': ['data'], 'outputs': ['data'],
        'params': {'method': ['variance', 'correlation', 'kbest'], 'k': 'int'}
    },
    'train_model': {
        'name': 'Train Model', 'category': 'Model', 'icon': '🤖',
        'inputs': ['data'], 'outputs': ['model', 'metrics'],
        'params': {'algorithm': ['random_forest', 'xgboost', 'logistic', 'svm']}
    },
    'evaluate': {
        'name': 'Evaluate', 'category': 'Evaluate', 'icon': '📊',
        'inputs': ['model', 'data'], 'outputs': ['metrics'],
        'params': {'cv_folds': 'int'}
    },
    'explain': {
        'name': 'SHAP Explain', 'category': 'Evaluate', 'icon': '🔍',
        'inputs': ['model', 'data'], 'outputs': ['explanation'],
        'params': {}
    },
    'export': {
        'name': 'Export Model', 'category': 'Deploy', 'icon': '📦',
        'inputs': ['model'], 'outputs': ['file'],
        'params': {'format': ['pickle', 'onnx', 'docker']}
    },
    'cluster': {
        'name': 'Clustering', 'category': 'Unsupervised', 'icon': '🔮',
        'inputs': ['data'], 'outputs': ['labels'],
        'params': {'algorithm': ['kmeans', 'dbscan', 'hierarchical'], 'n_clusters': 'int'}
    },
    'anomaly': {
        'name': 'Anomaly Detection', 'category': 'Unsupervised', 'icon': '🚨',
        'inputs': ['data'], 'outputs': ['labels'],
        'params': {'contamination': 'float'}
    },
}


class PipelineDAG:
    """Execute a user-defined pipeline of blocks as a DAG."""
    
    def __init__(self):
        self.blocks = []
        self.connections = []
        self.execution_log = []
    
    def add_block(self, block_id, block_type, params=None, position=None):
        """Add a block to the pipeline."""
        if block_type not in BLOCK_REGISTRY:
            return {'error': f'Unknown block type: {block_type}'}
        
        self.blocks.append({
            'id': block_id,
            'type': block_type,
            'params': params or {},
            'position': position or {'x': 0, 'y': 0},
            'metadata': BLOCK_REGISTRY[block_type],
        })
        return {'success': True, 'block_id': block_id}
    
    def connect(self, source_id, target_id, port='data'):
        """Connect two blocks."""
        self.connections.append({
            'source': source_id,
            'target': target_id,
            'port': port,
        })
        return {'success': True}
    
    def validate(self):
        """Validate the pipeline DAG."""
        issues = []
        block_ids = {b['id'] for b in self.blocks}
        
        for conn in self.connections:
            if conn['source'] not in block_ids:
                issues.append(f"Connection source '{conn['source']}' not found")
            if conn['target'] not in block_ids:
                issues.append(f"Connection target '{conn['target']}' not found")
        
        # Check for cycles
        adj = {b['id']: [] for b in self.blocks}
        for conn in self.connections:
            adj[conn['source']].append(conn['target'])
        
        visited = set()
        rec_stack = set()
        for node in adj:
            if self._has_cycle(node, visited, rec_stack, adj):
                issues.append('Pipeline contains a cycle')
                break
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def get_execution_order(self):
        """Topological sort for block execution order."""
        adj = {b['id']: [] for b in self.blocks}
        in_degree = {b['id']: 0 for b in self.blocks}
        
        for conn in self.connections:
            adj[conn['source']].append(conn['target'])
            in_degree[conn['target']] += 1
        
        queue = [n for n in in_degree if in_degree[n] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return order
    
    def export_config(self):
        """Export pipeline configuration as JSON."""
        return {
            'blocks': self.blocks,
            'connections': self.connections,
            'n_blocks': len(self.blocks),
            'n_connections': len(self.connections),
        }
    
    def get_block_registry(self):
        """Return available blocks grouped by category."""
        categories = {}
        for block_type, meta in BLOCK_REGISTRY.items():
            cat = meta['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                'type': block_type,
                'name': meta['name'],
                'icon': meta['icon'],
                'params': meta['params'],
            })
        return categories
    
    def _has_cycle(self, node, visited, rec_stack, adj):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if self._has_cycle(neighbor, visited, rec_stack, adj):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.discard(node)
        return False
