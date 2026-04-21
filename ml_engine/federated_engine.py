"""
Federated Learning Engine — Privacy-preserving distributed model training.
"""

import numpy as np
import copy
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score


class FederatedServer:
    """Central server that aggregates model updates from clients."""
    
    def __init__(self, model, problem_type='classification'):
        self.global_model = model
        self.problem_type = problem_type
        self.round_history = []
    
    def run_federated_training(self, client_datasets, n_rounds=5):
        """Run federated averaging across client datasets."""
        for round_num in range(n_rounds):
            client_updates = []
            
            for client_id, (X_client, y_client) in enumerate(client_datasets):
                client = FederatedClient(client_id, clone(self.global_model))
                update = client.train_local(X_client, y_client)
                client_updates.append(update)
            
            # Aggregate updates (FedAvg)
            self._aggregate(client_updates)
            
            # Evaluate
            round_scores = []
            for client_id, (X_client, y_client) in enumerate(client_datasets):
                y_pred = self.global_model.predict(X_client)
                if self.problem_type == 'classification':
                    score = accuracy_score(y_client, y_pred)
                else:
                    score = r2_score(y_client, y_pred)
                round_scores.append(round(float(score), 4))
            
            self.round_history.append({
                'round': round_num + 1,
                'n_clients': len(client_datasets),
                'client_scores': round_scores,
                'avg_score': round(float(np.mean(round_scores)), 4),
            })
        
        return {
            'rounds': self.round_history,
            'final_score': self.round_history[-1]['avg_score'] if self.round_history else 0,
            'n_rounds': n_rounds,
            'n_clients': len(client_datasets),
            'improvement': round(
                self.round_history[-1]['avg_score'] - self.round_history[0]['avg_score'], 4
            ) if len(self.round_history) > 1 else 0,
        }
    
    def _aggregate(self, client_updates):
        """Federated Averaging — average model parameters."""
        if not client_updates:
            return
        
        # For sklearn models, average coefficients
        if hasattr(self.global_model, 'coef_'):
            avg_coef = np.mean([u['coef'] for u in client_updates], axis=0)
            avg_intercept = np.mean([u['intercept'] for u in client_updates], axis=0)
            self.global_model.coef_ = avg_coef
            self.global_model.intercept_ = avg_intercept


class FederatedClient:
    """Client that trains on local data and sends updates to server."""
    
    def __init__(self, client_id, model):
        self.client_id = client_id
        self.model = model
    
    def train_local(self, X, y):
        """Train on local data and return model updates (not raw data)."""
        self.model.fit(X, y)
        
        update = {'client_id': self.client_id, 'n_samples': len(y)}
        
        if hasattr(self.model, 'coef_'):
            update['coef'] = self.model.coef_.copy()
            update['intercept'] = self.model.intercept_.copy() if hasattr(self.model, 'intercept_') else 0
        
        return update


def simulate_federated(X, y, n_clients=3, n_rounds=5, problem_type='classification'):
    """Simulate federated learning by splitting data across clients."""
    n = len(X)
    indices = np.random.permutation(n)
    splits = np.array_split(indices, n_clients)
    
    client_datasets = []
    client_info = []
    for i, split in enumerate(splits):
        X_client = X[split] if isinstance(X, np.ndarray) else X.iloc[split]
        y_client = y[split] if isinstance(y, np.ndarray) else y.iloc[split]
        client_datasets.append((X_client, y_client))
        client_info.append({
            'client_id': i,
            'n_samples': len(split),
            'pct': round(len(split) / n * 100, 1)
        })
    
    if problem_type == 'classification':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = Ridge(alpha=1.0)
    
    server = FederatedServer(model, problem_type)
    results = server.run_federated_training(client_datasets, n_rounds)
    results['clients'] = client_info
    results['privacy_note'] = 'No raw data was shared between clients. Only model parameters were exchanged.'
    
    return results
