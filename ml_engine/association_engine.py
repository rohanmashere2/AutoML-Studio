"""
Association Rule Mining Engine — Apriori-based frequent itemset and rule discovery.
"""

import numpy as np
import pandas as pd
from itertools import combinations


def mine_association_rules(df, min_support=0.05, min_confidence=0.5, max_items=3):
    """Mine association rules from transactional/binary data."""
    # Convert to binary matrix if needed
    binary_df = _to_binary_matrix(df)
    
    if binary_df is None or binary_df.shape[1] < 2:
        return {'error': 'Data not suitable for association rule mining', 'rules': []}
    
    n_transactions = len(binary_df)
    items = binary_df.columns.tolist()
    
    # Find frequent itemsets using Apriori
    frequent_itemsets = []
    
    # 1-itemsets
    for item in items:
        support = binary_df[item].mean()
        if support >= min_support:
            frequent_itemsets.append({
                'itemset': frozenset([item]),
                'support': support
            })
    
    frequent_items_1 = [fs['itemset'] for fs in frequent_itemsets]
    
    # 2-itemsets and 3-itemsets
    for k in range(2, max_items + 1):
        base_items = set()
        for fs in frequent_itemsets:
            if len(fs['itemset']) == k - 1:
                base_items.update(fs['itemset'])
        
        if len(base_items) < k:
            break
        
        for combo in combinations(sorted(base_items), k):
            itemset = frozenset(combo)
            mask = binary_df[list(combo)].all(axis=1)
            support = mask.mean()
            if support >= min_support:
                frequent_itemsets.append({
                    'itemset': itemset,
                    'support': support
                })
    
    # Generate rules from frequent itemsets
    rules = []
    for fs in frequent_itemsets:
        if len(fs['itemset']) < 2:
            continue
        
        items_list = list(fs['itemset'])
        for i in range(len(items_list)):
            consequent = frozenset([items_list[i]])
            antecedent = fs['itemset'] - consequent
            
            # Find antecedent support
            ant_support = _get_support(binary_df, antecedent)
            if ant_support == 0:
                continue
            
            confidence = fs['support'] / ant_support
            if confidence < min_confidence:
                continue
            
            cons_support = _get_support(binary_df, consequent)
            lift = confidence / cons_support if cons_support > 0 else 0
            leverage = fs['support'] - (ant_support * cons_support)
            conviction = (1 - cons_support) / (1 - confidence) if confidence < 1 else float('inf')
            
            rules.append({
                'antecedent': sorted(list(antecedent)),
                'consequent': sorted(list(consequent)),
                'support': round(fs['support'], 4),
                'confidence': round(confidence, 4),
                'lift': round(lift, 4),
                'leverage': round(leverage, 6),
                'conviction': round(min(conviction, 999), 4),
                'antecedent_support': round(ant_support, 4),
                'consequent_support': round(cons_support, 4),
            })
    
    # Sort by lift
    rules.sort(key=lambda r: r['lift'], reverse=True)
    
    return {
        'n_rules': len(rules),
        'n_frequent_itemsets': len(frequent_itemsets),
        'n_transactions': n_transactions,
        'n_items': len(items),
        'rules': rules[:100],
        'top_items': [{'item': list(fs['itemset'])[0], 'support': round(fs['support'], 4)} 
                      for fs in sorted(frequent_itemsets, key=lambda x: x['support'], reverse=True) 
                      if len(fs['itemset']) == 1][:20],
        'min_support': min_support,
        'min_confidence': min_confidence,
    }


def _to_binary_matrix(df):
    """Convert DataFrame to binary transaction matrix."""
    binary_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, True, False, 'true', 'false', 'yes', 'no'}):
            binary_cols.append(col)
        elif df[col].nunique() <= 2:
            binary_cols.append(col)
    
    if len(binary_cols) >= 2:
        result = df[binary_cols].copy()
        for col in result.columns:
            result[col] = result[col].apply(lambda x: 1 if x in [1, True, 'true', 'yes', 'True', 'Yes'] else 0)
        return result
    
    # Try one-hot encoding categorical columns  
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) >= 1:
        dummy = pd.get_dummies(df[cat_cols], prefix_sep='=')
        return dummy
    
    return None


def _get_support(binary_df, itemset):
    """Get support of an itemset in the binary matrix."""
    items = list(itemset)
    if len(items) == 1:
        return binary_df[items[0]].mean()
    return binary_df[items].all(axis=1).mean()
