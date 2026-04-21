"""
AutoML Problem Solver - Text/NLP Processor
Auto-detects text columns and extracts NLP features.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter


def detect_text_columns(df, min_avg_length=20, min_unique_ratio=0.3):
    """
    Detect columns that contain natural language text.
    
    Returns:
        list: Column names identified as text columns
    """
    text_cols = []
    
    for col in df.select_dtypes(include=['object']).columns:
        values = df[col].dropna().astype(str)
        
        if len(values) == 0:
            continue
        
        avg_length = values.str.len().mean()
        unique_ratio = values.nunique() / len(values)
        avg_words = values.str.split().str.len().mean()
        has_spaces = (values.str.contains(r'\s', regex=True)).mean()
        
        # Text column heuristics:
        # - Average length > threshold
        # - High unique ratio (not categorical)
        # - Average word count > 3
        # - Most values contain spaces
        if (avg_length > min_avg_length and unique_ratio > min_unique_ratio 
                and avg_words > 3 and has_spaces > 0.5):
            text_cols.append(col)
        elif avg_length > 50 and avg_words > 5:
            # Long text is likely NLP even with lower unique ratio
            text_cols.append(col)
    
    return text_cols


def process_text_columns(df, text_columns, max_tfidf_features=50):
    """
    Process text columns and extract features.
    
    Returns:
        tuple: (df_with_features, text_report)
    """
    report = {
        'text_columns_processed': [],
        'features_added': 0,
        'methods_used': [],
    }
    
    if not text_columns:
        return df, report
    
    new_features = []
    
    for col in text_columns:
        col_report = {'column': col, 'features': []}
        
        text_data = df[col].fillna('').astype(str)
        
        # 1. Statistical text features
        stat_features = _extract_statistical_features(text_data, col)
        new_features.append(stat_features)
        col_report['features'].append(f'{len(stat_features.columns)} statistical features')
        
        # 2. Sentiment features (basic rule-based)
        sentiment_features = _extract_sentiment(text_data, col)
        if sentiment_features is not None:
            new_features.append(sentiment_features)
            col_report['features'].append('sentiment scores')
        
        # 3. TF-IDF features
        tfidf_features = _extract_tfidf(text_data, col, max_features=max_tfidf_features)
        if tfidf_features is not None:
            new_features.append(tfidf_features)
            col_report['features'].append(f'{tfidf_features.shape[1]} TF-IDF features')
        
        report['text_columns_processed'].append(col_report)
    
    # Combine all new features
    if new_features:
        all_new = pd.concat(new_features, axis=1)
        report['features_added'] = all_new.shape[1]
        
        # Drop original text columns and add new features
        df = df.drop(columns=text_columns)
        df = pd.concat([df, all_new], axis=1)
        
        report['methods_used'] = ['Statistical Features', 'Sentiment Analysis', 'TF-IDF']
    
    return df, report


def _extract_statistical_features(text_series, col_name):
    """Extract statistical text features."""
    features = pd.DataFrame(index=text_series.index)
    
    # Length features
    features[f'{col_name}_char_count'] = text_series.str.len()
    features[f'{col_name}_word_count'] = text_series.str.split().str.len()
    features[f'{col_name}_avg_word_length'] = (
        features[f'{col_name}_char_count'] / features[f'{col_name}_word_count'].replace(0, 1)
    )
    
    # Sentence count (approximate)
    features[f'{col_name}_sentence_count'] = text_series.str.count(r'[.!?]+')
    
    # Special character features
    features[f'{col_name}_uppercase_ratio'] = (
        text_series.str.count(r'[A-Z]') / features[f'{col_name}_char_count'].replace(0, 1)
    )
    features[f'{col_name}_digit_ratio'] = (
        text_series.str.count(r'\d') / features[f'{col_name}_char_count'].replace(0, 1)
    )
    features[f'{col_name}_special_char_count'] = text_series.str.count(r'[^a-zA-Z0-9\s]')
    
    # Punctuation features
    features[f'{col_name}_exclamation_count'] = text_series.str.count('!')
    features[f'{col_name}_question_count'] = text_series.str.count(r'\?')
    
    # Unique word ratio
    features[f'{col_name}_unique_word_ratio'] = text_series.apply(
        lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
    )
    
    features = features.fillna(0)
    return features


def _extract_sentiment(text_series, col_name):
    """Basic rule-based sentiment analysis."""
    # Positive and negative word lists (basic)
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'best', 'perfect', 'awesome', 'happy', 'beautiful',
        'outstanding', 'brilliant', 'impressive', 'superb', 'nice',
        'like', 'enjoy', 'recommend', 'pleased', 'satisfied', 'positive',
        'helpful', 'incredible', 'remarkable', 'delightful', 'fabulous',
    }
    
    negative_words = {
        'bad', 'terrible', 'horrible', 'awful', 'worst', 'poor', 'hate',
        'disappointing', 'disappointed', 'negative', 'ugly', 'boring',
        'waste', 'useless', 'pathetic', 'annoying', 'frustrating',
        'broken', 'failed', 'failure', 'problem', 'issue', 'complaint',
        'unfortunately', 'regret', 'terrible', 'dreadful', 'unpleasant',
    }
    
    try:
        features = pd.DataFrame(index=text_series.index)
        
        def _score(text):
            words = text.lower().split()
            pos = sum(1 for w in words if w in positive_words)
            neg = sum(1 for w in words if w in negative_words)
            total = max(len(words), 1)
            return pos / total, neg / total, (pos - neg) / total
        
        scores = text_series.apply(_score)
        features[f'{col_name}_positive_ratio'] = scores.apply(lambda x: x[0])
        features[f'{col_name}_negative_ratio'] = scores.apply(lambda x: x[1])
        features[f'{col_name}_sentiment_score'] = scores.apply(lambda x: x[2])
        
        return features
    except Exception:
        return None


def _extract_tfidf(text_series, col_name, max_features=50):
    """Extract TF-IDF features from text."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Clean text
        cleaned = text_series.str.lower().str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
        cleaned = cleaned.str.strip()
        cleaned = cleaned.replace('', 'empty')
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        
        tfidf_matrix = vectorizer.fit_transform(cleaned)
        feature_names = [f'{col_name}_tfidf_{name}' for name in vectorizer.get_feature_names_out()]
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=text_series.index
        )
        
        return tfidf_df
    except Exception:
        return None
