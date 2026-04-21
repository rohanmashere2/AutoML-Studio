"""
Topic Modeling Engine — LDA and NMF for unsupervised text analysis.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


def discover_topics(texts, n_topics=5, method='both', max_features=1000):
    """Run topic modeling on text data."""
    if isinstance(texts, pd.Series):
        texts = texts.dropna().astype(str).tolist()
    
    texts = [t for t in texts if len(str(t).strip()) > 5]
    
    if len(texts) < 10:
        return {'error': 'Not enough text documents (need at least 10)'}
    
    n_topics = min(n_topics, len(texts) // 3, 15)
    if n_topics < 2:
        n_topics = 2
    
    results = {}
    
    # LDA
    if method in ('lda', 'both'):
        try:
            count_vec = CountVectorizer(max_features=max_features, stop_words='english',
                                         max_df=0.95, min_df=max(2, len(texts) // 100))
            count_matrix = count_vec.fit_transform(texts)
            feature_names = count_vec.get_feature_names_out()
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30)
            doc_topics = lda.fit_transform(count_matrix)
            
            topics = []
            for i, topic_weights in enumerate(lda.components_):
                top_idx = topic_weights.argsort()[-10:][::-1]
                top_words = [{'word': feature_names[j], 'weight': round(float(topic_weights[j]), 4)} for j in top_idx]
                topic_name = f"Topic {i+1}: {', '.join(w['word'] for w in top_words[:3])}"
                topics.append({'id': i, 'name': topic_name, 'words': top_words, 'prevalence': round(float(doc_topics[:, i].mean()), 4)})
            
            results['LDA'] = {
                'topics': topics,
                'n_topics': n_topics,
                'perplexity': round(float(lda.perplexity(count_matrix)), 2),
                'doc_topic_dist': doc_topics[:min(100, len(texts))].tolist(),
                'dominant_topic_per_doc': doc_topics.argmax(axis=1).tolist()
            }
        except Exception as e:
            results['LDA'] = {'error': str(e)}
    
    # NMF
    if method in ('nmf', 'both'):
        try:
            tfidf_vec = TfidfVectorizer(max_features=max_features, stop_words='english',
                                         max_df=0.95, min_df=max(2, len(texts) // 100))
            tfidf_matrix = tfidf_vec.fit_transform(texts)
            feature_names = tfidf_vec.get_feature_names_out()
            
            nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
            doc_topics = nmf.fit_transform(tfidf_matrix)
            
            topics = []
            for i, topic_weights in enumerate(nmf.components_):
                top_idx = topic_weights.argsort()[-10:][::-1]
                top_words = [{'word': feature_names[j], 'weight': round(float(topic_weights[j]), 4)} for j in top_idx]
                topic_name = f"Topic {i+1}: {', '.join(w['word'] for w in top_words[:3])}"
                topics.append({'id': i, 'name': topic_name, 'words': top_words, 'prevalence': round(float(doc_topics[:, i].mean()), 4)})
            
            reconstruction_err = round(float(nmf.reconstruction_err_), 2)
            results['NMF'] = {
                'topics': topics,
                'n_topics': n_topics,
                'reconstruction_error': reconstruction_err,
                'doc_topic_dist': doc_topics[:min(100, len(texts))].tolist(),
                'dominant_topic_per_doc': doc_topics.argmax(axis=1).tolist()
            }
        except Exception as e:
            results['NMF'] = {'error': str(e)}
    
    # Pick best method
    best = 'LDA' if 'LDA' in results and 'error' not in results.get('LDA', {}) else 'NMF'
    
    return {
        'best_method': best,
        'results': results,
        'n_documents': len(texts),
        'n_topics': n_topics,
    }
