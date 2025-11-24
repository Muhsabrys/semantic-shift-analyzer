"""
Word2Vec model training and alignment functions
"""

import streamlit as st
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine, euclidean


@st.cache_resource
def train_stable_models(_year_to_tokens, global_vocab, n_seeds=5, vector_size=200, 
                        window=5, min_count=1):
    """
    Train multiple Word2Vec models per year with different seeds
    and average their embeddings for stability.
    """
    models = {}
    years = sorted(_year_to_tokens.keys())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_iterations = len(years) * n_seeds
    current_iteration = 0
    
    for yr in years:
        tokens = _year_to_tokens[yr]
        
        # Filter tokens to global vocabulary
        filtered_tokens = [t for t in tokens if t in global_vocab]
        
        if len(filtered_tokens) < 30:
            st.warning(f"⚠️ Year {yr} has insufficient vocabulary ({len(filtered_tokens)} tokens)")
            continue
        
        # Train multiple models with different seeds
        all_embeddings = {}
        
        for seed_idx in range(n_seeds):
            status_text.text(f"Training year {yr} (seed {seed_idx+1}/{n_seeds})...")
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
            
            model = Word2Vec(
                sentences=[filtered_tokens],
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                sg=1,
                workers=1,
                seed=seed_idx,
                epochs=20,
                negative=10
            )
            
            # Collect embeddings
            for word in model.wv.index_to_key:
                if word not in all_embeddings:
                    all_embeddings[word] = []
                all_embeddings[word].append(model.wv[word])
        
        # Average embeddings across seeds
        averaged_model = Word2Vec(
            sentences=[filtered_tokens],
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1,
            workers=1,
            seed=0
        )
        
        # Replace with averaged embeddings
        for word in all_embeddings:
            if word in averaged_model.wv:
                averaged_model.wv[word] = np.mean(all_embeddings[word], axis=0)
        
        models[yr] = averaged_model
    
    progress_bar.empty()
    status_text.empty()
    
    return models


def align_embeddings_robust(base_model, other_model, global_vocab):
    """Robust Procrustes alignment using global vocabulary"""
    shared = list(global_vocab & 
                  set(base_model.wv.index_to_key) & 
                  set(other_model.wv.index_to_key))
    
    if len(shared) < 20:
        return None, None, len(shared)
    
    # Get embeddings for shared vocabulary
    base_vectors = np.array([base_model.wv[w] for w in shared])
    other_vectors = np.array([other_model.wv[w] for w in shared])
    
    # Normalize
    base_vectors = normalize(base_vectors, norm='l2')
    other_vectors = normalize(other_vectors, norm='l2')
    
    # Procrustes alignment
    R, _ = orthogonal_procrustes(other_vectors, base_vectors)
    aligned_vectors = other_vectors @ R
    
    return base_vectors, aligned_vectors, len(shared)


def compute_drift_score_robust(word, base_year, target_year, models, global_vocab, metric='cosine'):
    """Compute semantic drift with better error handling and vocabulary checking"""
    if base_year not in models or target_year not in models:
        return None
    
    base_model = models[base_year]
    target_model = models[target_year]
    
    # Check if word exists in both models
    if word not in base_model.wv or word not in target_model.wv:
        return None
    
    # Check if word is in global vocabulary
    if word not in global_vocab:
        return None
    
    # Align embeddings
    base_vecs, aligned_vecs, n_shared = align_embeddings_robust(
        base_model, target_model, global_vocab
    )
    
    if base_vecs is None or n_shared < 20:
        return None
    
    # Get word vectors
    try:
        base_vec = base_model.wv[word].reshape(1, -1)
        base_vec = normalize(base_vec, norm='l2')[0]
        
        target_vec = target_model.wv[word].reshape(1, -1)
        target_vec = normalize(target_vec, norm='l2')[0]
        
        # Apply alignment transformation
        shared_words = list(global_vocab & 
                          set(base_model.wv.index_to_key) & 
                          set(target_model.wv.index_to_key))
        
        base_shared = np.array([base_model.wv[w] for w in shared_words])
        target_shared = np.array([target_model.wv[w] for w in shared_words])
        
        base_shared = normalize(base_shared, norm='l2')
        target_shared = normalize(target_shared, norm='l2')
        
        R, _ = orthogonal_procrustes(target_shared, base_shared)
        target_vec_aligned = target_vec @ R
        
        # Compute distance
        if metric == 'cosine':
            return cosine(base_vec, target_vec_aligned)
        else:
            return euclidean(base_vec, target_vec_aligned)
    except:
        return None
