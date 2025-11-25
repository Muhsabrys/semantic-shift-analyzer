"""
Enhanced data loading with precomputed embeddings support
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


PRECOMPUTED_URL = "https://github.com/Muhsabrys/semantic-shift-analyzer/raw/main/precomputed_embeddings.npz"


@st.cache_data
def download_precomputed_embeddings():
    """Download precomputed embeddings from GitHub"""
    try:
        response = requests.get(PRECOMPUTED_URL, timeout=30)
        response.raise_for_status()
        
        # Load from bytes
        npz_data = np.load(BytesIO(response.content), allow_pickle=True)
        
        return npz_data
    except Exception as e:
        st.error(f"❌ Failed to download precomputed embeddings: {str(e)}")
        return None


@st.cache_data
def load_precomputed_corpus():
    """
    Load precomputed State of the Union embeddings
    Returns: models dict, years list, global_vocab set, metadata dict
    """
    with st.spinner("Downloading State of the Union embeddings..."):
        npz_data = download_precomputed_embeddings()
    
    if npz_data is None:
        return None, None, None, None
    
    try:
        # Extract data from npz
        years = npz_data['years']
        embeddings_dict = npz_data['embeddings'].item()  # Dictionary of year -> embeddings
        vocabulary = npz_data['vocabulary']  # Global vocabulary
        metadata = npz_data['metadata'].item() if 'metadata' in npz_data else {}
        
        # Convert embeddings to Word2Vec models
        models = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, year in enumerate(years):
            status_text.text(f"Loading model for year {year}...")
            progress_bar.progress((idx + 1) / len(years))
            
            # Get embeddings for this year
            year_embeddings = embeddings_dict[year]
            
            # Create a mock Word2Vec model with these embeddings
            # We'll create KeyedVectors directly
            kv = KeyedVectors(vector_size=year_embeddings.shape[1])
            
            # Add vectors
            kv.add_vectors(vocabulary, year_embeddings)
            
            # Create a minimal Word2Vec wrapper
            model = Word2Vec(vector_size=year_embeddings.shape[1], min_count=1)
            model.wv = kv
            
            models[int(year)] = model
        
        progress_bar.empty()
        status_text.empty()
        
        # Convert vocabulary to set
        global_vocab = set(vocabulary)
        
        # Create metadata summary
        if not metadata:
            metadata = {
                'source': 'State of the Union Addresses',
                'description': 'Precomputed embeddings from historical State of the Union speeches',
                'years': f"{min(years)}-{max(years)}",
                'vocabulary_size': len(vocabulary),
                'embedding_dimension': year_embeddings.shape[1]
            }
        
        st.success(f"✅ Loaded {len(years)} years of State of the Union embeddings!")
        
        return models, sorted([int(y) for y in years]), global_vocab, metadata
        
    except Exception as e:
        st.error(f"❌ Error processing embeddings: {str(e)}")
        return None, None, None, None


@st.cache_data 
def get_precomputed_word_stats(global_vocab, models):
    """
    Generate word statistics for precomputed embeddings
    """
    word_to_years = {}
    word_to_total_count = {}
    
    for year, model in models.items():
        for word in model.wv.index_to_key:
            if word not in word_to_years:
                word_to_years[word] = set()
            word_to_years[word].add(year)
            
            # For precomputed, we don't have actual counts, so estimate
            if word not in word_to_total_count:
                word_to_total_count[word] = 0
            word_to_total_count[word] += 10  # Placeholder value
    
    return word_to_years, word_to_total_count
