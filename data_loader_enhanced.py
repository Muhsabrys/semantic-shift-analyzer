"""
Enhanced data loading with precomputed embeddings support
"""

import streamlit as st
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
        response = requests.get(PRECOMPUTED_URL, timeout=60)
        response.raise_for_status()
        npz_data = np.load(BytesIO(response.content), allow_pickle=True)
        return npz_data
    except Exception as e:
        st.error(f"❌ Failed to download: {str(e)}")
        return None


@st.cache_data
def load_precomputed_corpus():
    """Load precomputed State of the Union embeddings"""
    with st.spinner("Downloading embeddings..."):
        npz_data = download_precomputed_embeddings()
    
    if npz_data is None:
        return None, None, None, None
    
    try:
        years = npz_data['years']
        embeddings_dict = npz_data['embeddings'].item()
        vocabulary = npz_data['vocabulary']
        metadata = npz_data['metadata'].item() if 'metadata' in npz_data else {}
        
        models = {}
        progress_bar = st.progress(0)
        
        for idx, year in enumerate(years):
            progress_bar.progress((idx + 1) / len(years))
            year_embeddings = embeddings_dict[year]
            
            kv = KeyedVectors(vector_size=year_embeddings.shape[1])
            kv.add_vectors(vocabulary, year_embeddings)
            
            model = Word2Vec(vector_size=year_embeddings.shape[1], min_count=1)
            model.wv = kv
            models[int(year)] = model
        
        progress_bar.empty()
        
        global_vocab = set(vocabulary)
        
        if not metadata:
            metadata = {
                'source': 'State of the Union Addresses',
                'description': 'Precomputed embeddings',
                'years': f"{min(years)}-{max(years)}",
                'vocabulary_size': len(vocabulary),
                'embedding_dimension': year_embeddings.shape[1]
            }
        
        st.success(f"✅ Loaded {len(years)} years!")
        return models, sorted([int(y) for y in years]), global_vocab, metadata
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None, None


@st.cache_data 
def get_precomputed_word_stats(global_vocab, models):
    """Generate word statistics for precomputed embeddings"""
    word_to_years = {}
    word_to_total_count = {}
    
    for year, model in models.items():
        for word in model.wv.index_to_key:
            if word not in word_to_years:
                word_to_years[word] = set()
            word_to_years[word].add(year)
            
            if word not in word_to_total_count:
                word_to_total_count[word] = 0
            word_to_total_count[word] += 10
    
    return word_to_years, word_to_total_count
