import streamlit as st
import numpy as np
import requests
from io import BytesIO
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

PRECOMPUTED_URL = "https://github.com/Muhsabrys/semantic-shift-analyzer/raw/main/precomputed_embeddings.npz"

@st.cache_data
def load_precomputed_corpus():
    """Load precomputed State of the Union embeddings"""
    try:
        with st.spinner("Downloading embeddings..."):
            response = requests.get(PRECOMPUTED_URL, timeout=60)
            response.raise_for_status()
        
        with np.load(BytesIO(response.content), allow_pickle=True) as npz_data:
            years = [int(y) for y in npz_data['years']]
            embeddings_raw = npz_data['embeddings'].item()
            vocabulary = list(npz_data['vocabulary'])
            metadata = dict(npz_data['metadata'].item()) if 'metadata' in npz_data else {}
        
        # Convert embeddings dict keys to regular ints
        embeddings_dict = {int(k): v for k, v in embeddings_raw.items()}
        
        models = {}
        progress_bar = st.progress(0)
        
        for idx, year in enumerate(years):
            progress_bar.progress((idx + 1) / len(years))
            year_embeddings = embeddings_dict[year]
            embedding_dim = year_embeddings.shape[1]
            
            kv = KeyedVectors(vector_size=embedding_dim)
            kv.add_vectors(vocabulary, year_embeddings)
            
            model = Word2Vec(vector_size=embedding_dim, min_count=1)
            model.wv = kv
            models[year] = model
        
        progress_bar.empty()
        
        if not metadata:
            metadata = {
                'source': 'State of the Union Addresses',
                'vocabulary_size': len(vocabulary),
                'embedding_dimension': embedding_dim
            }
        
        st.success(f"✅ Loaded {len(years)} years!")
        return models, sorted(years), set(vocabulary), metadata
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None, None

@st.cache_data 
def get_precomputed_word_stats(global_vocab, models):
    word_to_years = {}
    word_to_total_count = {}
    
    for year, model in models.items():
        for word in model.wv.index_to_key:
            if word not in word_to_years:
                word_to_years[word] = set()
            word_to_years[word].add(year)
            word_to_total_count[word] = word_to_total_count.get(word, 0) + 10
    
    return word_to_years, word_to_total_count
