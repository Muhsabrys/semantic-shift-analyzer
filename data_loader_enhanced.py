import streamlit as st
import numpy as np
import requests
from io import BytesIO
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

PRECOMPUTED_URL = "https://github.com/Muhsabrys/semantic-shift-analyzer/raw/main/state_of_union_embeddings.npz"

@st.cache_data
def load_precomputed_corpus():
    """Load precomputed State of the Union embeddings"""
    try:
        with st.spinner("Downloading embeddings..."):
            response = requests.get(PRECOMPUTED_URL, timeout=60)
            response.raise_for_status()
        
        with np.load(BytesIO(response.content), allow_pickle=True) as npz_data:
            embeddings_array = npz_data['embeddings']  # (n_years, vocab_size, embedding_dim)
            vocabulary = npz_data['vocabulary'].tolist()
            years = npz_data['years'].tolist()
            metadata = npz_data['metadata'].item() if 'metadata' in npz_data else {}
        
        # Convert to Word2Vec models
        models = {}
        progress_bar = st.progress(0)
        
        for idx, year in enumerate(years):
            progress_bar.progress((idx + 1) / len(years))
            
            # Get embeddings for this year
            year_embeddings = embeddings_array[idx]  # (vocab_size, embedding_dim)
            embedding_dim = year_embeddings.shape[1]
            
            # Create KeyedVectors
            kv = KeyedVectors(vector_size=embedding_dim)
            kv.add_vectors(vocabulary, year_embeddings)
            
            # Wrap in Word2Vec
            model = Word2Vec(vector_size=embedding_dim, min_count=1)
            model.wv = kv
            models[int(year)] = model
        
        progress_bar.empty()
        
        metadata_out = {
            'source': 'State of the Union Addresses',
            'vocabulary_size': len(vocabulary),
            'embedding_dimension': embedding_dim,
            'years': f"{min(years)}-{max(years)}"
        }
        
        st.success(f"✅ Loaded {len(years)} years ({min(years)}-{max(years)})!")
        return models, sorted([int(y) for y in years]), set(vocabulary), metadata_out
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
