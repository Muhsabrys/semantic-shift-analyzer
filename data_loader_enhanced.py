import streamlit as st
import numpy as np
import requests
import os
import joblib
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

PRECOMPUTED_URL = "https://github.com/Muhsabrys/semantic-shift-analyzer/raw/main/state_of_union_embeddings.npz"
CACHE_FILE = "state_of_union_models.joblib"

def _create_year_model(args):
    """Worker function to create a model for a single year (runs in parallel)"""
    year, vectors, vocab = args
    embedding_dim = vectors.shape[1]
    
    # Create KeyedVectors
    kv = KeyedVectors(vector_size=embedding_dim)
    kv.add_vectors(vocab, vectors)
    
    # Wrap in Word2Vec
    model = Word2Vec(vector_size=embedding_dim, min_count=1)
    model.wv = kv
    return int(year), model

@st.cache_data
def load_precomputed_corpus():
    """Load precomputed State of the Union embeddings with local caching"""
    # 1. Try loading from fast local cache first
    if os.path.exists(CACHE_FILE):
        try:
            # st.info("Loading from fast cache...")
            return joblib.load(CACHE_FILE)
        except Exception as e:
            st.warning(f"Cache corrupted, reloading from source: {e}")

    try:
        # 2. Load source .npz file
        data_source = None
        local_path = "state_of_union_embeddings.npz"
        
        if os.path.exists(local_path):
            data_source = local_path
        else:
            with st.spinner("Downloading embeddings..."):
                response = requests.get(PRECOMPUTED_URL, timeout=60)
                response.raise_for_status()
                data_source = BytesIO(response.content)
        
        with np.load(data_source, allow_pickle=True) as npz_data:
            embeddings_array = npz_data['embeddings']  # (n_years, vocab_size, embedding_dim)
            vocabulary = npz_data['vocabulary'].tolist()
            years = npz_data['years'].tolist()
            metadata = npz_data['metadata'].item() if 'metadata' in npz_data else {}
        
        # 3. Convert to Word2Vec models in parallel
        models = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Constructing models in parallel...")
        
        # Prepare arguments for parallel workers
        process_args = []
        for idx, year in enumerate(years):
            process_args.append((year, embeddings_array[idx], vocabulary))
            
        # Execute in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_create_year_model, arg) for arg in process_args]
            
            for i, future in enumerate(futures):
                year, model = future.result()
                models[year] = model
                progress_bar.progress((i + 1) / len(years))
        
        progress_bar.empty()
        status_text.empty()
        
        metadata_out = {
            'source': 'State of the Union Addresses',
            'vocabulary_size': len(vocabulary),
            'embedding_dimension': embeddings_array[0].shape[1],
            'years': f"{min(years)}-{max(years)}"
        }
        
        result = (models, sorted([int(y) for y in years]), set(vocabulary), metadata_out)
        
        # 4. Save to fast cache for next time
        try:
            joblib.dump(result, CACHE_FILE)
            # st.success("Saved to fast cache for future runs!")
        except Exception as e:
            st.warning(f"Could not save fast cache: {e}")
            
        st.success(f"✅ Loaded {len(years)} years ({min(years)}-{max(years)})!")
        return result
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

@st.cache_data 
def get_precomputed_word_stats(global_vocab, _models):
    word_to_years = {}
    word_to_total_count = {}
    
    for year, model in _models.items():
        for word in model.wv.index_to_key:
            if word not in word_to_years:
                word_to_years[word] = set()
            word_to_years[word].add(year)
            word_to_total_count[word] = word_to_total_count.get(word, 0) + 10
    
    return word_to_years, word_to_total_count
