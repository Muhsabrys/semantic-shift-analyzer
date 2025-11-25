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
            # Check what keys exist
            st.write("NPZ keys:", list(npz_data.keys()))
            
            # Try to extract data - adjust based on actual structure
            years = npz_data['years']
            vocabulary = npz_data['vocabulary']
            
            # Show first year to understand structure
            st.write("Years:", years[:5] if len(years) > 5 else years)
            st.write("Vocabulary size:", len(vocabulary))
            
            # The embeddings might be stored differently
            # Let's check all keys
            for key in npz_data.keys():
                st.write(f"{key}: {type(npz_data[key])}, shape: {npz_data[key].shape if hasattr(npz_data[key], 'shape') else 'N/A'}")
        
        return None, None, None, None
        
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
