"""
Semantic Shift Analysis - Interactive Web Application
Analyzes semantic drift in text corpora using Word2Vec embeddings
"""

import streamlit as st
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine

# -------------------------------------------------------------------------
# NLTK downloads
# -------------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

try:
    from nltk.corpus import stopwords
    ALL_STOPWORDS = set(stopwords.words('english'))
except:
    ALL_STOPWORDS = {
        'i','me','my','myself','we','our','ours','ourselves','you',
        'your','yours','yourself','yourselves','he','him','his','himself',
        'she','her','hers','herself','it','its','itself','they','them',
        'their','theirs','themselves','what','which','who','whom','this',
        'that','these','those','am','is','are','was','were','be','been',
        'being','have','has','had','having','do','does','did','doing',
        'a','an','the','and','but','if','or','because','as','until',
        'while','of','at','by','for','with','about','against','between',
        'into','through','during','before','after','above','below','to',
        'from','up','down','in','out','on','off','over','under','again',
        'further','then','once'
    }

# -------------------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Semantic Shift Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# Text cleaning and loading
# -------------------------------------------------------------------------
@st.cache_data
def clean_text(text):
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

@st.cache_data
def load_corpus_from_file(uploaded_file):
    year_to_text = {}
    ext = uploaded_file.name.split('.')[-1].lower()

    try:
        if ext == "txt":
            data = uploaded_file.getvalue().decode("utf8").splitlines()
            for line in data:
                if "\t" in line:
                    y, t = line.split("\t", 1)
                elif "," in line:
                    y, t = line.split(",", 1)
                else:
                    continue
                try:
                    year_to_text[int(y.strip())] = t.strip()
                except:
                    pass

        elif ext == "csv":
            df = pd.read_csv(uploaded_file)
            ycol = None
            tcol = None
            for c in df.columns:
                cl = c.lower()
                if cl in ["year","years","date"]:
                    ycol = c
                if cl in ["text","content","corpus","document"]:
                    tcol = c
            if ycol and tcol:
                for _, row in df.iterrows():
                    try:
                        year_to_text[int(row[ycol])] = str(row[tcol])
                    except:
                        pass

        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            ycol = None
            tcol = None
            for c in df.columns:
                cl = c.lower()
                if cl in ["year","years","date"]:
                    ycol = c
                if cl in ["text","content","corpus","document"]:
                    tcol = c
            if ycol and tcol:
                for _, row in df.iterrows():
                    try:
                        year_to_text[int(row[ycol])] = str(row[tcol])
                    except:
                        pass

        else:
            st.error(f"Unsupported file type: {ext}")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    if not year_to_text:
        st.error("No valid (year,text) lines found.")
        return None

    return year_to_text

# -------------------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------------------
@st.cache_data
def tokenize_corpus(year_to_text):
    year_to_tokens = {}
    for yr, text in year_to_text.items():
        cleaned = clean_text(text)
        tokens = [t for t in word_tokenize(cleaned) if t.isalpha()]
        year_to_tokens[yr] = tokens
    return year_to_tokens

# -------------------------------------------------------------------------
# Train Word2Vec models
# -------------------------------------------------------------------------
@st.cache_resource
def train_models(year_to_tokens):
    models = {}
    for yr, tokens in year_to_tokens.items():
        if len(tokens) < 10:
            continue
        sentences = [tokens[i:i+20] for i in range(0, len(tokens), 20)]
        model = Word2Vec(
            sentences=sentences,
            vector_size=200,
            window=5,
            min_count=1,
            sg=1,
            workers=4,
        )
        models[yr] = model
    return models

# -------------------------------------------------------------------------
# Embedding Alignment
# -------------------------------------------------------------------------
def align_embeddings(base_model, other_model):
    shared = list(set(base_model.wv.index_to_key) & set(other_model.wv.index_to_key))
    if len(shared) < 5:
        return {}

    B = np.array([base_model.wv[w] for w in shared])
    O = np.array([other_model.wv[w] for w in shared])

    B = normalize(B)
    O = normalize(O)

    R, _ = orthogonal_procrustes(O, B)
    A = O @ R

    return {w: v for (w, v) in zip(shared, A) if w not in ALL_STOPWORDS}

def get_aligned_embeddings(models, target_word, all_years):
    candidate_years = [y for y in all_years if y in models and target_word in models[y].wv.key_to_index]
    if len(candidate_years) == 0:
        return None, None, None

    base_year = candidate_years[0]
    aligned = {}

    aligned[base_year] = {
        w: models[base_year].wv[w]
        for w in models[base_year].wv.index_to_key
        if w not in ALL_STOPWORDS
    }

    for yr in candidate_years[1:]:
        aligned[yr] = align_embeddings(models[base_year], models[yr])

    vectors = []
    years_out = []

    for yr in candidate_years:
        if target_word in aligned[yr]:
            vectors.append(aligned[yr][target_word])
            years_out.append(yr)

    return aligned, vectors, years_out

# -------------------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------------------
def plot_drift(vectors, years, word):
    base_vec = vectors[0]
    drift = [cosine(base_vec, v) for v in vectors]

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(years, drift, marker='o', linewidth=3, color="#2E86AB")
    ax.set_title(f"Semantic Drift of '{word}'")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine Distance from Base")
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

def plot_3d_trajectory(vectors, years, word):
    if len(vectors) < 3:
        return None
    arr = np.array(vectors)
    pca = PCA(n_components=3)
    try:
        pts = pca.fit_transform(arr)
    except:
        return None

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', linewidth=2, markersize=8)
    for i, yr in enumerate(years):
        ax.text(pts[i,0], pts[i,1], pts[i,2], str(yr))

    ax.set_title(f"3D Trajectory of '{word}'")
    return fig

def plot_similarity_matrix(vectors, years, word):
    if len(vectors) < 2:
        return None

    n = len(vectors)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mat[i,j] = 1 - cosine(vectors[i], vectors[j])

    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(mat, cmap="YlOrRd")
    plt.colorbar(im)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(years)
    ax.set_yticklabels(years)
    ax.set_title(f"Similarity Matrix for '{word}'")
    return fig

def nearest_neighbors(aligned, year, word, topn=10):
    if year not in aligned:
        return []
    if word not in aligned[year]:
        return []
    wv = aligned[year][word]
    sims = {w: 1 - cosine(wv, v) for w, v in aligned[year].items()}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:topn]

def plot_semantic_network(aligned, year, word, topn=10):
    neigh = nearest_neighbors(aligned, year, word, topn)
    if not neigh:
        return None

    G = nx.Graph()
    G.add_node(word)

    for w, s in neigh:
        G.add_node(w)
        G.add_edge(word, w, weight=s)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10,8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=1200, edge_color="gray", ax=ax)
    ax.set_title(f"Semantic Network of '{word}' ({year})")
    return fig

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    st.markdown("<h1 style='text-align:center;'>📊 Semantic Shift Analyzer</h1>", unsafe_allow_html=True)

    st.sidebar.header("Upload Corpus")
    file = st.sidebar.file_uploader("Upload TXT / CSV / XLSX", type=['txt','csv','xlsx'])

    if not file:
        st.info("Upload a file to begin.")
        return

    year_to_text = load_corpus_from_file(file)
    if not year_to_text:
        return

    years = sorted(year_to_text.keys())

    year_to_tokens = tokenize_corpus(year_to_text)

    models_key = f"models_{file.name}_{len(years)}"
    if models_key not in st.session_state:
        with st.spinner("Training Word2Vec models..."):
            st.session_state[models_key] = train_models(year_to_tokens)

    models = st.session_state[models_key]

    st.sidebar.header("Analysis Mode")
    mode = st.sidebar.radio("Choose:", ["Single Word Drift", "Semantic Network"])

    if mode == "Single Word Drift":
        st.header("Single Word Drift")
        word = st.text_input("Enter a word", "economy")
        if st.button("Analyze"):
            aligned, vecs, yrs = get_aligned_embeddings(models, word, years)
            if not vecs:
                st.error("Word not found enough times.")
                return

            tab1, tab2, tab3 = st.tabs(["Drift", "3D Trajectory", "Similarity Matrix"])

            with tab1:
                st.pyplot(plot_drift(vecs, yrs, word))

            with tab2:
                fig3d = plot_3d_trajectory(vecs, yrs, word)
                if fig3d:
                    st.pyplot(fig3d)
                else:
                    st.warning("Not enough data for 3D.")

            with tab3:
                mat = plot_similarity_matrix(vecs, yrs, word)
                if mat:
                    st.pyplot(mat)
                else:
                    st.warning("Need ≥ 2 years.")

    if mode == "Semantic Network":
        st.header("Semantic Network")
        word = st.text_input("Word", "economy")
        year = st.selectbox("Year", years)
        if st.button("Generate"):
            aligned, vecs, yrs = get_aligned_embeddings(models, word, years)
            if not aligned or year not in aligned:
                st.error("Word not available in that year.")
            else:
                fig = plot_semantic_network(aligned, year, word, topn=12)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Cannot build network.")

if __name__ == "__main__":
    main()
