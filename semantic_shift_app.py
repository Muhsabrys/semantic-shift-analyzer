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
from matplotlib.patches import FancyBboxPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec
import pandas as pd
import io
from datetime import datetime

from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine, euclidean

# Download NLTK data
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

# Get stopwords from NLTK
try:
    from nltk.corpus import stopwords
    ALL_STOPWORDS = set(stopwords.words('english'))
except:
    # Fallback stopwords list
    ALL_STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                         'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                         'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                         'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                         'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                         'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                         'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                         'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                         'further', 'then', 'once'])

# Page configuration
st.set_page_config(
    page_title="Semantic Shift Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Data loading and processing functions
@st.cache_data
def load_corpus_from_file(uploaded_file):
    """Load corpus from uploaded file (TXT, CSV, or XLSX)"""
    year_to_text = {}
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            # TXT format: each line should be "YEAR<tab>TEXT" or "YEAR,TEXT"
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try tab separator first, then comma
                if '\t' in line:
                    parts = line.split('\t', 1)
                elif ',' in line:
                    parts = line.split(',', 1)
                else:
                    st.warning(f"⚠️ Skipping line (no separator found): {line[:50]}...")
                    continue
                
                if len(parts) == 2:
                    try:
                        year = int(parts[0].strip())
                        text = parts[1].strip()
                        if text:
                            year_to_text[year] = text
                    except ValueError:
                        st.warning(f"⚠️ Invalid year format: {parts[0]}")
                        
        elif file_extension == 'csv':
            # CSV format: expects columns 'year' and 'text' (case-insensitive)
            df = pd.read_csv(uploaded_file)
            
            # Find year and text columns (case-insensitive)
            year_col = None
            text_col = None
            
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['year', 'years', 'date']:
                    year_col = col
                elif col_lower in ['text', 'content', 'document', 'corpus']:
                    text_col = col
            
            if year_col is None or text_col is None:
                st.error(f"❌ CSV must have 'year' and 'text' columns. Found: {list(df.columns)}")
                return None
            
            for _, row in df.iterrows():
                try:
                    year = int(row[year_col])
                    text = str(row[text_col]).strip()
                    if text and text != 'nan':
                        year_to_text[year] = text
                except (ValueError, TypeError) as e:
                    st.warning(f"⚠️ Skipping row with invalid data: {e}")
                    
        elif file_extension in ['xlsx', 'xls']:
            # Excel format: expects columns 'year' and 'text' (case-insensitive)
            df = pd.read_excel(uploaded_file)
            
            # Find year and text columns (case-insensitive)
            year_col = None
            text_col = None
            
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['year', 'years', 'date']:
                    year_col = col
                elif col_lower in ['text', 'content', 'document', 'corpus']:
                    text_col = col
            
            if year_col is None or text_col is None:
                st.error(f"❌ Excel must have 'year' and 'text' columns. Found: {list(df.columns)}")
                return None
            
            for _, row in df.iterrows():
                try:
                    year = int(row[year_col])
                    text = str(row[text_col]).strip()
                    if text and text != 'nan':
                        year_to_text[year] = text
                except (ValueError, TypeError) as e:
                    st.warning(f"⚠️ Skipping row with invalid data: {e}")
        else:
            st.error(f"❌ Unsupported file format: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None
    
    if not year_to_text:
        st.error("❌ No valid data found in uploaded file")
        return None
    
    return year_to_text

@st.cache_data
def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

@st.cache_data
def tokenize_corpus(year_to_text):
    """Tokenize all speeches"""
    year_to_tokens = {}
    
    for yr, text in year_to_text.items():
        cleaned = clean_text(text)
        tokens = [t for t in word_tokenize(cleaned) if t.isalpha()]
        year_to_tokens[yr] = tokens
    
    return year_to_tokens

@st.cache_resource
def train_models(_year_to_tokens):
    """Train Word2Vec models for each year using proper sentences."""
    models = {}

    for yr, tokens in _year_to_tokens.items():
        if len(tokens) < 10:
            continue

        # Break tokens into pseudo-sentences of length 20
        # (Realistic context windows for small yearly corpora)
        sentences = [tokens[i:i+20] for i in range(0, len(tokens), 20)]

        model = Word2Vec(
            sentences=sentences,
            vector_size=200,
            window=5,
            min_count=1,   # <-- CRUCIAL FIX
            sg=1,
            workers=4
        )

        models[yr] = model

    return models


def align_embeddings(base_model, other_model):
    """Align embeddings using Orthogonal Procrustes"""
    shared = list(set(base_model.wv.index_to_key) &
                  set(other_model.wv.index_to_key))
    
    if len(shared) < 20:
        st.warning(f"⚠️ Small shared vocabulary ({len(shared)} words)")
    
    B = np.array([base_model.wv[w] for w in shared])
    O = np.array([other_model.wv[w] for w in shared])
    
    B = normalize(B)
    O = normalize(O)
    
    R, _ = orthogonal_procrustes(O, B)
    A = O @ R
    
    aligned = {w: v for (w, v) in zip(shared, A)
               if w not in ALL_STOPWORDS}
    
    return aligned

def get_aligned_embeddings(models, target_word, years):
    """Get aligned embeddings for a target word across years"""
    candidate_years = [yr for yr in years if yr in models and target_word in models[yr].wv.key_to_index]
    
    if len(candidate_years) == 0:
        return None, None, None
    
    base_year = candidate_years[0]
    aligned = {}
    
    # Baseline
    aligned[base_year] = {
        w: models[base_year].wv[w]
        for w in models[base_year].wv.index_to_key
        if w not in ALL_STOPWORDS
    }
    
    # Align other years
    for yr in candidate_years[1:]:
        aligned[yr] = align_embeddings(models[base_year], models[yr])
    
    # Extract vectors for target word
    vectors = []
    valid_years = []
    
    for yr in candidate_years:
        if target_word in aligned[yr]:
            vectors.append(aligned[yr][target_word])
            valid_years.append(yr)
    
    return aligned, vectors, valid_years

def nearest_neighbors(aligned, year, word, topn=10):
    """Get nearest neighbors for a word in a specific year"""
    vecs = aligned[year]
    if word not in vecs:
        return []
    wv = vecs[word]
    sims = {w: 1 - cosine(wv, v) for w, v in vecs.items()}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:topn]

# Visualization functions
def plot_drift(vectors, valid_years, word):
    """Plot semantic drift over time"""
    base_vec = vectors[0]
    drift_scores = [cosine(base_vec, v) for v in vectors]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(valid_years, drift_scores, marker="o", linewidth=3, markersize=10, color='#2E86AB')
    
    # Trend line (only if we have enough data points)
    if len(valid_years) >= 3:
        try:
            z = np.polyfit(range(len(valid_years)), drift_scores, min(2, len(valid_years) - 1))
            p = np.poly1d(z)
            ax.plot(valid_years, p(range(len(valid_years))), "--", linewidth=2, color='#A23B72', alpha=0.7, label='Trend')
        except (np.linalg.LinAlgError, ValueError):
            # Skip trend line if polyfit fails
            pass
    
    ax.set_title(f"Semantic Drift of '{word}' Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Cosine Distance from Base Year", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    return fig

def plot_3d_trajectory(vectors, valid_years, word):
    """Plot 3D trajectory of semantic change"""
    vectors_array = np.array(vectors)
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors_array)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(vectors_3d)))
    
    for i in range(len(vectors_3d)-1):
        ax.plot(vectors_3d[i:i+2, 0], vectors_3d[i:i+2, 1], vectors_3d[i:i+2, 2],
                color=colors[i], linewidth=3, alpha=0.7)
    
    scatter = ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2],
                        c=range(len(valid_years)), cmap='viridis', 
                        s=200, edgecolor='black', linewidth=2, alpha=0.8)
    
    for i, year in enumerate(valid_years):
        ax.text(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2], 
                str(year), fontsize=9, fontweight='bold')
    
    ax.scatter([vectors_3d[0, 0]], [vectors_3d[0, 1]], [vectors_3d[0, 2]], 
              color='green', s=400, marker='*', edgecolor='black', linewidth=2)
    ax.scatter([vectors_3d[-1, 0]], [vectors_3d[-1, 1]], [vectors_3d[-1, 2]], 
              color='red', s=400, marker='*', edgecolor='black', linewidth=2)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=11)
    ax.set_title(f"3D Semantic Trajectory of '{word}'", fontsize=14, fontweight='bold')
    
    return fig

def plot_similarity_matrix(vectors, valid_years, word):
    """Plot cross-year similarity matrix"""
    n_years = len(valid_years)
    similarity_matrix = np.zeros((n_years, n_years))
    
    for i in range(n_years):
        for j in range(n_years):
            similarity_matrix[i, j] = 1 - cosine(vectors[i], vectors[j])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', interpolation='nearest')
    
    for i in range(n_years):
        for j in range(n_years):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="black" if similarity_matrix[i, j] > 0.5 else "white",
                          fontsize=8)
    
    ax.set_xticks(range(n_years))
    ax.set_yticks(range(n_years))
    ax.set_xticklabels(valid_years, rotation=45, ha='right')
    ax.set_yticklabels(valid_years)
    ax.set_title(f"Cross-Temporal Similarity Matrix for '{word}'", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    return fig

def plot_semantic_network(aligned, year, word, topn=15):
    """Plot semantic network for a word in a specific year"""
    neighbors = nearest_neighbors(aligned, year, word, topn=topn)
    
    G = nx.Graph()
    G.add_node(word, node_type='target')
    
    for w, sim in neighbors:
        G.add_node(w, node_type='neighbor')
        G.add_edge(word, w, weight=sim)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Draw target node
    nx.draw_networkx_nodes(G, pos, nodelist=[word], node_color='red', 
                           node_size=2500, node_shape='*', ax=ax,
                           edgecolors='black', linewidths=3)
    
    # Draw neighbor nodes
    neighbors_list = [n for n in G.nodes() if n != word]
    nx.draw_networkx_nodes(G, pos, nodelist=neighbors_list, 
                           node_color='lightblue', node_size=1200, ax=ax,
                           edgecolors='black', linewidths=2, alpha=0.8)
    
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)
    
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in edge_weights], 
                          alpha=0.6, edge_color='gray', ax=ax)
    
    ax.set_title(f"Semantic Network for '{word}' ({year})", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig

def plot_word_distance_evolution(aligned, word1, word2, years_list):
    """Plot distance evolution between two words"""
    distances = []
    valid_years = []
    
    for year in years_list:
        if year in aligned and word1 in aligned[year] and word2 in aligned[year]:
            vec1 = aligned[year][word1]
            vec2 = aligned[year][word2]
            cos_dist = cosine(vec1, vec2)
            distances.append(cos_dist)
            valid_years.append(year)
    
    if len(valid_years) == 0:
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distance over time
    ax1.plot(valid_years, distances, marker='o', linewidth=3, markersize=10, color='#E63946')
    
    z = np.polyfit(range(len(valid_years)), distances, 2)
    p = np.poly1d(z)
    ax1.plot(valid_years, p(range(len(valid_years))), "--", linewidth=2.5, color='#F4A261', alpha=0.8)
    
    min_idx = np.argmin(distances)
    max_idx = np.argmax(distances)
    
    ax1.scatter([valid_years[min_idx]], [distances[min_idx]], 
               color='green', s=300, marker='*', edgecolor='black', linewidth=2, zorder=5)
    ax1.scatter([valid_years[max_idx]], [distances[max_idx]], 
               color='red', s=300, marker='*', edgecolor='black', linewidth=2, zorder=5)
    
    ax1.set_title(f"Distance Between '{word1}' and '{word2}'", fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Cosine Distance', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Statistics
    stats_data = {
        'Mean': np.mean(distances),
        'Median': np.median(distances),
        'Min': np.min(distances),
        'Max': np.max(distances),
        'Std Dev': np.std(distances)
    }
    
    bars = ax2.barh(list(stats_data.keys()), list(stats_data.values()),
                    color=['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_title('Distance Statistics', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, stats_data.values()):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, stats_data

# Main application
def main():
    st.markdown('<div class="main-header">📊 Semantic Shift Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore how word meanings change over time in your text corpus</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source - upload only
    st.sidebar.subheader("📁 Upload Your Corpus")
    st.sidebar.markdown("""
    **File Formats:**
    - **TXT**: `YEAR<tab>TEXT` or `YEAR,TEXT` (one per line)
    - **CSV/XLSX**: Columns named `year` and `text`
    
    **Requirements:**
    - Years must be integers
    - Each year needs text content
    - Minimum 3 years recommended
    """)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your corpus file",
        type=['txt', 'csv', 'xlsx', 'xls'],
        help="Upload a file with year and text data"
    )
    
    year_to_text = None
    
    if uploaded_file is not None:
        with st.spinner("Loading your corpus..."):
            year_to_text = load_corpus_from_file(uploaded_file)
            
        if year_to_text is not None:
            years = sorted(year_to_text.keys())
            st.sidebar.success(f"✅ Loaded {len(years)} documents ({min(years)}-{max(years)})")
            
            # Show preview
            with st.sidebar.expander("📊 Preview Data"):
                st.write(f"**Years:** {', '.join(map(str, years[:10]))}" + ("..." if len(years) > 10 else ""))
                st.write(f"**Sample text from year {years[0]}:**")
                st.text(year_to_text[years[0]][:200] + "...")
    else:
        st.info("👆 Please upload a corpus file to begin analysis")
        st.markdown("""
        ### 📝 How to Format Your File:
        
        **TXT Format (Tab-separated):**
        ```
        2020	Your text content for 2020
        2021	Your text content for 2021
        2022	Your text content for 2022
        ```
        
        **TXT Format (Comma-separated):**
        ```
        2020,Your text content for 2020
        2021,Your text content for 2021
        ```
        
        **CSV/Excel Format:**
        | year | text |
        |------|------|
        | 2020 | Your text content |
        | 2021 | More text content |
        
        ### 💡 Tips:
        - Years can be any integers (2000, 2005, 2010, etc.)
        - Non-consecutive years are fine
        - Longer text per year = better embeddings
        - Minimum 3 years recommended for analysis
        
        ### 📄 Example Files:
        Download example templates to get started:
        - example_corpus.txt
        - example_corpus.csv
        - example_corpus.xlsx
        """)
        return
    
    # Process corpus
    if year_to_text is not None:
        with st.spinner("Processing corpus..."):
            year_to_tokens = tokenize_corpus(year_to_text)
            years = sorted(year_to_text.keys())
    
            # ---- NEW: Show top 5 frequent words from the whole corpus ----
            all_tokens_flat = []
            for toks in year_to_tokens.values():
                all_tokens_flat.extend([t for t in toks if t not in ALL_STOPWORDS])
    
            freq_dist = nltk.FreqDist(all_tokens_flat)
    
            top5_words = [w for w, _ in freq_dist.most_common(5)]
    
            st.sidebar.markdown("### 🔍 Suggested Words to Explore")
            st.sidebar.success(", ".join(top5_words))

        # Train models
        cache_key = f"models_{uploaded_file.name}_{len(years)}"
        if cache_key not in st.session_state:
            with st.spinner("Training Word2Vec models... This may take a minute."):
                st.session_state[cache_key] = train_models(year_to_tokens)
            st.sidebar.success(f"✅ Trained {len(st.session_state[cache_key])} models")
        
        models = st.session_state[cache_key]
    
    # Analysis mode
    st.sidebar.subheader("📋 Analysis Mode")
    mode = st.sidebar.radio(
        "Choose analysis type:",
        ["Single Word Drift", "Word-to-Word Distance", "Semantic Network", "Multi-Word Comparison"]
    )
    
    # Main content
    if mode == "Single Word Drift":
        st.header("🔍 Single Word Semantic Drift Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            target_word = st.text_input("Enter a word to analyze:", value="crisis")
        
        with col2:
            st.write("")
            st.write("")
            analyze_btn = st.button("🚀 Analyze", type="primary")
        
        if analyze_btn:
            aligned, vectors, valid_years = get_aligned_embeddings(models, target_word, years)
            
            if vectors is None or len(vectors) == 0:
                st.error(f"❌ Word '{target_word}' not found in the corpus or insufficient data.")
            else:
                st.success(f"✅ Found '{target_word}' in {len(valid_years)} speeches")
                
                # Tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Drift Plot", "🌐 3D Trajectory", "🔥 Similarity Matrix", "📊 Statistics"])
                
                with tab1:
                    st.pyplot(plot_drift(vectors, valid_years, target_word))
                
                with tab2:
                    st.pyplot(plot_3d_trajectory(vectors, valid_years, target_word))
                    st.info("💡 The 3D plot shows how the word moves through semantic space. Points far apart indicate meaning change.")
                
                with tab3:
                    st.pyplot(plot_similarity_matrix(vectors, valid_years, target_word))
                    st.info("💡 Darker colors indicate higher similarity. Diagonal is always 1.0 (perfect self-similarity).")
                
                with tab4:
                    base_vec = vectors[0]
                    drift_scores = [cosine(base_vec, v) for v in vectors]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Drift", f"{np.mean(drift_scores):.4f}")
                    col2.metric("Max Drift", f"{np.max(drift_scores):.4f}")
                    col3.metric("Min Drift", f"{np.min(drift_scores):.4f}")
                    col4.metric("Std Dev", f"{np.std(drift_scores):.4f}")
                    
                    # Year-by-year data
                    st.subheader("Year-by-Year Drift Scores")
                    df = pd.DataFrame({
                        'Year': valid_years,
                        'Drift Score': drift_scores
                    })
                    st.dataframe(df, use_container_width=True)
    
    elif mode == "Word-to-Word Distance":
        st.header("🔗 Word-to-Word Distance Evolution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            word1 = st.text_input("First word:", value="crisis")
        
        with col2:
            word2 = st.text_input("Second word:", value="problem")
        
        analyze_btn = st.button("🚀 Compare Words", type="primary")
        
        if analyze_btn or (word1 and word2):
            # Get aligned embeddings for both words
            aligned1, vectors1, valid_years1 = get_aligned_embeddings(models, word1, years)
            aligned2, vectors2, valid_years2 = get_aligned_embeddings(models, word2, years)
            
            if vectors1 is None or vectors2 is None:
                st.error(f"❌ One or both words not found in the corpus.")
            else:
                # Use the aligned embeddings from word1 (they're in the same space)
                fig, stats = plot_word_distance_evolution(aligned1, word1, word2, valid_years1)
                
                if fig is None:
                    st.error(f"❌ Cannot compare '{word1}' and '{word2}' - insufficient overlap in years.")
                else:
                    st.pyplot(fig)
                    
                    # Display statistics
                    st.subheader("📊 Distance Statistics")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Mean", f"{stats['Mean']:.4f}")
                    col2.metric("Median", f"{stats['Median']:.4f}")
                    col3.metric("Min", f"{stats['Min']:.4f}")
                    col4.metric("Max", f"{stats['Max']:.4f}")
                    col5.metric("Std Dev", f"{stats['Std Dev']:.4f}")
                    
                    # Interpretation
                    st.subheader("💡 Interpretation")
                    if stats['Mean'] < 0.3:
                        st.success(f"✅ '{word1}' and '{word2}' are **very similar** in meaning across the years.")
                    elif stats['Mean'] < 0.7:
                        st.info(f"ℹ️ '{word1}' and '{word2}' are **moderately related** with some semantic overlap.")
                    else:
                        st.warning(f"⚠️ '{word1}' and '{word2}' are **quite different** in their usage contexts.")
    
    elif mode == "Semantic Network":
        st.header("🕸️ Semantic Network Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            word = st.text_input("Enter a word:", value="crisis")
        
        with col2:
            year = st.selectbox("Select year:", sorted(models.keys()))
        
        with col3:
            topn = st.slider("Number of neighbors:", 5, 30, 15)
        
        analyze_btn = st.button("🚀 Generate Network", type="primary")
        
        if analyze_btn or word:
            aligned, vectors, valid_years = get_aligned_embeddings(models, word, years)
            
            if aligned is None or year not in aligned:
                st.error(f"❌ Word '{word}' not found in year {year}")
            else:
                st.pyplot(plot_semantic_network(aligned, year, word, topn))
                
                # Show neighbor list
                st.subheader(f"📝 Top {topn} Neighbors of '{word}' in {year}")
                neighbors = nearest_neighbors(aligned, year, word, topn)
                
                df = pd.DataFrame(neighbors, columns=['Word', 'Similarity'])
                df['Similarity'] = df['Similarity'].round(4)
                df.index = range(1, len(df) + 1)
                st.dataframe(df, use_container_width=True)
    
    else:  # Multi-Word Comparison
        st.header("📊 Multi-Word Comparative Analysis")
        
        words_input = st.text_input(
            "Enter words to compare (comma-separated):",
            value="crisis, war, peace, economy"
        )
        
        words_list = [w.strip() for w in words_input.split(",")]
        
        analyze_btn = st.button("🚀 Compare Words", type="primary")
        
        if analyze_btn or words_list:
            all_drift_data = {}
            
            for word in words_list:
                aligned, vectors, valid_years = get_aligned_embeddings(models, word, years)
                
                if vectors is not None and len(vectors) >= 2:
                    base_vec = vectors[0]
                    drift = [cosine(base_vec, v) for v in vectors]
                    all_drift_data[word] = {'years': valid_years, 'drift': drift}
            
            if len(all_drift_data) == 0:
                st.error("❌ None of the words found in the corpus.")
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(all_drift_data)))
                
                # Plot drift over time
                for idx, (word, data) in enumerate(all_drift_data.items()):
                    ax1.plot(data['years'], data['drift'], marker='o', linewidth=2.5, 
                            markersize=8, color=colors[idx], label=word, alpha=0.8)
                
                ax1.set_xlabel('Year', fontsize=13)
                ax1.set_ylabel('Cosine Distance from Base Year', fontsize=13)
                ax1.set_title('Comparative Semantic Drift', fontsize=15, fontweight='bold')
                ax1.legend(fontsize=11)
                ax1.grid(True, linestyle='--', alpha=0.5)
                
                # Bar chart of total drift
                total_drifts = {word: data['drift'][-1] - data['drift'][0] 
                               for word, data in all_drift_data.items()}
                
                sorted_words = sorted(total_drifts.items(), key=lambda x: abs(x[1]), reverse=True)
                words_sorted = [w for w, _ in sorted_words]
                drifts_sorted = [d for _, d in sorted_words]
                
                bars = ax2.barh(range(len(words_sorted)), drifts_sorted, 
                               color=[colors[words_list.index(w)] if w in words_list else 'gray' 
                                     for w in words_sorted],
                               edgecolor='black', linewidth=1.5, alpha=0.8)
                
                ax2.set_yticks(range(len(words_sorted)))
                ax2.set_yticklabels(words_sorted, fontsize=12)
                ax2.set_xlabel('Total Drift (End - Start)', fontsize=13)
                ax2.set_title('Total Semantic Shift by Word', fontsize=15, fontweight='bold')
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
                ax2.grid(axis='x', alpha=0.3)
                
                for i, (bar, drift) in enumerate(zip(bars, drifts_sorted)):
                    ax2.text(drift + (0.01 if drift > 0 else -0.01), i, f'{drift:.3f}', 
                            va='center', ha='left' if drift > 0 else 'right', 
                            fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.success(f"✅ Successfully compared {len(all_drift_data)} words")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info("""
    This app analyzes semantic drift in text corpora using:
    - **Word2Vec** embeddings
    - **Procrustes alignment** for temporal comparison
    - **PCA** for dimensionality reduction
    
    Built with Streamlit 🎈
    """)

if __name__ == "__main__":
    main()
