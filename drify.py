"""
Semantic Shift Analysis - Interactive Web Application
Analyzes semantic drift in text corpora using improved Word2Vec methodology
with lemmatization, vocabulary alignment, and stability enhancements
"""

import streamlit as st
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec
import pandas as pd
import io
from datetime import datetime
import warnings
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes

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
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Get stopwords and lemmatizer
try:
    from nltk.corpus import stopwords
    ALL_STOPWORDS = set(stopwords.words('english'))
    
    # Remove words that are often important for semantic analysis
    # These are technically stopwords but can have meaningful semantic shifts
    KEEP_WORDS = {
        # Time-related words
        'day', 'days', 'year', 'years', 'time', 'times', 
        'week', 'weeks', 'month', 'months', 'today', 'now',
        # Negations (important for sentiment/meaning)
        'no', 'not', 'nor', 'neither',
        # Modal verbs (important for certainty/obligation)
        'can', 'could', 'may', 'might', 'must', 'should', 'would',
        # Other potentially meaningful words
        'own', 'same', 'such', 'than', 'too', 'very', 'just', 'still'
    }
    ALL_STOPWORDS = ALL_STOPWORDS - KEEP_WORDS
except:
    ALL_STOPWORDS = set()

# Initialize lemmatizer globally
lemmatizer = WordNetLemmatizer()

# Page config
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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_corpus_from_file(uploaded_file):
    """Load corpus from uploaded file (TXT, CSV, or XLSX)"""
    year_to_text = {}
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
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
            df = pd.read_csv(uploaded_file)
            
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
            
            # Group by year and combine all text (handles multiple rows per year)
            grouped = df.groupby(year_col)[text_col].apply(lambda x: ' '.join(x.astype(str))).reset_index()
            
            for _, row in grouped.iterrows():
                try:
                    year = int(row[year_col])
                    text = str(row[text_col]).strip()
                    if text and text != 'nan':
                        year_to_text[year] = text
                except (ValueError, TypeError) as e:
                    st.warning(f"⚠️ Skipping row with invalid data: {e}")
                    
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            
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
            
            # Group by year and combine all text (handles multiple rows per year)
            grouped = df.groupby(year_col)[text_col].apply(lambda x: ' '.join(x.astype(str))).reset_index()
            
            for _, row in grouped.iterrows():
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

# Text processing with lemmatization
@st.cache_data
def clean_and_lemmatize(text):
    """Clean, normalize, and lemmatize text"""
    # Remove non-alphabetic characters
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize and filter
    lemmatized = []
    for token in tokens:
        if token.isalpha() and len(token) >= 2:  # Filter very short words (1 character)
            lemma = lemmatizer.lemmatize(token, pos='v')  # Verb lemmatization
            lemma = lemmatizer.lemmatize(lemma, pos='n')  # Noun lemmatization
            if lemma not in ALL_STOPWORDS and len(lemma) >= 2:  # Check lemma length too
                lemmatized.append(lemma)
    
    return lemmatized

@st.cache_data
def tokenize_corpus(year_to_text):
    """Tokenize and lemmatize all texts"""
    year_to_tokens = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    years = sorted(year_to_text.keys())
    for idx, yr in enumerate(years):
        status_text.text(f"Processing year {yr}...")
        progress_bar.progress((idx + 1) / len(years))
        year_to_tokens[yr] = clean_and_lemmatize(year_to_text[yr])
    
    progress_bar.empty()
    status_text.empty()
    
    return year_to_tokens

# Improved vocabulary building
@st.cache_data
def build_global_vocabulary(_year_to_tokens, min_years=2, min_total_count=3):
    """
    Build a global vocabulary that appears across multiple years.
    This ensures vocabulary stability and reduces OOV problems.
    """
    from collections import Counter
    
    # Count word occurrences per year
    word_to_years = {}
    word_to_total_count = Counter()
    
    for yr, tokens in _year_to_tokens.items():
        unique_words = set(tokens)
        token_counts = Counter(tokens)
        
        for word in unique_words:
            if word not in word_to_years:
                word_to_years[word] = set()
            word_to_years[word].add(yr)
            word_to_total_count[word] += token_counts[word]
    
    # Filter vocabulary
    global_vocab = set()
    for word, years_set in word_to_years.items():
        # Must appear in at least min_years AND have min_total_count occurrences
        if len(years_set) >= min_years and word_to_total_count[word] >= min_total_count:
            global_vocab.add(word)
    
    return global_vocab, word_to_years, word_to_total_count

# Improved training with multiple random seeds
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
                sg=1,  # Skip-gram
                workers=1,  # Single worker for reproducibility per seed
                seed=seed_idx,
                epochs=20,  # More epochs for better convergence
                negative=10  # More negative samples
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
    """
    Robust Procrustes alignment using global vocabulary
    """
    # Use global vocabulary for alignment
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
    """
    Compute semantic drift with better error handling and vocabulary checking
    """
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

# ===============================================================
# ADVANCED VISUALIZATION FUNCTIONS
# ===============================================================

def plot_enhanced_drift_with_statistics(word, models, years, global_vocab):
    """Enhanced drift plot with statistics and annotations"""
    # Get vectors for all years
    vectors = []
    valid_years = []
    
    for yr in years:
        if yr in models and word in models[yr].wv:
            vectors.append(models[yr].wv[word])
            valid_years.append(yr)
    
    if len(vectors) < 2:
        st.error(f"Insufficient data for '{word}' - found in {len(vectors)} year(s)")
        return
    
    # Compute drift scores
    base_vec = vectors[0]
    drift_scores = [cosine(base_vec, v) for v in vectors]
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main drift plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(valid_years, drift_scores, marker="o", linewidth=3, markersize=10,
             color='#2E86AB', label='Cosine Distance')
    
    # Add trend line
    if len(valid_years) >= 3:
        z = np.polyfit(range(len(valid_years)), drift_scores, min(2, len(valid_years)-1))
        p = np.poly1d(z)
        ax1.plot(valid_years, p(range(len(valid_years))), "--",
                 linewidth=2, color='#A23B72', alpha=0.7, label='Polynomial Trend')
    
    # Highlight significant changes
    if len(drift_scores) > 1:
        drift_changes = np.diff(drift_scores)
        threshold = np.std(drift_changes) * 1.5 if len(drift_changes) > 1 else 0
        for i, change in enumerate(drift_changes):
            if abs(change) > threshold:
                ax1.axvspan(valid_years[i], valid_years[i+1], alpha=0.2, color='red')
                ax1.annotate(f'Δ={change:.3f}', xy=(valid_years[i], drift_scores[i]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                            fontsize=9)
    
    ax1.set_title(f"Semantic Drift of '{word}' Over Time",
                  fontsize=18, fontweight='bold')
    ax1.set_xlabel("Year", fontsize=14)
    ax1.set_ylabel("Cosine Distance from Base Year", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Distribution of drift scores
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(drift_scores, bins=min(15, len(drift_scores)), color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(drift_scores), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(np.median(drift_scores), color='green', linestyle='--', linewidth=2, label='Median')
    ax2.set_title('Distribution of Drift Scores', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Drift Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    
    # Year-over-year change
    if len(drift_scores) > 1:
        ax3 = fig.add_subplot(gs[1, 1])
        drift_changes = np.diff(drift_scores)
        ax3.bar(range(len(drift_changes)), drift_changes, 
                color=['green' if x > 0 else 'red' for x in drift_changes],
                alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_title('Year-over-Year Drift Change', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time Period Index', fontsize=12)
        ax3.set_ylabel('Change in Drift', fontsize=12)
        ax3.set_xticks(range(len(drift_changes)))
        ax3.set_xticklabels([f"{valid_years[i]}-{valid_years[i+1]}" for i in range(len(drift_changes))],
                             rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_3d_semantic_trajectory(word, models, years, global_vocab):
    """3D visualization of semantic trajectory using PCA"""
    # Get vectors for all years
    vectors = []
    valid_years = []
    
    for yr in years:
        if yr in models and word in models[yr].wv:
            vectors.append(models[yr].wv[word])
            valid_years.append(yr)
    
    if len(vectors) < 3:
        st.error(f"Need at least 3 years for 3D visualization - found {len(vectors)}")
        return
    
    # Use PCA to reduce to 3D
    vectors_array = np.array(vectors)
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors_array)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, len(vectors_3d)))
    for i in range(len(vectors_3d)-1):
        ax.plot(vectors_3d[i:i+2, 0], vectors_3d[i:i+2, 1], vectors_3d[i:i+2, 2],
                color=colors[i], linewidth=3, alpha=0.7)
    
    # Plot points
    scatter = ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2],
                        c=range(len(valid_years)), cmap='viridis',
                        s=200, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Annotate points
    for i, year in enumerate(valid_years):
        ax.text(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2],
                str(year), fontsize=10, fontweight='bold')
    
    # Highlight start and end
    ax.scatter([vectors_3d[0, 0]], [vectors_3d[0, 1]], [vectors_3d[0, 2]],
              color='green', s=400, marker='*', edgecolor='black', linewidth=2, label='Start')
    ax.scatter([vectors_3d[-1, 0]], [vectors_3d[-1, 1]], [vectors_3d[-1, 2]],
              color='red', s=400, marker='*', edgecolor='black', linewidth=2, label='End')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12)
    ax.set_title(f"3D Semantic Trajectory of '{word}' Over Time", fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Time Period', fontsize=12)
    cbar.set_ticks(range(0, len(valid_years), max(1, len(valid_years)//5)))
    cbar.set_ticklabels([valid_years[i] for i in range(0, len(valid_years), max(1, len(valid_years)//5))])
    
    ax.legend(fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_temporal_heatmap(word, models, years, global_vocab):
    """Temporal heatmap showing dimensional changes"""
    # Get vectors for all years
    vectors = []
    valid_years = []
    
    for yr in years:
        if yr in models and word in models[yr].wv:
            vectors.append(models[yr].wv[word])
            valid_years.append(yr)
    
    if len(vectors) < 2:
        st.error(f"Insufficient data for heatmap - found {len(vectors)} year(s)")
        return
    
    vectors_array = np.array(vectors)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Top 50 dimensions by variance
    dim_variance = np.var(vectors_array, axis=0)
    top_dims = np.argsort(dim_variance)[-50:]
    
    vectors_subset = vectors_array[:, top_dims]
    
    # Heatmap of raw values
    im1 = axes[0].imshow(vectors_subset.T, aspect='auto', cmap='RdBu_r',
                         interpolation='nearest')
    axes[0].set_yticks(range(0, 50, 5))
    axes[0].set_yticklabels([f'Dim {d}' for d in top_dims[::5]])
    axes[0].set_xticks(range(len(valid_years)))
    axes[0].set_xticklabels(valid_years, rotation=45, ha='right')
    axes[0].set_title(f"Top 50 Most Variable Dimensions for '{word}'",
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Dimension', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Embedding Value')
    
    # Heatmap of changes
    if len(vectors_subset) > 1:
        changes = np.diff(vectors_subset, axis=0)
        im2 = axes[1].imshow(changes.T, aspect='auto', cmap='seismic',
                             interpolation='nearest', vmin=-np.max(np.abs(changes)),
                             vmax=np.max(np.abs(changes)))
        axes[1].set_yticks(range(0, 50, 5))
        axes[1].set_yticklabels([f'Dim {d}' for d in top_dims[::5]])
        axes[1].set_xticks(range(len(valid_years)-1))
        axes[1].set_xticklabels([f"{valid_years[i]}-{valid_years[i+1]}"
                                 for i in range(len(valid_years)-1)],
                                rotation=45, ha='right')
        axes[1].set_title(f"Year-over-Year Dimensional Changes",
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Period', fontsize=12)
        axes[1].set_ylabel('Dimension', fontsize=12)
        plt.colorbar(im2, ax=axes[1], label='Change in Value')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_similarity_matrix(word, models, years, global_vocab):
    """Cross-year similarity matrix"""
    # Get vectors for all years
    vectors = []
    valid_years = []
    
    for yr in years:
        if yr in models and word in models[yr].wv:
            vectors.append(models[yr].wv[word])
            valid_years.append(yr)
    
    if len(vectors) < 2:
        st.error(f"Insufficient data for similarity matrix - found {len(vectors)} year(s)")
        return
    
    # Create similarity matrix
    n_years = len(valid_years)
    similarity_matrix = np.zeros((n_years, n_years))
    
    for i in range(n_years):
        for j in range(n_years):
            similarity_matrix[i, j] = 1 - cosine(vectors[i], vectors[j])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', interpolation='nearest')
    
    # Add text annotations
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
    ax.set_title(f"Cross-Temporal Semantic Similarity Matrix for '{word}'",
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def nearest_neighbors(year, word, model, topn=10):
    """Get nearest neighbors for a word in a specific year"""
    if word not in model.wv:
        return []
    wv = model.wv[word]
    sims = {w: 1 - cosine(wv, model.wv[w]) for w in model.wv.index_to_key}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:topn]

def visualize_enhanced_semantic_network(year, word, model, global_vocab, topn=15):
    """Enhanced semantic network with community detection"""
    if word not in model.wv:
        st.error(f"'{word}' not found in year {year}")
        return
    
    neighbors = nearest_neighbors(year, word, model, topn=topn)
    
    G = nx.Graph()
    G.add_node(word, node_type='target')
    
    # Add edges with similarity weights
    for w, sim in neighbors:
        G.add_node(w, node_type='neighbor')
        G.add_edge(word, w, weight=sim)
    
    # Compute community structure
    communities = list(nx.community.greedy_modularity_communities(G))
    
    # Assign colors based on communities
    color_map = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    node_to_community = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = idx
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # ---- Left plot: Spring layout ----
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Draw nodes
    for node in G.nodes():
        if node == word:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='red',
                                   node_size=2500, node_shape='*', ax=ax1,
                                   edgecolors='black', linewidths=3)
        else:
            comm_idx = node_to_community[node]
            nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                   node_color=[color_map[comm_idx]],
                                   node_size=1200, ax=ax1,
                                   edgecolors='black', linewidths=2, alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax1)
    
    # Draw edges with varying thickness
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in edge_weights],
                          alpha=0.6, edge_color='gray', ax=ax1)
    
    # Add edge labels for top connections
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    top_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)[:5]
    top_edge_labels = {(u, v): edge_labels[(u, v)] for u, v, _ in top_edges}
    nx.draw_networkx_edge_labels(G, pos, top_edge_labels, font_size=9, ax=ax1)
    
    ax1.set_title(f"Semantic Network for '{word}' ({year})\nSpring Layout",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # ---- Right plot: Circular layout with hierarchy ----
    pos_circular = nx.circular_layout(G)
    
    # Draw nodes
    for node in G.nodes():
        if node == word:
            nx.draw_networkx_nodes(G, pos_circular, nodelist=[node], node_color='red',
                                   node_size=2500, node_shape='*', ax=ax2,
                                   edgecolors='black', linewidths=3)
        else:
            comm_idx = node_to_community[node]
            nx.draw_networkx_nodes(G, pos_circular, nodelist=[node],
                                   node_color=[color_map[comm_idx]],
                                   node_size=1200, ax=ax2,
                                   edgecolors='black', linewidths=2, alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos_circular, font_size=11, font_weight='bold', ax=ax2)
    
    # Draw edges with gradient colors based on weight
    for (u, v, d) in edges:
        weight = d['weight']
        color = plt.cm.YlOrRd(weight)
        nx.draw_networkx_edges(G, pos_circular, [(u, v)], width=weight*5,
                              alpha=0.7, edge_color=[color], ax=ax2)
    
    ax2.set_title(f"Semantic Network for '{word}' ({year})\nCircular Layout",
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Print network statistics
    st.markdown(f"""
    ### Network Statistics for '{word}' in {year}
    - **Number of nodes:** {G.number_of_nodes()}
    - **Number of edges:** {G.number_of_edges()}
    - **Average clustering coefficient:** {nx.average_clustering(G):.3f}
    - **Network density:** {nx.density(G):.3f}
    - **Number of communities:** {len(communities)}
    
    **Top 5 strongest connections:**
    """)
    
    for u, v, d in sorted(edges, key=lambda x: x[2]['weight'], reverse=True)[:5]:
        st.write(f"- {u} ↔ {v}: {d['weight']:.3f}")

def visualize_neighbor_evolution(word, models, years_to_compare, global_vocab, topn=10):
    """Shows how semantic neighborhood changes over time"""
    year_neighbors = {}
    all_neighbors = set()
    
    # Collect all neighbors across years
    for year in years_to_compare:
        if year in models and word in models[year].wv:
            neighbors = nearest_neighbors(year, word, models[year], topn=topn)
            year_neighbors[year] = {w: sim for w, sim in neighbors if w != word}
            all_neighbors.update(year_neighbors[year].keys())
    
    if not year_neighbors:
        st.error(f"No data found for '{word}' in selected years")
        return
    
    all_neighbors = sorted(all_neighbors)
    
    # Create heatmap data
    heatmap_data = np.zeros((len(all_neighbors), len(years_to_compare)))
    for j, year in enumerate(years_to_compare):
        if year in year_neighbors:
            for i, neighbor in enumerate(all_neighbors):
                if neighbor in year_neighbors[year]:
                    heatmap_data[i, j] = year_neighbors[year][neighbor]
    
    # Plot bar charts for each year
    n_years = len(years_to_compare)
    fig, axes = plt.subplots(n_years, 1, figsize=(14, 5*n_years))
    
    if n_years == 1:
        axes = [axes]
    
    for idx, year in enumerate(years_to_compare):
        if year in year_neighbors:
            neighbors = sorted(year_neighbors[year].items(),
                              key=lambda x: x[1], reverse=True)
            words = [w for w, _ in neighbors]
            similarities = [s for _, s in neighbors]
            
            bars = axes[idx].barh(range(len(words)), similarities,
                                 color=plt.cm.viridis(similarities),
                                 edgecolor='black', linewidth=1.5)
            
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words, fontsize=11)
            axes[idx].set_xlabel('Similarity Score', fontsize=12)
            axes[idx].set_title(f"Top Neighbors of '{word}' in {year}",
                               fontsize=13, fontweight='bold')
            axes[idx].set_xlim(0, 1)
            axes[idx].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, sim) in enumerate(zip(bars, similarities)):
                axes[idx].text(sim + 0.01, i, f'{sim:.3f}',
                              va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Show heatmap of all neighbors across years
    if len(years_to_compare) > 1:
        fig, ax = plt.subplots(figsize=(12, max(8, len(all_neighbors)*0.3)))
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                      interpolation='nearest')
        
        ax.set_xticks(range(len(years_to_compare)))
        ax.set_xticklabels(years_to_compare)
        ax.set_yticks(range(len(all_neighbors)))
        ax.set_yticklabels(all_neighbors, fontsize=10)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Neighbor Words', fontsize=12)
        ax.set_title(f"Temporal Evolution of Semantic Neighbors for '{word}'",
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations for non-zero cells
        for i in range(len(all_neighbors)):
            for j in range(len(years_to_compare)):
                if heatmap_data[i, j] > 0:
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                  ha="center", va="center",
                                  color="black" if heatmap_data[i, j] < 0.5 else "white",
                                  fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Similarity Score')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def compare_multiple_words(words_list, models, year_range, global_vocab):
    """Compare semantic drift of multiple words simultaneously"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(words_list)))
    
    # Track valid data for each word
    all_drift_data = {}
    
    for idx, word in enumerate(words_list):
        if word not in global_vocab:
            continue
        
        word_vectors = []
        word_years = []
        
        for yr in year_range:
            if yr in models and word in models[yr].wv:
                word_vectors.append(models[yr].wv[word])
                word_years.append(yr)
        
        if len(word_vectors) < 2:
            continue
        
        # Calculate drift
        base_vec = word_vectors[0]
        drift = [cosine(base_vec, v) for v in word_vectors]
        
        all_drift_data[word] = {'years': word_years, 'drift': drift}
        
        # Plot drift over time
        ax1.plot(word_years, drift, marker='o', linewidth=2.5,
                markersize=8, color=colors[idx], label=word, alpha=0.8)
    
    if not all_drift_data:
        st.error("No valid words found for comparison")
        plt.close()
        return
    
    ax1.set_xlabel('Year', fontsize=13)
    ax1.set_ylabel('Cosine Distance from Base Year', fontsize=13)
    ax1.set_title('Comparative Semantic Drift Analysis',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Bar chart of total drift
    total_drifts = {word: data['drift'][-1] - data['drift'][0]
                   for word, data in all_drift_data.items()}
    
    sorted_words = sorted(total_drifts.items(), key=lambda x: abs(x[1]), reverse=True)
    words_sorted = [w for w, _ in sorted_words]
    drifts_sorted = [d for _, d in sorted_words]
    
    bars = ax2.barh(range(len(words_sorted)), drifts_sorted,
                    color=[colors[words_list.index(w)] for w in words_sorted if w in words_list],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_yticks(range(len(words_sorted)))
    ax2.set_yticklabels(words_sorted, fontsize=12)
    ax2.set_xlabel('Total Drift (End - Start)', fontsize=13)
    ax2.set_title('Total Semantic Shift by Word',
                  fontsize=15, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, drift) in enumerate(zip(bars, drifts_sorted)):
        ax2.text(drift + (0.01 if drift > 0 else -0.01), i, f'{drift:.3f}',
                va='center', ha='left' if drift > 0 else 'right',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Plotting functions with better error handling
def plot_drift_robust(word, models, years, global_vocab, metric='cosine'):
    """Plot semantic drift with robust error handling"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Check if word exists in global vocabulary
    if word not in global_vocab:
        ax.text(0.5, 0.5, f'Word "{word}" not found in global vocabulary\n\n' +
                'This word may not appear frequently enough across years,\n' +
                'or may not appear in at least 2 different time periods.',
                ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
        return
    
    # Check word presence per year
    years_with_word = []
    for yr in years:
        if yr in models and word in models[yr].wv:
            years_with_word.append(yr)
    
    if len(years_with_word) < 2:
        ax.text(0.5, 0.5, f'Word "{word}" appears in too few years ({len(years_with_word)})\n\n' +
                'Need at least 2 years for drift analysis.',
                ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
        return
    
    base_year = years_with_word[0]
    
    drift_scores = []
    valid_years = []
    
    for target_yr in years_with_word[1:]:
        score = compute_drift_score_robust(word, base_year, target_yr, models, 
                                          global_vocab, metric)
        if score is not None:
            drift_scores.append(score)
            valid_years.append(target_yr)
    
    if not drift_scores:
        ax.text(0.5, 0.5, 'Insufficient data for drift analysis',
                ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
        return
    
    # Plot drift
    ax.plot(valid_years, drift_scores, marker='o', linewidth=2, markersize=8, 
            color='#1f77b4')
    
    # Add trend line if enough points
    if len(valid_years) >= 3:
        try:
            degree = min(2, len(valid_years) - 1)
            z = np.polyfit(range(len(valid_years)), drift_scores, degree)
            p = np.poly1d(z)
            ax.plot(valid_years, p(range(len(valid_years))), 
                   linestyle='--', color='red', alpha=0.5, label='Trend')
            ax.legend()
        except (np.linalg.LinAlgError, ValueError):
            pass
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} Distance', fontsize=12)
    ax.set_title(f'Semantic Drift: "{word}" (baseline: {base_year})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

def plot_semantic_network_robust(words, year, models, global_vocab, threshold=0.7):
    """Plot semantic network with robust checking"""
    if year not in models:
        st.error(f"❌ No model available for year {year}")
        return
    
    model = models[year]
    
    # Filter words that exist
    valid_words = [w for w in words if w in model.wv and w in global_vocab]
    
    if len(valid_words) < 2:
        st.error(f"❌ Insufficient valid words in year {year}. Found: {len(valid_words)}")
        st.info(f"Words in global vocabulary: {[w for w in words if w in global_vocab]}")
        st.info(f"Words in year {year} model: {[w for w in words if w in model.wv]}")
        return
    
    # Compute similarity matrix
    n = len(valid_words)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                sim = model.wv.similarity(valid_words[i], valid_words[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            except:
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
    
    # Create network plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Position nodes in circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Draw edges
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] > threshold:
                ax.plot([x[i], x[j]], [y[i], y[j]], 
                       'gray', alpha=similarity_matrix[i, j], 
                       linewidth=similarity_matrix[i, j]*3)
    
    # Draw nodes
    ax.scatter(x, y, s=500, c='lightblue', edgecolors='navy', linewidths=2, zorder=5)
    
    # Add labels
    for i, word in enumerate(valid_words):
        ax.text(x[i]*1.15, y[i]*1.15, word, 
               fontsize=11, ha='center', va='center', fontweight='bold')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title(f'Semantic Network - Year {year}\n(threshold={threshold:.2f})', 
                fontsize=14, fontweight='bold')
    
    st.pyplot(fig)
    plt.close()
    
    # Show similarity matrix
    st.subheader("Similarity Matrix")
    df_sim = pd.DataFrame(similarity_matrix, 
                         index=valid_words, 
                         columns=valid_words)
    st.dataframe(df_sim.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1))

# Main application
def main():
    st.markdown('<p class="main-header">📊 Semantic Shift Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Robust Analysis of Word Meaning Evolution Over Time</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("📁 Upload Your Corpus")
    st.sidebar.markdown("""
    **File Formats:**
    - **TXT**: `YEAR<tab>TEXT` or `YEAR,TEXT` (one per line)
    - **CSV/XLSX**: Columns named `year` and `text`
    
    **Requirements:**
    - Years must be integers
    - Each year needs text content
    - Minimum 3-5 years recommended
    - At least 500+ words per year for stability
    """)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your corpus file",
        type=['txt', 'csv', 'xlsx', 'xls'],
        help="Upload a file with year and text data"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        vector_size = st.slider("Vector Size", 50, 300, 200, 50,
                               help="Larger = more expressive but slower")
        window_size = st.slider("Context Window", 3, 10, 5, 1,
                               help="How many surrounding words to consider")
        min_years = st.slider("Min Years for Vocabulary", 2, 5, 2, 1,
                             help="Word must appear in at least this many years")
        min_total_count = st.slider("Min Total Occurrences", 2, 10, 3, 1,
                                   help="Word must appear this many times total")
        n_seeds = st.slider("Training Seeds", 1, 10, 5, 1,
                           help="More seeds = more stable but slower")
    
    year_to_text = None
    
    if uploaded_file is not None:
        with st.spinner("Loading your corpus..."):
            year_to_text = load_corpus_from_file(uploaded_file)
            
        if year_to_text is not None:
            years = sorted(year_to_text.keys())
            
            # Show corpus statistics
            total_words = sum(len(text.split()) for text in year_to_text.values())
            avg_words = total_words / len(years)
            
            st.sidebar.success(f"✅ Loaded {len(years)} documents ({min(years)}-{max(years)})")
            st.sidebar.metric("Total Words", f"{total_words:,}")
            st.sidebar.metric("Avg Words/Year", f"{avg_words:.0f}")
            
            # Warning for small corpora
            if avg_words < 500:
                st.sidebar.markdown("""
                <div class="warning-box">
                ⚠️ <strong>Small Corpus Warning</strong><br>
                Your corpus has <500 words per year on average.<br>
                Results may be unstable. Consider adding more text.
                </div>
                """, unsafe_allow_html=True)
            
            # Show preview
            with st.sidebar.expander("📊 Preview Data"):
                st.write(f"**Years:** {', '.join(map(str, years[:10]))}" + 
                        ("..." if len(years) > 10 else ""))
                st.write(f"**Sample text from year {years[0]}:**")
                st.text(year_to_text[years[0]][:200] + "...")
    else:
        st.info("👆 Please upload a corpus file to begin analysis")
        st.markdown("""
        ### 🎯 Key Improvements in This Version:
        
        **✅ Lemmatization**
        - Converts words to base forms (running → run)
        - Reduces vocabulary fragmentation
        - Better semantic tracking
        
        **✅ Global Vocabulary**
        - Filters words appearing across multiple years
        - Reduces OOV (Out of Vocabulary) errors
        - Ensures alignment stability
        
        **✅ Stability via Averaging**
        - Trains multiple models with different random seeds
        - Averages embeddings to reduce variance
        - More reliable results
        
        **✅ Better Error Handling**
        - Clear messages when words don't exist
        - Explains why analysis failed
        - Suggests fixes
        
        **✅ Corpus Quality Checks**
        - Warns about small corpora
        - Shows vocabulary statistics
        - Recommends minimum data sizes
        
        ### 📝 How to Format Your File:
        
        **TXT Format (Tab-separated):**
        ```
        2020	Your text content for 2020 (aim for 500+ words)
        2021	Your text content for 2021
        2022	Your text content for 2022
        ```
        
        **CSV/Excel Format:**
        | year | text |
        |------|------|
        | 2020 | Your text content (500+ words) |
        | 2021 | More text content |
        
        ### 💡 Tips for Best Results:
        - **More text per year = better embeddings** (aim for 500+ words)
        - **More years = better drift tracking** (5+ years ideal)
        - **Consistent topics** help maintain vocabulary overlap
        - **Lemmatization** will automatically group word variants
        """)
        return
    
    # Process corpus
    if year_to_text is not None:
        with st.spinner("Processing corpus with lemmatization..."):
            year_to_tokens = tokenize_corpus(year_to_text)
            years = sorted(year_to_text.keys())
        
        # Build global vocabulary
        with st.spinner("Building global vocabulary..."):
            global_vocab, word_to_years, word_to_total_count = build_global_vocabulary(
                year_to_tokens, min_years=min_years, min_total_count=min_total_count
            )
        
        st.sidebar.success(f"✅ Global vocabulary: {len(global_vocab):,} words")
        
        # Show vocabulary info
        with st.sidebar.expander("📖 Vocabulary Info"):
            st.write(f"**Total unique words (before filtering):** {len(word_to_years):,}")
            st.write(f"**Global vocabulary (after filtering):** {len(global_vocab):,}")
            st.write(f"**Filter criteria:**")
            st.write(f"  - Min years: {min_years}")
            st.write(f"  - Min total occurrences: {min_total_count}")
            
            # Sample words
            sample_words = sorted(list(global_vocab))[:20]
            st.write(f"**Sample words:** {', '.join(sample_words)}")
        
        # Train models
        cache_key = f"models_{uploaded_file.name}_{len(years)}_{vector_size}_{n_seeds}"
        if cache_key not in st.session_state:
            with st.spinner(f"Training models with {n_seeds} seeds for stability... This may take a few minutes."):
                st.session_state[cache_key] = train_stable_models(
                    year_to_tokens, 
                    global_vocab,
                    n_seeds=n_seeds,
                    vector_size=vector_size,
                    window=window_size
                )
            st.sidebar.success(f"✅ Trained {len(st.session_state[cache_key])} models")
        
        models = st.session_state[cache_key]
        
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🔍 Single Word Drift",
            "📏 Word-to-Word Distance",
            "🕸️ Semantic Network",
            "📊 Multi-Word Comparison",
            "📈 Enhanced Drift Plot",
            "🎯 3D Trajectory",
            "🔥 Temporal Heatmap & Matrix",
            "🌐 Enhanced Networks & Evolution"
        ])
        
        with tab1:
            st.subheader("📈 Single Word Semantic Drift")
            st.markdown("""
            Track how a word's meaning shifts over time relative to a baseline year.
            
            **Note:** The word must:
            - Be in the global vocabulary
            - Appear in at least 2 different years
            - Have sufficient occurrences
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                word = st.text_input("Enter a word to analyze:", "government").lower()
            with col2:
                metric = st.selectbox("Distance metric:", ["cosine", "euclidean"])
            
            # Check word validity
            if word:
                # Check if this word or its lemmatized form is in vocabulary
                lemmatized = lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n')
                
                if word not in global_vocab and lemmatized not in global_vocab:
                    # Check if the word exists in any year (before filtering)
                    word_in_any_year = word in word_to_years or lemmatized in word_to_years
                    
                    error_msg = f'❌ Word "{word}" not in global vocabulary.\n\n'
                    
                    if lemmatized != word:
                        error_msg += f'💡 **Lemmatization:** "{word}" → "{lemmatized}"\n\n'
                    
                    if word_in_any_year:
                        # Word exists but filtered out
                        actual_word = word if word in word_to_years else lemmatized
                        years_with_word = len(word_to_years.get(actual_word, set()))
                        total_occurrences = word_to_total_count.get(actual_word, 0)
                        
                        error_msg += f'**Word found but filtered out:**\n'
                        error_msg += f'- Appears in {years_with_word} year(s) (need {min_years}+)\n'
                        error_msg += f'- Total occurrences: {total_occurrences} (need {min_total_count}+)\n\n'
                        
                        if years_with_word >= min_years and total_occurrences >= min_total_count:
                            error_msg += f'⚠️ **BUG DETECTED:** Word meets criteria but still filtered!\n'
                            error_msg += f'This shouldn\'t happen. Please report this issue.\n\n'
                        
                        error_msg += f'**Solution:**\n'
                        if years_with_word < min_years:
                            error_msg += f'- Lower "Min Years" to {years_with_word} in Advanced Settings\n'
                        if total_occurrences < min_total_count:
                            error_msg += f'- Lower "Min Total Occurrences" to {total_occurrences} in Advanced Settings\n'
                        error_msg += f'- Then retrain the models (cache will be cleared)'
                    else:
                        error_msg += f'**Word not found in corpus**\n'
                        error_msg += f'- Neither "{word}" nor "{lemmatized}" appear in your text\n'
                        error_msg += f'- May have been filtered during cleaning:\n'
                        error_msg += f'  • Removed as stopword\n'
                        error_msg += f'  • Too short (<2 characters after lemmatization)\n'
                        error_msg += f'  • Contains non-alphabetic characters\n\n'
                        error_msg += f'**Try:**\n'
                        error_msg += f'- Check spelling\n'
                        error_msg += f'- Search vocabulary list below to see what\'s available\n'
                        error_msg += f'- Try synonyms or related words'
                    
                    st.error(error_msg)
                    
                    # Show some similar words from vocabulary
                    similar_words = [w for w in sorted(global_vocab) if word[:3] in w or lemmatized[:3] in w]
                    if similar_words:
                        st.info(f"**Similar words in vocabulary:** {', '.join(similar_words[:15])}")
                else:
                    # Use lemmatized form if original not in vocab
                    word_to_use = word if word in global_vocab else lemmatized
                    
                    if word_to_use != word:
                        st.info(f'💡 Using lemmatized form: "{word}" → "{word_to_use}"')
                    
                    # Show word statistics
                    years_present = word_to_years.get(word_to_use, set())
                    total_count = word_to_total_count.get(word_to_use, 0)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Years Present", len(years_present))
                    col2.metric("Total Occurrences", total_count)
                    col3.metric("Years Available", ', '.join(map(str, sorted(years_present))))
                    
                    plot_drift_robust(word_to_use, models, years, global_vocab, metric)
        
        with tab2:
            st.subheader("📏 Word-to-Word Distance Evolution")
            st.markdown("""
            Track how the semantic distance between two words changes over time.
            Useful for analyzing relationship shifts.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                word1 = st.text_input("First word:", "economy").lower()
            with col2:
                word2 = st.text_input("Second word:", "jobs").lower()
            
            if word1 and word2:
                # Validate both words
                errors = []
                if word1 not in global_vocab:
                    errors.append(f'"{word1}" not in global vocabulary')
                if word2 not in global_vocab:
                    errors.append(f'"{word2}" not in global vocabulary')
                
                if errors:
                    st.error("❌ " + " and ".join(errors))
                else:
                    # Compute distance evolution
                    distances = []
                    valid_years_pair = []
                    
                    for yr in years:
                        if yr in models and word1 in models[yr].wv and word2 in models[yr].wv:
                            try:
                                dist = 1 - models[yr].wv.similarity(word1, word2)
                                distances.append(dist)
                                valid_years_pair.append(yr)
                            except:
                                pass
                    
                    if distances:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(valid_years_pair, distances, marker='o', linewidth=2, 
                               markersize=8, color='#2ca02c')
                        ax.set_xlabel('Year', fontsize=12)
                        ax.set_ylabel('Semantic Distance', fontsize=12)
                        ax.set_title(f'Distance Evolution: "{word1}" ↔ "{word2}"', 
                                   fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("⚠️ Insufficient data for distance analysis")
        
        with tab3:
            st.subheader("🕸️ Semantic Network Visualization")
            st.markdown("Visualize semantic relationships between multiple words in a specific year.")
            
            year_select = st.selectbox("Select year:", years)
            words_input = st.text_input(
                "Enter words (comma-separated):",
                "economy, government, people, nation, freedom"
            )
            threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.7, 0.05)
            
            if words_input:
                words_list = [w.strip().lower() for w in words_input.split(',')]
                plot_semantic_network_robust(words_list, year_select, models, 
                                            global_vocab, threshold)
        
        with tab4:
            st.subheader("📊 Multi-Word Drift Comparison")
            st.markdown("Compare semantic drift across multiple words simultaneously.")
            
            words_compare = st.text_input(
                "Enter words to compare (comma-separated):",
                "freedom, liberty, democracy"
            )
            
            if words_compare:
                words_list = [w.strip().lower() for w in words_compare.split(',')]
                
                # Validate all words
                valid_words = [w for w in words_list if w in global_vocab]
                invalid_words = [w for w in words_list if w not in global_vocab]
                
                if invalid_words:
                    st.warning(f"⚠️ Skipping invalid words: {', '.join(invalid_words)}")
                
                if len(valid_words) >= 2:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for word in valid_words:
                        base_year = years[0]
                        drift_scores = []
                        valid_years_word = []
                        
                        for target_yr in years[1:]:
                            score = compute_drift_score_robust(
                                word, base_year, target_yr, models, global_vocab
                            )
                            if score is not None:
                                drift_scores.append(score)
                                valid_years_word.append(target_yr)
                        
                        if drift_scores:
                            ax.plot(valid_years_word, drift_scores, marker='o', 
                                   linewidth=2, label=word)
                    
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Cosine Distance from Baseline', fontsize=12)
                    ax.set_title(f'Comparative Semantic Drift (baseline: {years[0]})', 
                               fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.error("❌ Need at least 2 valid words for comparison")
        
        with tab5:
            st.subheader("📈 Enhanced Drift Plot with Statistics")
            st.markdown("""
            Advanced drift visualization with:
            - Trend analysis
            - Significant change detection
            - Distribution analysis
            - Year-over-year changes
            """)
            
            word_enhanced = st.text_input("Enter word for enhanced analysis:", "government", key="enhanced_word").lower()
            
            if word_enhanced and word_enhanced in global_vocab:
                plot_enhanced_drift_with_statistics(word_enhanced, models, years, global_vocab)
            elif word_enhanced:
                st.error(f"❌ '{word_enhanced}' not in global vocabulary")
        
        with tab6:
            st.subheader("🎯 3D Semantic Trajectory")
            st.markdown("""
            Visualize semantic evolution in 3D space using PCA.
            Shows the trajectory of word meaning through time.
            """)
            
            word_3d = st.text_input("Enter word for 3D visualization:", "crisis", key="3d_word").lower()
            
            if word_3d and word_3d in global_vocab:
                plot_3d_semantic_trajectory(word_3d, models, years, global_vocab)
            elif word_3d:
                st.error(f"❌ '{word_3d}' not in global vocabulary")
        
        with tab7:
            st.subheader("🔥 Temporal Analysis: Heatmap & Similarity Matrix")
            st.markdown("""
            Deep dive into dimensional changes and cross-temporal similarities.
            """)
            
            word_heatmap = st.text_input("Enter word for temporal analysis:", "freedom", key="heatmap_word").lower()
            
            if word_heatmap and word_heatmap in global_vocab:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Dimensional Heatmap")
                    plot_temporal_heatmap(word_heatmap, models, years, global_vocab)
                
                with col2:
                    st.markdown("### Similarity Matrix")
                    plot_similarity_matrix(word_heatmap, models, years, global_vocab)
            elif word_heatmap:
                st.error(f"❌ '{word_heatmap}' not in global vocabulary")
        
        with tab8:
            st.subheader("🌐 Enhanced Semantic Networks & Neighbor Evolution")
            st.markdown("""
            Advanced network analysis with community detection and temporal neighbor tracking.
            """)
            
            # Enhanced network section
            st.markdown("### Enhanced Semantic Network")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                word_network = st.text_input("Enter word for network:", "economy", key="network_word").lower()
            with col2:
                year_network = st.selectbox("Select year:", years, key="network_year")
            with col3:
                topn_network = st.slider("Number of neighbors:", 5, 25, 15, key="network_topn")
            
            if word_network and word_network in global_vocab:
                if year_network in models:
                    visualize_enhanced_semantic_network(year_network, word_network, 
                                                       models[year_network], global_vocab, topn_network)
                else:
                    st.error(f"❌ No model for year {year_network}")
            elif word_network:
                st.error(f"❌ '{word_network}' not in global vocabulary")
            
            st.markdown("---")
            
            # Neighbor evolution section
            st.markdown("### Temporal Evolution of Neighbors")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                word_evolution = st.text_input("Enter word to track:", "technology", key="evolution_word").lower()
            with col2:
                topn_evolution = st.slider("Top N neighbors:", 5, 20, 10, key="evolution_topn")
            
            # Select years to compare
            if len(years) > 4:
                step = max(1, len(years) // 4)
                default_years = years[::step]
            else:
                default_years = years
            
            selected_years = st.multiselect(
                "Select years to compare:",
                years,
                default=default_years,
                key="evolution_years"
            )
            
            if word_evolution and word_evolution in global_vocab and selected_years:
                visualize_neighbor_evolution(word_evolution, models, selected_years, 
                                            global_vocab, topn_evolution)
            elif word_evolution and word_evolution not in global_vocab:
                st.error(f"❌ '{word_evolution}' not in global vocabulary")
            elif word_evolution and not selected_years:
                st.warning("⚠️ Please select at least one year to compare")
        
        # Vocabulary explorer
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔍 Explore Vocabulary"):
            search_term = st.text_input("Search vocabulary:", "")
            if search_term:
                matching = [w for w in sorted(global_vocab) if search_term.lower() in w]
                st.write(f"**Found {len(matching)} matching words:**")
                st.write(", ".join(matching[:50]))
                if len(matching) > 50:
                    st.write(f"... and {len(matching)-50} more")
            else:
                st.write("**All vocabulary words:**")
                all_words = sorted(list(global_vocab))
                st.write(f"Total: {len(all_words)} words")
                st.text_area("Vocabulary preview", value=", ".join(all_words[:100]), height=200, label_visibility="hidden")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info("""
    **Improved Semantic Shift Analyzer**
    
    This version includes:
    - Lemmatization for vocabulary stability
    - Global vocabulary filtering
    - Multi-seed training for robustness
    - Better error handling
    - Corpus quality checks
    
    Built with Streamlit 🎈
    """)

if __name__ == "__main__":
    main()