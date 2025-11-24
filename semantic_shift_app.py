"""
Semantic Shift Analysis - FULL EXTENDED VERSION
PART 1 — Core Engine (Imports, Cleaning, Tokenization, Model Training, Alignment)
"""

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------
import streamlit as st
import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Semantic Shift Analyzer – Extended",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# NLTK DOWNLOADS
# -----------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

download_nltk_data()

# -----------------------------------------------------------
# STOPWORDS
# -----------------------------------------------------------
try:
    from nltk.corpus import stopwords
    ALL_STOPWORDS = set(stopwords.words("english"))
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

# -----------------------------------------------------------
# CLEAN TEXT
# -----------------------------------------------------------
@st.cache_data
def clean_text(text):
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# -----------------------------------------------------------
# FILE LOADING (TXT, CSV, XLSX)
# -----------------------------------------------------------
@st.cache_data
def load_corpus_from_file(uploaded_file):
    year_to_text = {}
    ext = uploaded_file.name.split(".")[-1].lower()

    try:
        if ext == "txt":
            lines = uploaded_file.getvalue().decode("utf8").splitlines()
            for line in lines:
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
                cl = c.lower().strip()
                if cl in ["year", "years", "date"]:
                    ycol = c
                if cl in ["text", "content", "document", "corpus"]:
                    tcol = c

            if ycol is None or tcol is None:
                st.error("CSV must contain 'year' and 'text' columns.")
                return None

            for _, row in df.iterrows():
                try:
                    yr = int(row[ycol])
                    txt = str(row[tcol]).strip()
                    year_to_text[yr] = txt
                except:
                    pass

        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            ycol = None
            tcol = None
            for c in df.columns:
                cl = c.lower().strip()
                if cl in ["year", "years", "date"]:
                    ycol = c
                if cl in ["text", "content", "document", "corpus"]:
                    tcol = c

            if ycol is None or tcol is None:
                st.error("Excel must contain 'year' and 'text' columns.")
                return None

            for _, row in df.iterrows():
                try:
                    yr = int(row[ycol])
                    txt = str(row[tcol]).strip()
                    year_to_text[yr] = txt
                except:
                    pass

        else:
            st.error(f"Unsupported file type: {ext}")
            return None

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    if not year_to_text:
        st.error("File contains no valid (year, text) entries.")
        return None

    return year_to_text

# -----------------------------------------------------------
# TOKENIZATION
# -----------------------------------------------------------
@st.cache_data
def tokenize_corpus(year_to_text):
    result = {}
    for yr, text in year_to_text.items():
        cleaned = clean_text(text)
        tokens = [t for t in word_tokenize(cleaned) if t.isalpha()]
        result[yr] = tokens
    return result

# -----------------------------------------------------------
# WORD2VEC TRAINING
# -----------------------------------------------------------
@st.cache_resource
def train_models(year_to_tokens):
    models = {}

    for yr, tokens in year_to_tokens.items():
        if len(tokens) < 10:
            continue

        # pseudo-sentences of length 20
        sentences = [tokens[i:i+20] for i in range(0, len(tokens), 20)]

        model = Word2Vec(
            sentences=sentences,
            vector_size=200,
            window=5,
            min_count=1,
            sg=1,
            workers=4
        )

        models[yr] = model

    return models

# -----------------------------------------------------------
# ALIGNMENT ENGINE
# -----------------------------------------------------------
def align_embeddings(base_model, other_model):
    shared = list(set(base_model.wv.index_to_key) &
                  set(other_model.wv.index_to_key))

    if len(shared) < 5:
        return {}

    B = np.array([base_model.wv[w] for w in shared])
    O = np.array([other_model.wv[w] for w in shared])

    B = normalize(B)
    O = normalize(O)

    R, _ = orthogonal_procrustes(O, B)
    aligned = O @ R

    return {
        w: v for (w, v) in zip(shared, aligned)
        if w not in ALL_STOPWORDS
    }

# -----------------------------------------------------------
# GET ALIGNED EMBEDDINGS FOR A WORD
# -----------------------------------------------------------
def get_aligned_embeddings(models, target_word, years):
    candidate_years = [
        yr for yr in years
        if yr in models and target_word in models[yr].wv.index_to_key
    ]

    if len(candidate_years) == 0:
        return None, None, None

    base_year = candidate_years[0]
    aligned = {}

    # baseline (no rotation)
    aligned[base_year] = {
        w: models[base_year].wv[w]
        for w in models[base_year].wv.index_to_key
        if w not in ALL_STOPWORDS
    }

    # rotate other years
    for yr in candidate_years[1:]:
        aligned[yr] = align_embeddings(models[base_year], models[yr])

    vectors = []
    out_years = []

    for yr in candidate_years:
        if target_word in aligned[yr]:
            vectors.append(aligned[yr][target_word])
            out_years.append(yr)

    return aligned, vectors, out_years


# ===========================================================
# PART 2 — VISUALIZATION MODULE (Drift, 3D, Similarity, Network)
# ===========================================================

# -----------------------------------------------------------
# 1) SEMANTIC DRIFT PLOT
# -----------------------------------------------------------
def plot_drift(vectors, years, word):
    base_vec = vectors[0]
    drift = [cosine(base_vec, v) for v in vectors]

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(
        years, drift,
        marker='o', markersize=9,
        linewidth=3,
        color="#1768AC"
    )

    ax.set_title(f"Semantic Drift of '{word}'", fontsize=18)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine Distance from Base")
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig

# -----------------------------------------------------------
# 2) 3D PCA TRAJECTORY
# -----------------------------------------------------------
def plot_3d_trajectory(vectors, years, word):
    if len(vectors) < 3:
        return None

    X = np.array(vectors)

    try:
        pca = PCA(n_components=3)
        pts = pca.fit_transform(X)
    except ValueError:
        return None

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        pts[:,0], pts[:,1], pts[:,2],
        '-o', linewidth=2, markersize=8, color="#DB3069"
    )

    for i, yr in enumerate(years):
        ax.text(pts[i,0], pts[i,1], pts[i,2], str(yr))

    ax.set_title(f"3D Trajectory of '{word}'", fontsize=17)

    return fig

# -----------------------------------------------------------
# 3) SIMILARITY MATRIX
# -----------------------------------------------------------
def plot_similarity_matrix(vectors, years, word):
    if len(vectors) < 2:
        return None

    n = len(vectors)
    mat = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            mat[i,j] = 1 - cosine(vectors[i], vectors[j])

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(
        mat,
        cmap="YlOrRd",
        annot=True,
        xticklabels=years,
        yticklabels=years,
        ax=ax
    )

    ax.set_title(f"Similarity Matrix for '{word}'", fontsize=17)
    return fig

# -----------------------------------------------------------
# 4) NEAREST NEIGHBORS
# -----------------------------------------------------------
def nearest_neighbors(aligned, year, word, topn=10):
    if year not in aligned:
        return []

    if word not in aligned[year]:
        return []

    wv = aligned[year][word]
    sims = {w: 1 - cosine(wv, v) for w, v in aligned[year].items()}
    sims_sorted = sorted(sims.items(), key=lambda x: x[1], reverse=True)

    return sims_sorted[:topn]

# -----------------------------------------------------------
# 5) SEMANTIC NETWORK
# -----------------------------------------------------------
def plot_semantic_network(aligned, year, word, topn=12):
    neigh = nearest_neighbors(aligned, year, word, topn)
    if not neigh:
        return None

    G = nx.Graph()
    G.add_node(word)

    for w, s in neigh:
        G.add_node(w)
        G.add_edge(word, w, weight=float(s))

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(11,8))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=1300,
        node_color="#8AB6D6"
    )
    nx.draw_networkx_edges(
        G, pos,
        width=2,
        edge_color="#555"
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight="bold"
    )

    ax.set_title(f"Semantic Network of '{word}' — {year}", fontsize=18)
    ax.axis("off")

    return fig

# -----------------------------------------------------------
# 6) STATISTICS SUMMARY
# -----------------------------------------------------------
def show_stats(word, years, vectors):
    import pandas as pd

    rows = []
    for yr, vec in zip(years, vectors):
        rows.append([yr, float(np.linalg.norm(vec))])

    df = pd.DataFrame(rows, columns=["Year", "Vector Norm"])
    return df

# ===========================================================
# PART 3 — INTERACTIVE UI + APP FLOW
# ===========================================================

def main():
    st.markdown("""
        <h1 style='text-align:center; font-size:42px;'>
            📊 Semantic Shift Analyzer
        </h1>
    """, unsafe_allow_html=True)

    st.sidebar.header("📁 Upload Corpus")
    file = st.sidebar.file_uploader("Upload TXT / CSV / XLSX", type=['txt','csv','xlsx'])

    if not file:
        st.info("⬆️ Upload a corpus file to begin analysis.")
        return

    # Load corpus
    year_to_text = load_corpus_from_file(file)
    if not year_to_text:
        return

    years = sorted(year_to_text.keys())

    # Tokenization
    year_to_tokens = tokenize_corpus(year_to_text)

    # Train or load models
    models_key = f"models_{file.name}_{len(years)}"
    if models_key not in st.session_state:
        with st.spinner("⚙️ Training Word2Vec models for each year…"):
            st.session_state[models_key] = train_models(year_to_tokens)

    models = st.session_state[models_key]

    st.sidebar.header("📌 Analysis Mode")
    mode = st.sidebar.radio("Select:", [
        "Single Word Drift",
        "Semantic Network",
    ])

    # ===========================================================
    # MODE 1 — SINGLE WORD DRIFT
    # ===========================================================
    if mode == "Single Word Drift":
        st.markdown("## 🔍 Single Word Semantic Drift Analysis")

        target_word = st.text_input("Enter a word to analyze:", "economy")

        analyze_btn = st.button("Analyze")

        if analyze_btn:
            aligned, vectors, vyears = get_aligned_embeddings(models, target_word, years)

            # --------------------------------------------
            # VALIDATION
            # --------------------------------------------
            if vectors is None or len(vectors) == 0:
                st.error(f"❌ The word '{target_word}' does not appear enough times.")
                return

            count = len(vectors)
            if count == 1:
                st.warning(f"ℹ️ The word '{target_word}' appears in **1 year only**.")
            else:
                st.success(f"✅ Found '{target_word}' in **{count} different years**.")

            # --------------------------------------------
            # TABS
            # --------------------------------------------
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Drift Plot", 
                "🌐 3D Trajectory",
                "🔥 Similarity Matrix",
                "📊 Statistics"
            ])

            # --------------------------------------------------
            # TAB 1 — Drift
            # --------------------------------------------------
            with tab1:
                fig1 = plot_drift(vectors, vyears, target_word)
                st.pyplot(fig1)

            # --------------------------------------------------
            # TAB 2 — 3D
            # --------------------------------------------------
            with tab2:
                fig2 = plot_3d_trajectory(vectors, vyears, target_word)
                if fig2:
                    st.pyplot(fig2)
                else:
                    st.warning("⚠️ Not enough data for 3D PCA trajectory.")

            # --------------------------------------------------
            # TAB 3 — Similarity Matrix
            # --------------------------------------------------
            with tab3:
                fig3 = plot_similarity_matrix(vectors, vyears, target_word)
                if fig3:
                    st.pyplot(fig3)
                else:
                    st.warning("⚠️ Need at least 2 years to compute similarity.")

            # --------------------------------------------------
            # TAB 4 — Stats
            # --------------------------------------------------
            with tab4:
                df_stats = show_stats(target_word, vyears, vectors)
                st.dataframe(df_stats, use_container_width=True)

    # ===========================================================
    # MODE 2 — SEMANTIC NETWORK
    # ===========================================================
    if mode == "Semantic Network":
        st.markdown("## 🔗 Semantic Network Explorer")

        target_word = st.text_input("Word:", "economy")
        year_choice = st.selectbox("Select year:", years)

        if st.button("Generate Network"):
            aligned, vectors, vyears = get_aligned_embeddings(models, target_word, years)

            if not aligned or year_choice not in aligned:
                st.error("❌ Word not available in this year or cannot be aligned.")
                return

            figN = plot_semantic_network(aligned, year_choice, target_word, topn=12)

            if figN:
                st.pyplot(figN)
            else:
                st.error("⚠️ Could not build network. Not enough neighbors.")


# ===========================================================
# PART 4 — MULTI-WORD COMPARISON MODULE
# ===========================================================

def plot_multi_word_drift(models, words, years):
    """Compare drift for several words across years."""
    plt.figure(figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(words)))

    for idx, word in enumerate(words):
        aligned, vectors, yrs = get_aligned_embeddings(models, word, years)
        if vectors is None or len(vectors) < 2:
            continue

        base_vec = vectors[0]
        drift_scores = [cosine(base_vec, v) for v in vectors]

        plt.plot(
            yrs,
            drift_scores,
            marker='o',
            linewidth=3,
            markersize=8,
            color=colors[idx],
            label=word
        )

    plt.title("Multi-Word Semantic Drift Comparison", fontsize=18)
    plt.xlabel("Year")
    plt.ylabel("Cosine Distance from Base")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)
    return plt.gcf()


# ===========================================================
# PART 5 — WORD-TO-WORD DISTANCE EVOLUTION
# ===========================================================

def plot_word_distance(models, word1, word2, years):
    aligned1, vectors1, yrs1 = get_aligned_embeddings(models, word1, years)
    aligned2, vectors2, yrs2 = get_aligned_embeddings(models, word2, years)

    if vectors1 is None or vectors2 is None:
        return None, None, None

    common_years = sorted(list(set(yrs1) & set(yrs2)))
    if len(common_years) < 2:
        return None, None, None

    distances = []
    for yr in common_years:
        v1 = aligned1[yr][word1]
        v2 = aligned2[yr][word2]
        distances.append(cosine(v1, v2))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        common_years,
        distances,
        marker='o',
        linewidth=3,
        markersize=9,
        color="#E63946"
    )

    ax.set_title(f"Distance Between '{word1}' and '{word2}' Over Time", fontsize=17)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine Distance")
    ax.grid(True, linestyle="--", alpha=0.3)

    stats = {
        "Mean": float(np.mean(distances)),
        "Min": float(np.min(distances)),
        "Max": float(np.max(distances)),
        "Std": float(np.std(distances))
    }

    return fig, stats, common_years


# ===========================================================
# PART 6 — DASHBOARD COMBINED VIEW
# ===========================================================

def dashboard_view(models, word, years):
    """One-page dashboard: drift + PCA + similarity + neighbors"""
    aligned, vectors, yrs = get_aligned_embeddings(models, word, years)
    if vectors is None or len(vectors) == 0:
        return None

    fig_drift = plot_drift(vectors, yrs, word)
    fig_pca = plot_3d_trajectory(vectors, yrs, word)
    fig_sim = plot_similarity_matrix(vectors, yrs, word)

    neighbors_by_year = {}
    for yr in yrs:
        neighbors_by_year[yr] = nearest_neighbors(aligned, yr, word, 8)

    return fig_drift, fig_pca, fig_sim, neighbors_by_year


# ===========================================================
# PART 7 — EXTRA UI + STYLE BOOST
# ===========================================================

def apply_custom_styles():
    """Apply custom CSS styles for the extended UI"""
    st.markdown("""
    <style>
    .big-header {
        font-size: 38px;
        text-align: center;
        color: #145DA0;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 26px;
        margin-top: 35px;
        margin-bottom: 15px;
        font-weight: bold;
        color: #0C2D48;
    }
    .result-box {
        background: #f2f6fa;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #0C2D48;
        margin-bottom: 18px;
    }
    </style>
    """, unsafe_allow_html=True)


# ===========================================================
# PART 8 — EXTENDED MAIN UI
# ===========================================================

def main_extended():
    apply_custom_styles()
    st.markdown("<div class='big-header'>🔮 FULL EXTENDED SEMANTIC SHIFT SUITE</div>", unsafe_allow_html=True)

    st.sidebar.header("📁 Upload Corpus")
    file = st.sidebar.file_uploader("Upload TXT / CSV / XLSX", type=['txt', 'csv', 'xlsx'])

    if not file:
        st.info("⬆️ Please upload a corpus file.")
        return

    year_to_text = load_corpus_from_file(file)
    if not year_to_text:
        return

    years = sorted(year_to_text.keys())
    year_to_tokens = tokenize_corpus(year_to_text)

    models_key = f"models_{file.name}_{len(years)}"
    if models_key not in st.session_state:
        with st.spinner("⚙️ Training models…"):
            st.session_state[models_key] = train_models(year_to_tokens)

    models = st.session_state[models_key]

    st.sidebar.header("📌 SELECT MODE")
    mode = st.sidebar.radio("", [
        "Single Word Drift",
        "Semantic Network",
        "Multi-Word Comparison",
        "Word-to-Word Distance",
        "Dashboard (All-in-One)"
    ])


    # ========================
    # MODE: MULTI-WORD
    # ========================
    if mode == "Multi-Word Comparison":
        st.markdown("<div class='section-title'>📊 Multi-Word Semantic Drift</div>", unsafe_allow_html=True)

        words_input = st.text_input("Enter words (comma-separated):", "economy, crisis, war")
        words = [w.strip() for w in words_input.split(",") if w.strip()]

        if st.button("Compare"):
            fig = plot_multi_word_drift(models, words, years)
            st.pyplot(fig)


    # ========================
    # MODE: WORD-TO-WORD
    # ========================
    if mode == "Word-to-Word Distance":
        st.markdown("<div class='section-title'>🔗 Word-to-Word Distance</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            w1 = st.text_input("Word 1", "economy")
        with col2:
            w2 = st.text_input("Word 2", "inflation")

        if st.button("Compute Distance"):
            fig, stats, yrs = plot_word_distance(models, w1, w2, years)
            if fig:
                st.pyplot(fig)
                st.markdown("<div class='section-title'>📊 Statistics</div>", unsafe_allow_html=True)
                st.write(pd.DataFrame([stats]))
            else:
                st.error("Not enough overlapping years.")


    # ========================
    # MODE: DASHBOARD
    # ========================
    if mode == "Dashboard (All-in-One)":
        st.markdown("<div class='section-title'>📌 Dashboard Overview</div>", unsafe_allow_html=True)

        word = st.text_input("Enter word:", "economy")

        if st.button("Generate Dashboard"):
            result = dashboard_view(models, word, years)
            if not result:
                st.error("Word missing in corpus.")
                return

            fig1, fig2, fig3, neighbors = result

            st.subheader("📈 Drift")
            st.pyplot(fig1)

            st.subheader("🌐 3D PCA Trajectory")
            if fig2:
                st.pyplot(fig2)
            else:
                st.warning("Not enough years for 3D PCA.")

            st.subheader("🔥 Similarity Matrix")
            st.pyplot(fig3)

            st.subheader("📌 Top Neighbors per Year")
            for yr, neigh in neighbors.items():
                st.markdown(f"### {yr}")
                df = pd.DataFrame(neigh, columns=["Neighbor", "Similarity"])
                st.dataframe(df)


# ===========================================================
# RUN APP
# ===========================================================

if __name__ == "__main__":
    # Choose which main function to run:
    # - main() for basic UI with Single Word Drift & Semantic Network
    # - main_extended() for full suite with Multi-Word, Word-to-Word, Dashboard
    
    main_extended()  # Using extended version by default
