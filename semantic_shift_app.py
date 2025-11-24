import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Lightweight sentence splitter (no NLTK)
# ---------------------------------------------------------
def split_into_sentences(text):
    if not isinstance(text, str):
        text = str(text)
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]

# ---------------------------------------------------------
# 2. Lightweight lemmatizer (no spaCy)
# ---------------------------------------------------------
def simple_lemma(word):
    w = word.lower().strip()
    w = re.sub(r"[^a-z0-9'-]", "", w)

    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes"):
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w

# ---------------------------------------------------------
# 3. File Loader (universal: CSV, TXT, XLSX)
# ---------------------------------------------------------
def load_corpus(uploaded):
    ext = uploaded.name.split(".")[-1].lower()

    if ext == "csv":
        df = pd.read_csv(uploaded)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded, sep=None, engine="python", header=None)

    # Detect header
    first_cell = str(df.iloc[0, 0])
    if first_cell.isdigit():
        df.columns = ["year", "text"]
    else:
        df.columns = [c.lower().strip() for c in df.columns]
        df.rename(columns={df.columns[0]: "year", df.columns[1]: "text"}, inplace=True)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["text"] = df["text"].astype(str)

    return df[["year", "text"]]

# ---------------------------------------------------------
# 4. Embed each year using SBERT
# ---------------------------------------------------------
def compute_year_embeddings(df, model):
    year_embeddings = {}

    for yr in sorted(df["year"].unique()):
        year_text = " ".join(df[df["year"] == yr]["text"].astype(str).tolist())
        sentences = split_into_sentences(year_text)

        if len(sentences) == 0:
            continue

        embeddings = model.encode(sentences, show_progress_bar=False)
        year_embeddings[yr] = {
            "sentences": sentences,
            "embeddings": embeddings
        }

    return year_embeddings

# ---------------------------------------------------------
# 5. Extract vector for a target word across years
# ---------------------------------------------------------
def get_word_vectors(year_embeddings, target_word):
    lemma = simple_lemma(target_word)
    vectors = []
    valid_years = []

    for yr, data in year_embeddings.items():
        sents = data["sentences"]
        embeds = data["embeddings"]

        matches = []
        for i, sent in enumerate(sents):
            tokens = [simple_lemma(t) for t in sent.lower().split()]
            if lemma in tokens:
                matches.append(embeds[i])

        if len(matches) > 0:
            vec = np.mean(matches, axis=0)
            vectors.append(vec)
            valid_years.append(yr)

    if len(vectors) < 2:
        return None, None

    return vectors, valid_years

# ---------------------------------------------------------
# 6. Plot: Drift over time
# ---------------------------------------------------------
def plot_drift(vectors, years, word):
    base = vectors[0]
    drift = [cosine(base, v) for v in vectors]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(years, drift, marker="o", linewidth=3)
    ax.set_title(f"Semantic Drift of '{word}' Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine Distance from First Year")
    ax.grid(True)
    return fig

# ---------------------------------------------------------
# 7. Plot 2D PCA trajectory
# ---------------------------------------------------------
def plot_trajectory(vectors, years, word):
    arr = np.array(vectors)

    if len(arr) < 2:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.text(0.5,0.5,"Not enough data for PCA",ha="center")
        ax.axis("off")
        return fig

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(arr)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(reduced[:,0], reduced[:,1], "-o")
    for i, yr in enumerate(years):
        ax.text(reduced[i,0], reduced[i,1], str(yr))
    ax.set_title(f"2D PCA Semantic Trajectory of '{word}'")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    return fig

# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------
def main():
    st.title("📊 Semantic Shift Analyzer (SBERT – MiniLM-L6-v2)")
    st.write("Analyze how word meanings change over time using SBERT sentence embeddings.")

    uploaded = st.file_uploader("Upload CSV, TXT, or XLSX with two columns: year, text")

    if not uploaded:
        st.info("Please upload a file to begin.")
        return

    df = load_corpus(uploaded)
    st.success(f"Loaded {len(df)} rows, years {df['year'].min()}–{df['year'].max()}")

    # SBERT
    st.subheader("Embedding Model: all-MiniLM-L6-v2")
    with st.spinner("Loading SBERT model..."):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Embed
    with st.spinner("Embedding corpus..."):
        year_embeddings = compute_year_embeddings(df, model)

    # Analysis
    st.subheader("Analyze Word Drift")
    word = st.text_input("Enter word:", "crisis")

    if st.button("Analyze"):
        with st.spinner("Searching for word across years..."):
            vectors, years = get_word_vectors(year_embeddings, word)

        if vectors is None:
            st.error("Word not found in at least 2 different years.")
            return

        st.success(f"Found in {len(years)} years: {years}")

        tab1, tab2 = st.tabs(["📈 Drift", "🧭 Trajectory"])

        with tab1:
            st.pyplot(plot_drift(vectors, years, word))

        with tab2:
            st.pyplot(plot_trajectory(vectors, years, word))


if __name__ == "__main__":
    main()
