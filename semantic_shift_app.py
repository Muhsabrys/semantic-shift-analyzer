# ========================================================
# SEMANTIC SHIFT ANALYZER — CLEAN SBERT VERSION
# ========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import networkx as nx
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

import spacy
from spacy.lang.en import English
nlp = English()
nlp.add_pipe("lemmatizer", config={"mode": "lookup"})

# Ensure tokenizers + lemmatizer
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")  # for lemmatization

# ─────────────────────────────────────────────────────────────
# CLEAN + PARSE CORPUS
# ─────────────────────────────────────────────────────────────

def load_corpus(upload):
    ext = upload.name.split(".")[-1].lower()
    df = None

    if ext == "csv":
        df = pd.read_csv(upload, header=None)

    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(upload, header=None)

    elif ext == "txt":
        lines = upload.getvalue().decode("utf-8").split("\n")
        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            elif "," in line:
                parts = line.split(",", 1)
            else:
                continue
            rows.append(parts)
        df = pd.DataFrame(rows)

    # Expect first column = year, second column = text
    df.columns = ["year", "text"]

    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")

    return df


def clean_sentence(s):
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_into_sentences(text):
    sents = nltk.sent_tokenize(text)
    return [clean_sentence(s) for s in sents if len(s.strip()) > 0]


# ─────────────────────────────────────────────────────────────
# SBERT EMBEDDING PER YEAR
# ─────────────────────────────────────────────────────────────

def compute_year_embeddings(df, model):
    year_embeddings = {}

    for yr in df["year"].unique():
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


# ─────────────────────────────────────────────────────────────
# WORD-VECTOR EXTRACTION (LEMMA MATCH)
# ─────────────────────────────────────────────────────────────

def sentence_contains_word(sentence, target_lemma):
    """
    Lemma-aware word detection using spaCy.
    """
    doc = nlp(sentence.lower())
    lemmas = [t.lemma_ for t in doc]
    return target_lemma in lemmas


def get_word_vectors(year_embeddings, target_word):
    """
    For each year, find all sentences whose *lemmas* contain the target word,
    and average their embeddings.
    """
    target_word = target_word.lower()
    target_lemma = nlp(target_word)[0].lemma_

    vectors = []
    valid_years = []

    for yr, data in year_embeddings.items():
        sents = data["sentences"]
        embeds = data["embeddings"]

        idx = []
        for i, s in enumerate(sents):
            if sentence_contains_word(s, target_lemma):
                idx.append(i)

        if len(idx) > 0:
            v = np.mean([embeds[i] for i in idx], axis=0)
            vectors.append(v)
            valid_years.append(yr)

    if len(vectors) < 2:
        return None, None

    return np.array(vectors), valid_years


# ─────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

def plot_drift(vectors, years, word):
    base = vectors[0]
    drift = [cosine(base, v) for v in vectors]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, drift, marker="o", linewidth=3)
    ax.set_title(f"Semantic Drift of '{word}'")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine distance from base year")
    ax.grid(True)
    return fig


def plot_similarity_matrix(vectors, years, word):
    n = len(vectors)
    sim = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sim[i, j] = 1 - cosine(vectors[i], vectors[j])

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(sim, cmap="YlOrRd")
    fig.colorbar(cax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(years)
    ax.set_yticklabels(years)
    ax.set_title(f"Cross-Year Similarity Matrix for '{word}'")

    return fig


def plot_3d_trajectory(vectors, years, word):
    if len(vectors) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data for 3D plot", ha="center")
        ax.axis("off")
        return fig

    pca = PCA(n_components=3)
    pts = pca.fit_transform(vectors)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o")
    for i, yr in enumerate(years):
        ax.text(pts[i, 0], pts[i, 1], pts[i, 2], str(yr))

    ax.set_title(f"3D Trajectory of '{word}'")
    return fig


def build_semantic_network(year_embeddings, year, target_word, topn=10):
    vectors, years = get_word_vectors(year_embeddings, target_word)
    if vectors is None:
        return None

    # Use that year's embedding
    yr_idx = years.index(year)
    v = vectors[yr_idx]

    # Compute similarity to all sentences
    sents = year_embeddings[year]["sentences"]
    embeds = year_embeddings[year]["embeddings"]

    sims = [(sents[i], 1 - cosine(v, embeds[i])) for i in range(len(sents))]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:topn]

    # Build network
    G = nx.Graph()
    G.add_node(target_word)

    for s, score in sims:
        G.add_node(s)
        G.add_edge(target_word, s, weight=score)

    return G


def plot_semantic_network(G, word):
    if G is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No meaningful neighbors for '{word}'", ha="center")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw(G, pos, with_labels=True, font_size=8, ax=ax,
            node_color="lightblue", edge_color="gray")
    ax.set_title(f"Semantic Network for '{word}'")

    return fig


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────

def main():
    st.title("📊 Semantic Shift Analyzer (SBERT Version)")

    uploaded = st.file_uploader("Upload corpus", type=["csv", "txt", "xlsx"])

    if not uploaded:
        st.info("Please upload a file to begin.")
        return

    df = load_corpus(uploaded)
    st.success(f"Loaded {len(df)} rows, years {df['year'].min()}–{df['year'].max()}")

    # SBERT model selector
    model_choice = st.selectbox(
        "Choose embedding model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    )
    model = SentenceTransformer(model_choice)

    # Embed per year
    with st.spinner("Embedding corpus..."):
        year_embeddings = compute_year_embeddings(df, model)

    mode = st.radio(
        "Select analysis mode",
        ["Single Word Drift", "Semantic Network", "Word-to-Word Distance", "Multi-Word Comparison"]
    )

    # ───── SINGLE WORD DRIFT ─────

    if mode == "Single Word Drift":
        word = st.text_input("Enter word:", "crisis")

        if st.button("Analyze"):
            vectors, years = get_word_vectors(year_embeddings, word)

            if vectors is None:
                st.error("Not enough occurrences of this word across years.")
                return

            tab1, tab2, tab3 = st.tabs(["Drift Plot", "Similarity Matrix", "3D Trajectory"])

            with tab1:
                st.pyplot(plot_drift(vectors, years, word))

            with tab2:
                st.pyplot(plot_similarity_matrix(vectors, years, word))

            with tab3:
                st.pyplot(plot_3d_trajectory(vectors, years, word))

    # ───── SEMANTIC NETWORK ─────

    elif mode == "Semantic Network":
        word = st.text_input("Word:", "crisis")
        year = st.selectbox("Choose year:", sorted(year_embeddings.keys()))

        if st.button("Generate Network"):
            G = build_semantic_network(year_embeddings, year, word)
            st.pyplot(plot_semantic_network(G, word))

    # ───── WORD TO WORD DISTANCE ─────

    elif mode == "Word-to-Word Distance":
        w1 = st.text_input("Word 1:", "crisis")
        w2 = st.text_input("Word 2:", "economy")

        if st.button("Compare"):
            v1, y1 = get_word_vectors(year_embeddings, w1)
            v2, y2 = get_word_vectors(year_embeddings, w2)

            if v1 is None or v2 is None:
                st.error("One or both words missing.")
                return

            years = sorted(list(set(y1) & set(y2)))
            dists = []

            for yr in years:
                i = y1.index(yr)
                j = y2.index(yr)
                dists.append(cosine(v1[i], v2[j]))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(years, dists, marker="o")
            ax.set_title(f"Distance Between '{w1}' and '{w2}'")
            ax.set_xlabel("Year")
            ax.set_ylabel("Cosine distance")
            ax.grid(True)

            st.pyplot(fig)

    # ───── MULTI-WORD COMPARISON ─────

    elif mode == "Multi-Word Comparison":
        words = st.text_input("Words (comma-separated):", "crisis, war, peace").split(",")

        drift_data = {}

        for w in [x.strip() for x in words]:
            v, y = get_word_vectors(year_embeddings, w)
            if v is not None:
                base = v[0]
                drift = [cosine(base, vi) for vi in v]
                drift_data[w] = (y, drift)

        if st.button("Plot Comparison"):
            fig, ax = plt.subplots(figsize=(12, 6))

            for w, (years, drift) in drift_data.items():
                ax.plot(years, drift, marker="o", label=w)

            ax.set_title("Comparative Semantic Drift")
            ax.set_xlabel("Year")
            ax.set_ylabel("Drift")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)


if __name__ == "__main__":
    main()
