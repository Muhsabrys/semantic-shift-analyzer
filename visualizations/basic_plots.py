"""
Basic visualization functions for semantic drift analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cosine

from model_trainer import compute_drift_score_robust


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


def nearest_neighbors(year, word, model, topn=10):
    """Get nearest neighbors for a word in a specific year"""
    if word not in model.wv:
        return []
    wv = model.wv[word]
    sims = {w: 1 - cosine(wv, model.wv[w]) for w in model.wv.index_to_key}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:topn]
