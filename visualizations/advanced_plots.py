"""
Advanced visualization functions for semantic drift analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec

from visualizations.basic_plots import nearest_neighbors


def plot_enhanced_drift_with_statistics(word, models, years, global_vocab):
    """Enhanced drift plot with statistics and annotations"""
    vectors = []
    valid_years = []
    
    for yr in years:
        if yr in models and word in models[yr].wv:
            vectors.append(models[yr].wv[word])
            valid_years.append(yr)
    
    if len(vectors) < 2:
        st.error(f"Insufficient data for '{word}' - found in {len(vectors)} year(s)")
        return
    
    base_vec = vectors[0]
    drift_scores = [cosine(base_vec, v) for v in vectors]
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main drift plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(valid_years, drift_scores, marker="o", linewidth=3, markersize=10,
             color='#2E86AB', label='Cosine Distance')
    
    if len(valid_years) >= 3:
        z = np.polyfit(range(len(valid_years)), drift_scores, min(2, len(valid_years)-1))
        p = np.poly1d(z)
        ax1.plot(valid_years, p(range(len(valid_years))), "--",
                 linewidth=2, color='#A23B72', alpha=0.7, label='Polynomial Trend')
    
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
    
    ax1.set_title(f"Semantic Drift of '{word}' Over Time", fontsize=18, fontweight='bold')
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
    vectors = []
    valid_years = []
    
    for yr in years:
        if yr in models and word in models[yr].wv:
            vectors.append(models[yr].wv[word])
            valid_years.append(yr)
    
    if len(vectors) < 3:
        st.error(f"Need at least 3 years for 3D visualization - found {len(vectors)}")
        return
    
    vectors_array = np.array(vectors)
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors_array)
    
    fig = plt.figure(figsize=(14, 10))
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
                str(year), fontsize=10, fontweight='bold')
    
    ax.scatter([vectors_3d[0, 0]], [vectors_3d[0, 1]], [vectors_3d[0, 2]],
              color='green', s=400, marker='*', edgecolor='black', linewidth=2, label='Start')
    ax.scatter([vectors_3d[-1, 0]], [vectors_3d[-1, 1]], [vectors_3d[-1, 2]],
              color='red', s=400, marker='*', edgecolor='black', linewidth=2, label='End')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12)
    ax.set_title(f"3D Semantic Trajectory of '{word}' Over Time", fontsize=16, fontweight='bold')
    
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
    
    dim_variance = np.var(vectors_array, axis=0)
    top_dims = np.argsort(dim_variance)[-50:]
    
    vectors_subset = vectors_array[:, top_dims]
    
    im1 = axes[0].imshow(vectors_subset.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[0].set_yticks(range(0, 50, 5))
    axes[0].set_yticklabels([f'Dim {d}' for d in top_dims[::5]])
    axes[0].set_xticks(range(len(valid_years)))
    axes[0].set_xticklabels(valid_years, rotation=45, ha='right')
    axes[0].set_title(f"Top 50 Most Variable Dimensions for '{word}'", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Dimension', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Embedding Value')
    
    if len(vectors_subset) > 1:
        changes = np.diff(vectors_subset, axis=0)
        im2 = axes[1].imshow(changes.T, aspect='auto', cmap='seismic',
                             interpolation='nearest', vmin=-np.max(np.abs(changes)),
                             vmax=np.max(np.abs(changes)))
        axes[1].set_yticks(range(0, 50, 5))
        axes[1].set_yticklabels([f'Dim {d}' for d in top_dims[::5]])
        axes[1].set_xticks(range(len(valid_years)-1))
        axes[1].set_xticklabels([f"{valid_years[i]}-{valid_years[i+1]}" for i in range(len(valid_years)-1)],
                                rotation=45, ha='right')
        axes[1].set_title(f"Year-over-Year Dimensional Changes", fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Period', fontsize=12)
        axes[1].set_ylabel('Dimension', fontsize=12)
        plt.colorbar(im2, ax=axes[1], label='Change in Value')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_similarity_matrix(word, models, years, global_vocab):
    """Cross-year similarity matrix"""
    vectors = []
    valid_years = []
    
    for yr in years:
        if yr in models and word in models[yr].wv:
            vectors.append(models[yr].wv[word])
            valid_years.append(yr)
    
    if len(vectors) < 2:
        st.error(f"Insufficient data for similarity matrix - found {len(vectors)} year(s)")
        return
    
    n_years = len(valid_years)
    similarity_matrix = np.zeros((n_years, n_years))
    
    for i in range(n_years):
        for j in range(n_years):
            similarity_matrix[i, j] = 1 - cosine(vectors[i], vectors[j])
    
    fig, ax = plt.subplots(figsize=(12, 10))
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
    ax.set_title(f"Cross-Temporal Semantic Similarity Matrix for '{word}'",
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def visualize_enhanced_semantic_network(year, word, model, global_vocab, topn=15):
    """Enhanced semantic network with community detection"""
    if word not in model.wv:
        st.error(f"'{word}' not found in year {year}")
        return
    
    neighbors = nearest_neighbors(year, word, model, topn=topn)
    
    G = nx.Graph()
    G.add_node(word, node_type='target')
    
    for w, sim in neighbors:
        G.add_node(w, node_type='neighbor')
        G.add_edge(word, w, weight=sim)
    
    communities = list(nx.community.greedy_modularity_communities(G))
    
    color_map = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    node_to_community = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = idx
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Spring layout
    pos = nx.spring_layout(G, seed=42, k=2)
    
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
    
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax1)
    
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in edge_weights],
                          alpha=0.6, edge_color='gray', ax=ax1)
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    top_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)[:5]
    top_edge_labels = {(u, v): edge_labels[(u, v)] for u, v, _ in top_edges}
    nx.draw_networkx_edge_labels(G, pos, top_edge_labels, font_size=9, ax=ax1)
    
    ax1.set_title(f"Semantic Network for '{word}' ({year})\nSpring Layout",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Circular layout
    pos_circular = nx.circular_layout(G)
    
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
    
    nx.draw_networkx_labels(G, pos_circular, font_size=11, font_weight='bold', ax=ax2)
    
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
    
    for year in years_to_compare:
        if year in models and word in models[year].wv:
            neighbors = nearest_neighbors(year, word, models[year], topn=topn)
            year_neighbors[year] = {w: sim for w, sim in neighbors if w != word}
            all_neighbors.update(year_neighbors[year].keys())
    
    if not year_neighbors:
        st.error(f"No data found for '{word}' in selected years")
        return
    
    all_neighbors = sorted(all_neighbors)
    
    heatmap_data = np.zeros((len(all_neighbors), len(years_to_compare)))
    for j, year in enumerate(years_to_compare):
        if year in year_neighbors:
            for i, neighbor in enumerate(all_neighbors):
                if neighbor in year_neighbors[year]:
                    heatmap_data[i, j] = year_neighbors[year][neighbor]
    
    n_years = len(years_to_compare)
    fig, axes = plt.subplots(n_years, 1, figsize=(14, 5*n_years))
    
    if n_years == 1:
        axes = [axes]
    
    for idx, year in enumerate(years_to_compare):
        if year in year_neighbors:
            neighbors = sorted(year_neighbors[year].items(), key=lambda x: x[1], reverse=True)
            words = [w for w, _ in neighbors]
            similarities = [s for _, s in neighbors]
            
            bars = axes[idx].barh(range(len(words)), similarities,
                                 color=plt.cm.viridis(similarities),
                                 edgecolor='black', linewidth=1.5)
            
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words, fontsize=11)
            axes[idx].set_xlabel('Similarity Score', fontsize=12)
            axes[idx].set_title(f"Top Neighbors of '{word}' in {year}", fontsize=13, fontweight='bold')
            axes[idx].set_xlim(0, 1)
            axes[idx].grid(axis='x', alpha=0.3)
            
            for i, (bar, sim) in enumerate(zip(bars, similarities)):
                axes[idx].text(sim + 0.01, i, f'{sim:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    if len(years_to_compare) > 1:
        fig, ax = plt.subplots(figsize=(12, max(8, len(all_neighbors)*0.3)))
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        ax.set_xticks(range(len(years_to_compare)))
        ax.set_xticklabels(years_to_compare)
        ax.set_yticks(range(len(all_neighbors)))
        ax.set_yticklabels(all_neighbors, fontsize=10)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Neighbor Words', fontsize=12)
        ax.set_title(f"Temporal Evolution of Semantic Neighbors for '{word}'",
                    fontsize=14, fontweight='bold', pad=20)
        
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
        
        base_vec = word_vectors[0]
        drift = [cosine(base_vec, v) for v in word_vectors]
        
        all_drift_data[word] = {'years': word_years, 'drift': drift}
        
        ax1.plot(word_years, drift, marker='o', linewidth=2.5,
                markersize=8, color=colors[idx], label=word, alpha=0.8)
    
    if not all_drift_data:
        st.error("No valid words found for comparison")
        plt.close()
        return
    
    ax1.set_xlabel('Year', fontsize=13)
    ax1.set_ylabel('Cosine Distance from Base Year', fontsize=13)
    ax1.set_title('Comparative Semantic Drift Analysis', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
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
    ax2.set_title('Total Semantic Shift by Word', fontsize=15, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (bar, drift) in enumerate(zip(bars, drifts_sorted)):
        ax2.text(drift + (0.01 if drift > 0 else -0.01), i, f'{drift:.3f}',
                va='center', ha='left' if drift > 0 else 'right',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
