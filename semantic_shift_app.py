"""
Semantic Shift Analysis - Interactive Web Application
Analyzes semantic drift in text corpora using improved Word2Vec methodology
with lemmatization, vocabulary alignment, and stability enhancements

Main entry point for Streamlit app
"""

import streamlit as st
import warnings
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# Import configuration
from config import PAGE_CONFIG, CUSTOM_CSS

# Import data processing
from data_loader import (
    load_corpus_from_file,
    tokenize_corpus,
    build_global_vocabulary,
    lemmatizer
)

# Import model training
from model_trainer import (
    train_stable_models,
    compute_drift_score_robust
)

# Import visualizations
from visualizations import (
    plot_drift_robust,
    plot_semantic_network_robust,
    plot_enhanced_drift_with_statistics,
    plot_3d_semantic_trajectory,
    plot_temporal_heatmap,
    plot_similarity_matrix,
    visualize_enhanced_semantic_network,
    visualize_neighbor_evolution,
    compare_multiple_words
)

# Set page config
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header">📊 𝐓𝐡𝐞 𝐒𝐞𝐦𝐚𝐧𝐭𝐫𝐢𝐟𝐭: Semantic Shift Analyzer</p>', unsafe_allow_html=True)
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
        _show_welcome_message()
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
        
        # Create analysis tabs
        _create_analysis_tabs(models, years, global_vocab, word_to_years, word_to_total_count, min_years, min_total_count)
        
        # Vocabulary explorer
        _create_vocabulary_explorer(global_vocab)
    
    # About section
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
    - Advanced visualizations
    
    Built with Streamlit 🎈
    """)


def _show_welcome_message():
    """Display welcome message and instructions"""
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


def _create_analysis_tabs(models, years, global_vocab, word_to_years, word_to_total_count, min_years, min_total_count):
    """Create and populate all analysis tabs"""
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
        _tab_single_word_drift(models, years, global_vocab, word_to_years, word_to_total_count, min_years, min_total_count)
    
    with tab2:
        _tab_word_to_word_distance(models, years, global_vocab)
    
    with tab3:
        _tab_semantic_network(models, years, global_vocab)
    
    with tab4:
        _tab_multi_word_comparison(models, years, global_vocab)
    
    with tab5:
        _tab_enhanced_drift(models, years, global_vocab)
    
    with tab6:
        _tab_3d_trajectory(models, years, global_vocab)
    
    with tab7:
        _tab_temporal_heatmap(models, years, global_vocab)
    
    with tab8:
        _tab_enhanced_networks(models, years, global_vocab)


def _tab_single_word_drift(models, years, global_vocab, word_to_years, word_to_total_count, min_years, min_total_count):
    """Tab 1: Single Word Semantic Drift"""
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
    
    if word:
        lemmatized = lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n')
        
        if word not in global_vocab and lemmatized not in global_vocab:
            word_in_any_year = word in word_to_years or lemmatized in word_to_years
            
            error_msg = f'❌ Word "{word}" not in global vocabulary.\n\n'
            
            if lemmatized != word:
                error_msg += f'💡 **Lemmatization:** "{word}" → "{lemmatized}"\n\n'
            
            if word_in_any_year:
                actual_word = word if word in word_to_years else lemmatized
                years_with_word = len(word_to_years.get(actual_word, set()))
                total_occurrences = word_to_total_count.get(actual_word, 0)
                
                error_msg += f'**Word found but filtered out:**\n'
                error_msg += f'- Appears in {years_with_word} year(s) (need {min_years}+)\n'
                error_msg += f'- Total occurrences: {total_occurrences} (need {min_total_count}+)\n\n'
                
                error_msg += f'**Solution:**\n'
                if years_with_word < min_years:
                    error_msg += f'- Lower "Min Years" to {years_with_word} in Advanced Settings\n'
                if total_occurrences < min_total_count:
                    error_msg += f'- Lower "Min Total Occurrences" to {total_occurrences} in Advanced Settings\n'
                error_msg += f'- Then retrain the models (cache will be cleared)'
            else:
                error_msg += f'**Word not found in corpus**\n'
                error_msg += f'- Try searching vocabulary list below to see what\'s available\n'
                error_msg += f'- Try synonyms or related words'
            
            st.error(error_msg)
            
            similar_words = [w for w in sorted(global_vocab) if word[:3] in w or lemmatized[:3] in w]
            if similar_words:
                st.info(f"**Similar words in vocabulary:** {', '.join(similar_words[:15])}")
        else:
            word_to_use = word if word in global_vocab else lemmatized
            
            if word_to_use != word:
                st.info(f'💡 Using lemmatized form: "{word}" → "{word_to_use}"')
            
            years_present = word_to_years.get(word_to_use, set())
            total_count = word_to_total_count.get(word_to_use, 0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Years Present", len(years_present))
            col2.metric("Total Occurrences", total_count)
            col3.metric("Years Available", ', '.join(map(str, sorted(years_present))))
            
            plot_drift_robust(word_to_use, models, years, global_vocab, metric)


def _tab_word_to_word_distance(models, years, global_vocab):
    """Tab 2: Word-to-Word Distance Evolution"""
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
        errors = []
        if word1 not in global_vocab:
            errors.append(f'"{word1}" not in global vocabulary')
        if word2 not in global_vocab:
            errors.append(f'"{word2}" not in global vocabulary')
        
        if errors:
            st.error("❌ " + " and ".join(errors))
        else:
            import matplotlib.pyplot as plt
            
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


def _tab_semantic_network(models, years, global_vocab):
    """Tab 3: Semantic Network Visualization"""
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


def _tab_multi_word_comparison(models, years, global_vocab):
    """Tab 4: Multi-Word Drift Comparison"""
    st.subheader("📊 Multi-Word Drift Comparison")
    st.markdown("Compare semantic drift across multiple words simultaneously.")
    
    words_compare = st.text_input(
        "Enter words to compare (comma-separated):",
        "freedom, liberty, democracy"
    )
    
    if words_compare:
        import matplotlib.pyplot as plt
        
        words_list = [w.strip().lower() for w in words_compare.split(',')]
        
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


def _tab_enhanced_drift(models, years, global_vocab):
    """Tab 5: Enhanced Drift Plot with Statistics"""
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


def _tab_3d_trajectory(models, years, global_vocab):
    """Tab 6: 3D Semantic Trajectory"""
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


def _tab_temporal_heatmap(models, years, global_vocab):
    """Tab 7: Temporal Analysis: Heatmap & Similarity Matrix"""
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


def _tab_enhanced_networks(models, years, global_vocab):
    """Tab 8: Enhanced Semantic Networks & Neighbor Evolution"""
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


def _create_vocabulary_explorer(global_vocab):
    """Create vocabulary explorer in sidebar"""
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
            st.text_area("Vocabulary preview", value=", ".join(all_words[:100]), 
                        height=200, label_visibility="hidden")


if __name__ == "__main__":
    main()
