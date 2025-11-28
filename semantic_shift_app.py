"""
Semantic Shift Analysis - Main Application with Precomputed Embeddings Support
This file extends the original semantic_shift_app.py with precomputed corpus support
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import everything from the original drify.py since it has all the functions
import sys
import importlib.util

# Load drify module
spec = importlib.util.spec_from_file_location("drify", "drify.py")
drify = importlib.util.module_from_spec(spec)
sys.modules["drify"] = drify
spec.loader.exec_module(drify)

# Import precomputed embeddings support
from data_loader_enhanced import (
    load_precomputed_corpus,
    get_precomputed_word_stats
)

# Use functions from drify
load_corpus_from_file = drify.load_corpus_from_file
tokenize_corpus = drify.tokenize_corpus
build_global_vocabulary = drify.build_global_vocabulary
train_stable_models = drify.train_stable_models
lemmatizer = drify.lemmatizer

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
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header"> 🕰️🔍 The Semantrift — Semantic Shift Analyzer 💬📈 </p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"> “You shall know a word by the company it keeps (Firth, 1957)." </p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">The State of the Union model takes some time to load .. please be patient 😊 </p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    st.sidebar.subheader("📊 Choose Data Source")
    data_source = st.sidebar.radio(
        "Select corpus:",
        ["Upload Your Own File", "Use Precomputed State of the Union Corpus"],
        help="Upload your own data or explore our precomputed State of the Union embeddings"
    )
    
    models = None
    years = None
    global_vocab = None
    word_to_years = None
    word_to_total_count = None
    min_years = 2
    min_total_count = 3
    
    if data_source == "Upload Your Own File":
        # Original upload flow
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
            vector_size = st.slider("Vector Size", 50, 300, 200, 50)
            window_size = st.slider("Context Window", 3, 10, 5, 1)
            min_years = st.slider("Min Years for Vocabulary", 2, 5, 2, 1)
            min_total_count = st.slider("Min Total Occurrences", 2, 10, 3, 1)
            n_seeds = st.slider("Training Seeds", 1, 10, 5, 1)
        
        if uploaded_file is not None:
            # Process uploaded file
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
                
                # Process corpus
                with st.spinner("Processing corpus with lemmatization..."):
                    year_to_tokens = tokenize_corpus(year_to_text)
                
                # Build global vocabulary
                with st.spinner("Building global vocabulary..."):
                    global_vocab, word_to_years, word_to_total_count = build_global_vocabulary(
                        year_to_tokens, min_years=min_years, min_total_count=min_total_count
                    )
                
                st.sidebar.success(f"✅ Global vocabulary: {len(global_vocab):,} words")
                
                # Train models
                cache_key = f"models_{uploaded_file.name}_{len(years)}_{vector_size}_{n_seeds}"
                if cache_key not in st.session_state:
                    with st.spinner(f"Training models with {n_seeds} seeds for stability..."):
                        st.session_state[cache_key] = train_stable_models(
                            year_to_tokens, 
                            global_vocab,
                            n_seeds=n_seeds,
                            vector_size=vector_size,
                            window=window_size
                        )
                    st.sidebar.success(f"✅ Trained {len(st.session_state[cache_key])} models")
                
                models = st.session_state[cache_key]
        else:
            st.info("👆 Please upload a corpus file to begin analysis")
            show_welcome_message()
            return
    
    else:  # Precomputed corpus
        st.sidebar.markdown("""
        ### 📜 State of the Union Corpus
        
        This precomputed corpus contains embeddings from historical 
        State of the Union addresses, allowing you to explore semantic 
        shifts in American political discourse over time.
        
        **Features:**
        - Pre-trained embeddings (no training time!)
        - Multiple decades of data
        - High-quality historical corpus
        - Ready to analyze immediately
        """)
        
        # Use session state to persist the loaded corpus
        if 'precomputed_loaded' not in st.session_state:
            st.session_state.precomputed_loaded = False
        
        if st.sidebar.button("🚀 Load State of the Union Corpus", type="primary") or st.session_state.precomputed_loaded:
            if not st.session_state.precomputed_loaded:
                models, years, global_vocab, metadata = load_precomputed_corpus()
                
                if models is not None:
                    # Store in session state
                    st.session_state.precomputed_models = models
                    st.session_state.precomputed_years = years
                    st.session_state.precomputed_vocab = global_vocab
                    st.session_state.precomputed_metadata = metadata
                    st.session_state.precomputed_loaded = True
                else:
                    st.error("Failed to load precomputed corpus. Please try uploading your own file instead.")
                    return
            
            # Retrieve from session state
            models = st.session_state.precomputed_models
            years = st.session_state.precomputed_years
            global_vocab = st.session_state.precomputed_vocab
            metadata = st.session_state.precomputed_metadata
            
            # Generate word statistics
            word_to_years, word_to_total_count = get_precomputed_word_stats(global_vocab, models)
            
            # Show corpus info
            st.sidebar.success(f"✅ Loaded {len(years)} years ({min(years)}-{max(years)})")
            st.sidebar.metric("Vocabulary Size", f"{len(global_vocab):,}")
            st.sidebar.metric("Embedding Dimension", metadata.get('embedding_dimension', 'N/A'))
            
            # Show metadata
            with st.sidebar.expander("📖 Corpus Information"):
                st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
                st.write(f"**Description:** {metadata.get('description', 'N/A')}")
                st.write(f"**Year Range:** {metadata.get('years', 'N/A')}")
        else:
            st.info("👆 Click the button above to load the State of the Union corpus")
            show_precomputed_info()
            return
    
    # Main analysis section (common to both paths)
    if models is not None and years is not None and global_vocab is not None:
        # Call the original analysis from drify.py
        # We'll recreate the tabs here
        create_analysis_tabs(models, years, global_vocab, word_to_years, word_to_total_count, min_years, min_total_count)
        
        # Vocabulary explorer
        create_vocabulary_explorer(global_vocab)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info("""
    **The Semantrift: Semantic Shift Analyzer**
    
    Features:
    - Upload your own corpus OR
    - Explore precomputed State of the Union data
    - Lemmatization for vocabulary stability
    - Global vocabulary filtering
    - Multi-seed training for robustness
    - Advanced visualizations
    
    Built with Streamlit 🎈
    """)


def show_welcome_message():
    """Display welcome message and instructions"""
    st.markdown("""
    ### 🎯 WELCOME TO THE SEMANTRIFT!
    
    **Two ways to explore semantic drift:**
    
    1. **📁 Upload Your Own Corpus** - Bring your own time-series text data
    2. **📜 Use Precomputed Embeddings** - Explore State of the Union addresses instantly
    
    ### 📝 File Format for Upload:
    
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
    - **Lemmatization** automatically groups word variants
    """)


def show_precomputed_info():
    """Show information about precomputed corpus"""
    st.markdown("""
    ### 📜 About the State of the Union Corpus
    
    The State of the Union address is an annual message delivered by the President 
    of the United States to a joint session of Congress. This corpus contains 
    embeddings pre-trained on historical addresses, allowing you to:
    
    - **Track political language evolution** across decades
    - **Explore semantic shifts** in key terms (democracy, freedom, economy, etc.)
    - **Analyze historical context** through word embeddings
    - **No training time** - embeddings are pre-computed and ready to use
    
    ### 🔍 Example Analyses You Can Perform:
    
    - How has the meaning of "**freedom**" changed over time?
    - What words were semantically closest to "**war**" in different eras?
    - How did "**economy**" relate to other concepts across decades?
    - Compare drift patterns between multiple political terms
    
    ### 🚀 Ready to Explore?
    
    Click the **"Load State of the Union Corpus"** button in the sidebar to begin!
    """)


def create_analysis_tabs(models, years, global_vocab, word_to_years, word_to_total_count, min_years, min_total_count):
    """Create analysis tabs using functions from drify.py"""
    # Import visualization functions
    from drify import (
        plot_drift_robust,
        plot_semantic_network_robust,
        plot_enhanced_drift_with_statistics,
        plot_3d_semantic_trajectory,
        plot_temporal_heatmap,
        plot_similarity_matrix,
        visualize_enhanced_semantic_network,
        visualize_neighbor_evolution,
        compare_multiple_words,
        compute_drift_score_robust
    )
    
    import matplotlib.pyplot as plt
    
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
        st.markdown("Track how a word's meaning shifts over time relative to a baseline year.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            word = st.text_input("Enter a word to analyze:", "government").lower()
        with col2:
            metric = st.selectbox("Distance metric:", ["cosine", "euclidean"])
        
        if word and word in global_vocab:
            plot_drift_robust(word, models, years, global_vocab, metric)
        elif word:
            st.error(f"❌ '{word}' not in global vocabulary")
            similar_words = [w for w in sorted(global_vocab) if word[:3] in w]
            if similar_words:
                st.info(f"**Similar words in vocabulary:** {', '.join(similar_words[:15])}")
    
    with tab2:
        st.subheader("📏 Word-to-Word Distance Evolution")
        
        col1, col2 = st.columns(2)
        with col1:
            word1 = st.text_input("First word:", "economy").lower()
        with col2:
            word2 = st.text_input("Second word:", "jobs").lower()
        
        if word1 and word2 and word1 in global_vocab and word2 in global_vocab:
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
                ax.plot(valid_years_pair, distances, marker='o', linewidth=2, markersize=8, color='#2ca02c')
                ax.set_xlabel('Year', fontsize=12)
                ax.set_ylabel('Semantic Distance', fontsize=12)
                ax.set_title(f'Distance Evolution: "{word1}" ↔ "{word2}"', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
        elif word1 or word2:
            if word1 and word1 not in global_vocab:
                st.error(f"❌ '{word1}' not in global vocabulary")
            if word2 and word2 not in global_vocab:
                st.error(f"❌ '{word2}' not in global vocabulary")
    
    with tab3:
        st.subheader("🕸️ Semantic Network Visualization")
        
        year_select = st.selectbox("Select year:", years)
        words_input = st.text_input("Enter words (comma-separated):", "economy, government, people, nation, freedom")
        threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.7, 0.05)
        
        if words_input:
            words_list = [w.strip().lower() for w in words_input.split(',')]
            plot_semantic_network_robust(words_list, year_select, models, global_vocab, threshold)
    
    with tab4:
        st.subheader("📊 Multi-Word Drift Comparison")
        
        words_compare = st.text_input("Enter words to compare (comma-separated):", "freedom, liberty, democracy")
        
        if words_compare:
            words_list = [w.strip().lower() for w in words_compare.split(',')]
            compare_multiple_words(words_list, models, years, global_vocab)
    
    with tab5:
        st.subheader("📈 Enhanced Drift Plot with Statistics")
        
        word_enhanced = st.text_input("Enter word for enhanced analysis:", "government", key="enhanced_word").lower()
        
        if word_enhanced and word_enhanced in global_vocab:
            plot_enhanced_drift_with_statistics(word_enhanced, models, years, global_vocab)
        elif word_enhanced:
            st.error(f"❌ '{word_enhanced}' not in global vocabulary")
    
    with tab6:
        st.subheader("🎯 3D Semantic Trajectory")
        
        word_3d = st.text_input("Enter word for 3D visualization:", "crisis", key="3d_word").lower()
        
        if word_3d and word_3d in global_vocab:
            plot_3d_semantic_trajectory(word_3d, models, years, global_vocab)
        elif word_3d:
            st.error(f"❌ '{word_3d}' not in global vocabulary")
    
    with tab7:
        st.subheader("🔥 Temporal Analysis: Heatmap & Similarity Matrix")
        
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
        
        st.markdown("### Enhanced Semantic Network")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            word_network = st.text_input("Enter word for network:", "economy", key="network_word").lower()
        with col2:
            year_network = st.selectbox("Select year:", years, key="network_year")
        with col3:
            topn_network = st.slider("Number of neighbors:", 5, 25, 15, key="network_topn")
        
        if word_network and word_network in global_vocab and year_network in models:
            visualize_enhanced_semantic_network(year_network, word_network, models[year_network], global_vocab, topn_network)
        elif word_network and word_network not in global_vocab:
            st.error(f"❌ '{word_network}' not in global vocabulary")
        
        st.markdown("---")
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
        
        selected_years = st.multiselect("Select years to compare:", years, default=default_years, key="evolution_years")
        
        if word_evolution and word_evolution in global_vocab and selected_years:
            visualize_neighbor_evolution(word_evolution, models, selected_years, global_vocab, topn_evolution)
        elif word_evolution and word_evolution not in global_vocab:
            st.error(f"❌ '{word_evolution}' not in global vocabulary")


def create_vocabulary_explorer(global_vocab):
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
            st.text_area("Vocabulary preview", value=", ".join(all_words[:100]), height=200, label_visibility="hidden")


if __name__ == "__main__":
    main()
 
