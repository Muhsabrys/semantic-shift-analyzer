"""
Data loading and preprocessing functions
"""

import streamlit as st
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

from config import ALL_STOPWORDS

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


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
            year_to_text = _process_dataframe(df)
                    
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            year_to_text = _process_dataframe(df)
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


def _process_dataframe(df):
    """Process DataFrame to extract year and text columns"""
    year_col = None
    text_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['year', 'years', 'date']:
            year_col = col
        elif col_lower in ['text', 'content', 'document', 'corpus']:
            text_col = col
    
    if year_col is None or text_col is None:
        st.error(f"❌ File must have 'year' and 'text' columns. Found: {list(df.columns)}")
        return None
    
    # Group by year and combine all text
    grouped = df.groupby(year_col)[text_col].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    
    year_to_text = {}
    for _, row in grouped.iterrows():
        try:
            year = int(row[year_col])
            text = str(row[text_col]).strip()
            if text and text != 'nan':
                year_to_text[year] = text
        except (ValueError, TypeError) as e:
            st.warning(f"⚠️ Skipping row with invalid data: {e}")
    
    return year_to_text


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
        if token.isalpha() and len(token) >= 2:
            lemma = lemmatizer.lemmatize(token, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            if lemma not in ALL_STOPWORDS and len(lemma) >= 2:
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


@st.cache_data
def build_global_vocabulary(_year_to_tokens, min_years=2, min_total_count=3):
    """
    Build a global vocabulary that appears across multiple years.
    This ensures vocabulary stability and reduces OOV problems.
    """
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
        if len(years_set) >= min_years and word_to_total_count[word] >= min_total_count:
            global_vocab.add(word)
    
    return global_vocab, word_to_years, word_to_total_count
