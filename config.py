"""
Configuration constants for Semantic Shift Analyzer
"""

import nltk
from nltk.corpus import stopwords

# Download NLTK data
def download_nltk_data():
    """Download required NLTK data"""
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

# Initialize NLTK data
download_nltk_data()

# Get stopwords
try:
    ALL_STOPWORDS = set(stopwords.words('english'))
    
    # Remove words that are often important for semantic analysis
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

# Streamlit page config
PAGE_CONFIG = {
    "page_title": "Semantic Shift Analyzer",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Custom CSS
CUSTOM_CSS = """
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
"""
