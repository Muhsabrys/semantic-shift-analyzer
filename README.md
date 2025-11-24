# 📊 Semantic Shift Analyzer

An interactive web application for analyzing how word meanings change over time in State of the Union speeches (1945-2002).

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🌟 Features

- **Single Word Drift Analysis**: Track how a word's meaning evolves over time
- **3D Semantic Trajectories**: Visualize word movement through semantic space
- **Word-to-Word Distance**: Compare how two words relate across different years
- **Semantic Networks**: Explore word associations in specific years
- **Multi-Word Comparison**: Analyze multiple words simultaneously

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/semantic-shift-analyzer.git
cd semantic-shift-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run semantic_shift_app.py
```

4. **Open in browser**
The app will automatically open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud (FREE!)

### Step 1: Create GitHub Repository

1. Create a new repository on GitHub
2. Upload these files:
   - `semantic_shift_app.py`
   - `requirements.txt`
   - `README.md`

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set main file: `semantic_shift_app.py`
6. Click "Deploy"!

**Your app will be live at**: `https://YOUR_USERNAME-semantic-shift-analyzer.streamlit.app`

## 🎯 How to Use

### 1. Single Word Analysis
- Enter any word (e.g., "crisis", "freedom", "economy")
- View drift plots, 3D trajectories, and similarity matrices
- See year-by-year statistics

### 2. Word-to-Word Comparison
- Enter two words (e.g., "crisis" and "problem")
- See how their semantic distance changes over time
- Get statistical summaries and interpretations

### 3. Semantic Networks
- Explore word associations in specific years
- Visualize semantic neighborhoods
- See top similar words with similarity scores

### 4. Multi-Word Comparison
- Compare multiple words simultaneously
- See which words changed most over time
- Compare drift trajectories

## 📖 Technical Details

### Methodology

1. **Word2Vec Training**: Skip-gram model trained on each year's speech
2. **Alignment**: Orthogonal Procrustes alignment to compare embeddings across years
3. **Distance Metric**: Cosine distance measures semantic shift
4. **Visualization**: PCA reduces 200D embeddings to 3D for visualization

### Data Source

- **NLTK State of the Union Corpus**
- Years: 1945-2002
- Speeches from multiple U.S. presidents

### Key Metrics

- **Cosine Distance**: 0 = identical, 2 = opposite meanings
  - < 0.3: Very similar
  - 0.3-0.7: Moderately related
  - > 0.7: Quite different

## 🎨 Screenshots

[Add screenshots here after deployment]

## 📝 Example Analyses

### Crisis Over Time
The word "crisis" shows significant semantic shift:
- 1948-1961: Major change (post-WWII context shift)
- 1960s-1970s: Stable usage
- 1980s-2000s: Diversified contexts

### Crisis vs. Problem
These words show varying semantic distance:
- Some years: very similar (interchangeable)
- Other years: distinct uses (different severity)

## 🛠️ Technologies Used

- **Streamlit**: Web framework
- **Gensim**: Word2Vec implementation
- **NLTK**: Corpus and tokenization
- **scikit-learn**: PCA, Procrustes alignment
- **Matplotlib/Seaborn**: Visualizations
- **NetworkX**: Graph visualizations

## 📊 Project Structure

```
semantic-shift-analyzer/
├── semantic_shift_app.py      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .streamlit/
    └── config.toml            # Optional: Streamlit config
```

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- NLTK for the State of the Union corpus
- Streamlit for the amazing web framework
- Gensim for Word2Vec implementation

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ and Python**
