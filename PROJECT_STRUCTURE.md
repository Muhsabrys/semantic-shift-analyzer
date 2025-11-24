# 📁 Project Structure

```
semantic-shift-analyzer/
│
├── 📄 semantic_shift_app.py      # Main Streamlit application
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Project documentation
├── 📄 DEPLOYMENT.md               # Deployment instructions
├── 📄 .gitignore                  # Git ignore file
├── 📄 run_app.sh                  # Quick start script (Mac/Linux)
├── 📄 run_app.bat                 # Quick start script (Windows)
│
└── 📁 .streamlit/
    └── 📄 config.toml             # Streamlit configuration
```

## 🚀 Quick Start

### Mac/Linux:
```bash
./run_app.sh
```

### Windows:
```batch
run_app.bat
```

### Manual:
```bash
pip install -r requirements.txt
streamlit run semantic_shift_app.py
```

## 📤 Upload to GitHub

1. Create a new repository on GitHub
2. Upload all these files
3. Deploy to Streamlit Cloud (see DEPLOYMENT.md)

## 🎯 What Each File Does

| File | Purpose |
|------|---------|
| `semantic_shift_app.py` | Main application with GUI |
| `requirements.txt` | All Python packages needed |
| `README.md` | User documentation |
| `DEPLOYMENT.md` | Step-by-step deployment guide |
| `.gitignore` | Files to exclude from Git |
| `run_app.sh` | Easy startup for Mac/Linux |
| `run_app.bat` | Easy startup for Windows |
| `.streamlit/config.toml` | App styling and configuration |

## 🌟 Features in the App

✅ **Single Word Drift Analysis**
- Track semantic change over time
- 3D trajectory visualization
- Similarity matrices

✅ **Word-to-Word Distance**
- Compare two words across years
- Statistical analysis
- Trend visualization

✅ **Semantic Networks**
- Explore word associations
- Network graphs for each year
- Top neighbors list

✅ **Multi-Word Comparison**
- Compare multiple words simultaneously
- See which changed most
- Side-by-side drift plots

## 📊 Example Use Cases

1. **"crisis" over time** → How has political discourse changed?
2. **"crisis" vs "problem"** → Are they used differently?
3. **"freedom" network in 1960** → Cold War context
4. **Compare: war, peace, economy** → Which concepts shifted most?

## 🎨 Customization

### Change Colors
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor="#YOUR_COLOR"
backgroundColor="#YOUR_COLOR"
```

### Add More Visualizations
Edit `semantic_shift_app.py` and add new plot functions

### Use Different Corpus
Replace `state_union` with your own text data

## 🐛 Troubleshooting

**App won't start?**
- Check Python version (3.9+)
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

**Out of memory?**
- Reduce `vector_size` in Word2Vec (line ~153)
- Limit years analyzed

**NLTK data missing?**
- Run: `python -c "import nltk; nltk.download('state_union'); nltk.download('punkt')"`

## 📚 Learn More

- [Streamlit Documentation](https://docs.streamlit.io)
- [Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html)
- [Semantic Change Detection](https://aclanthology.org/)

---

**Ready to deploy? See DEPLOYMENT.md for full instructions! 🚀**
