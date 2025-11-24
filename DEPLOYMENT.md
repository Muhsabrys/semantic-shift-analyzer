# 🚀 Deployment Guide - Semantic Shift Analyzer

## Option 1: Deploy to Streamlit Cloud (Recommended - FREE!)

### Prerequisites
- GitHub account
- Git installed on your computer

### Step-by-Step Instructions

#### 1. Create a GitHub Repository

**Option A: Via GitHub Website**
1. Go to [github.com](https://github.com)
2. Click the "+" icon → "New repository"
3. Name: `semantic-shift-analyzer`
4. Description: "Interactive tool for analyzing semantic drift in text"
5. Make it **Public**
6. Click "Create repository"

**Option B: Via Command Line**
```bash
# Navigate to your project directory
cd /path/to/your/project

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Semantic Shift Analyzer"

# Create repository on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/semantic-shift-analyzer.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Choose your repository: `YOUR_USERNAME/semantic-shift-analyzer`
   - Branch: `main`
   - Main file path: `semantic_shift_app.py`
   - App URL: Choose a custom URL (e.g., `semantic-shift-analyzer`)

3. **Advanced Settings (Optional)**
   - Python version: 3.9 or higher
   - Click "Deploy!"

4. **Wait for Deployment**
   - Initial deployment takes 2-5 minutes
   - Watch the logs for any errors
   - Once complete, your app will be live!

5. **Your App URL**
   ```
   https://YOUR_USERNAME-semantic-shift-analyzer.streamlit.app
   ```

### Troubleshooting Deployment

**Issue: Dependencies not installing**
- Make sure `requirements.txt` is in the root directory
- Check that all package versions are compatible

**Issue: NLTK data not found**
- The app downloads NLTK data automatically on first run
- If issues persist, add to `.streamlit/config.toml`:
  ```toml
  [server]
  enableXsrfProtection = false
  ```

**Issue: App crashes on startup**
- Check the logs in Streamlit Cloud dashboard
- Common issue: Memory limit exceeded (downgrade to smaller models)

---

## Option 2: Deploy to Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed

### Files Needed

1. **Create `Procfile`**
```
web: sh setup.sh && streamlit run semantic_shift_app.py
```

2. **Create `setup.sh`**
```bash
#!/bin/bash

mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. **Create `runtime.txt`**
```
python-3.9.16
```

### Deployment Steps

```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create semantic-shift-analyzer

# Deploy
git push heroku main

# Open app
heroku open
```

---

## Option 3: Run Locally

### Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/semantic-shift-analyzer.git
cd semantic-shift-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run semantic_shift_app.py
```

The app will open automatically at `http://localhost:8501`

---

## Option 4: Docker Deployment

### Create `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY semantic_shift_app.py .
COPY .streamlit/ .streamlit/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('state_union'); nltk.download('punkt')"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "semantic_shift_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t semantic-shift-analyzer .

# Run container
docker run -p 8501:8501 semantic-shift-analyzer
```

---

## Performance Optimization Tips

### 1. Caching
The app uses `@st.cache_data` and `@st.cache_resource` for:
- Loading corpus data
- Training Word2Vec models
- Storing aligned embeddings

### 2. Memory Management
- Models are stored in session state
- Only train models once per session
- Clear cache if memory issues occur

### 3. Scaling
For large-scale deployment:
- Use Redis for caching (Streamlit Cloud supports this)
- Pre-compute embeddings and store them
- Use a database for model storage

---

## Monitoring & Maintenance

### Streamlit Cloud Dashboard
- View app analytics
- Check logs for errors
- Monitor resource usage
- See visitor statistics

### Updating the App
```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push origin main
```

Streamlit Cloud will automatically redeploy!

---

## Custom Domain (Optional)

1. **Streamlit Cloud**
   - Available on paid plans
   - Settings → Custom domain

2. **Your Own Domain**
   - Use Cloudflare or similar service
   - Point CNAME to Streamlit app URL

---

## Security Best Practices

1. **Never commit sensitive data**
   - Use `.gitignore` for secrets
   - Use Streamlit secrets management for API keys

2. **Rate Limiting**
   - Streamlit Cloud has built-in rate limiting
   - For custom deployment, use nginx

3. **Authentication**
   - Add password protection if needed
   - Use Streamlit's authentication features

---

## Support & Resources

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: For app-specific problems

---

## Cost Analysis

| Platform | Cost | Limitations |
|----------|------|-------------|
| Streamlit Cloud | **FREE** | 1GB RAM, Public apps only |
| Streamlit Cloud (Paid) | $25-250/mo | More resources, Private apps |
| Heroku | FREE tier | 550 hours/month |
| AWS/GCP | Variable | Pay for resources used |
| Local | FREE | Your own hardware |

**Recommendation**: Start with **Streamlit Cloud FREE** tier!

---

## Next Steps

1. ✅ Deploy to Streamlit Cloud
2. ✅ Share your app URL
3. ✅ Gather user feedback
4. ✅ Iterate and improve
5. ✅ Add more features!

Good luck with your deployment! 🚀
