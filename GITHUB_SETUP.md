# 🐙 GitHub Repository Setup Guide

## Method 1: Upload via GitHub Website (Easiest)

### Step 1: Create Repository
1. Go to [github.com](https://github.com)
2. Click "+" icon → "New repository"
3. Repository name: `semantic-shift-analyzer`
4. Description: "Interactive web app for analyzing semantic drift in State of the Union speeches"
5. Choose "Public" (required for free Streamlit deployment)
6. ✅ Check "Add a README file" 
7. Click "Create repository"

### Step 2: Upload Files
1. In your new repository, click "Add file" → "Upload files"
2. Drag and drop ALL these files:
   - `semantic_shift_app.py`
   - `requirements.txt`
   - `README.md`
   - `DEPLOYMENT.md`
   - `PROJECT_STRUCTURE.md`
   - `.gitignore`
   - `run_app.sh`
   - `run_app.bat`
3. Create folder `.streamlit` and upload `config.toml` inside it
4. Add commit message: "Initial commit: Semantic Shift Analyzer"
5. Click "Commit changes"

### Step 3: Verify
Your repository structure should look like:
```
semantic-shift-analyzer/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── DEPLOYMENT.md
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
├── run_app.bat
├── run_app.sh
└── semantic_shift_app.py
```

---

## Method 2: Command Line (For Git Users)

### Prerequisites
- Git installed on your computer
- GitHub account with SSH key or personal access token

### Step 1: Create Repository on GitHub
1. Go to [github.com](https://github.com)
2. Create new repository: `semantic-shift-analyzer`
3. Make it **Public**
4. **DO NOT** initialize with README

### Step 2: Initialize Local Repository

```bash
# Navigate to your project folder (where you downloaded the files)
cd /path/to/semantic-shift-analyzer

# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Semantic Shift Analyzer"

# Add remote origin (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/semantic-shift-analyzer.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 3: Verify Upload
Visit: `https://github.com/YOUR_USERNAME/semantic-shift-analyzer`

---

## Method 3: GitHub Desktop (User-Friendly)

### Step 1: Install GitHub Desktop
- Download from: [desktop.github.com](https://desktop.github.com)
- Install and sign in with your GitHub account

### Step 2: Create Repository
1. Click "File" → "New repository"
2. Name: `semantic-shift-analyzer`
3. Description: "Interactive web app for analyzing semantic drift"
4. Local path: Choose where your files are
5. Click "Create repository"

### Step 3: Add Files
1. Copy all project files to the repository folder
2. GitHub Desktop will show all new files
3. Add commit message: "Initial commit"
4. Click "Commit to main"

### Step 4: Publish
1. Click "Publish repository"
2. Make sure "Keep this code private" is **UNCHECKED**
3. Click "Publish repository"

---

## 🚀 Deploy to Streamlit Cloud (After GitHub Setup)

### Step 1: Go to Streamlit Cloud
Visit: [share.streamlit.io](https://share.streamlit.io)

### Step 2: Sign In
Click "Sign in with GitHub"

### Step 3: Deploy App
1. Click "New app"
2. Repository: `YOUR_USERNAME/semantic-shift-analyzer`
3. Branch: `main`
4. Main file path: `semantic_shift_app.py`
5. App URL (optional): Choose custom name
6. Click "Deploy"

### Step 4: Wait for Deployment
- First deployment: 2-5 minutes
- Watch the logs for progress
- Any errors will be shown in the logs

### Step 5: Share Your App! 🎉
Your app URL: `https://YOUR_USERNAME-semantic-shift-analyzer.streamlit.app`

---

## 🔄 Updating Your App

### Via GitHub Website
1. Navigate to the file you want to edit
2. Click the pencil icon (Edit)
3. Make changes
4. Commit changes
5. Streamlit auto-redeploys!

### Via Command Line
```bash
# Make changes to your files
git add .
git commit -m "Update: description of changes"
git push origin main
```

### Via GitHub Desktop
1. Make changes to files
2. GitHub Desktop shows changes
3. Write commit message
4. Click "Commit to main"
5. Click "Push origin"

Streamlit Cloud automatically redeploys when you push changes!

---

## ✅ Checklist

Before deploying, make sure you have:

- [ ] All files uploaded to GitHub
- [ ] Repository is **Public** (required for free Streamlit)
- [ ] `requirements.txt` is in root directory
- [ ] `semantic_shift_app.py` is in root directory
- [ ] `.streamlit/config.toml` exists (optional but recommended)
- [ ] No sensitive data or API keys in code

---

## 🎯 Your App Features

Once deployed, users can:
- ✅ Analyze semantic drift of any word
- ✅ Compare word distances over time
- ✅ Explore semantic networks
- ✅ Compare multiple words
- ✅ View 3D trajectories
- ✅ Download statistics

---

## 📧 Sharing Your App

After deployment, share your app with:

1. **Direct URL**: `https://YOUR_APP_URL.streamlit.app`
2. **GitHub repo**: Add link to README
3. **Social media**: Share screenshots + link
4. **Academic papers**: Cite your app
5. **Presentations**: Embed live demo

---

## 🔧 Troubleshooting

### "Repository not found"
- Make sure repository is **Public**
- Check repository name spelling

### "Deployment failed"
- Check logs in Streamlit Cloud dashboard
- Verify `requirements.txt` has all dependencies
- Check Python version compatibility

### "Module not found"
- Add missing package to `requirements.txt`
- Push changes to GitHub
- Streamlit will rebuild automatically

### "Out of memory"
- Streamlit free tier has 1GB RAM limit
- Reduce model size or cache data
- Consider upgrading to paid tier

---

## 📊 Analytics & Monitoring

Streamlit Cloud provides:
- ✅ Visitor count
- ✅ Error logs
- ✅ Resource usage
- ✅ Uptime monitoring

Access via: Streamlit Cloud Dashboard

---

## 🆙 Upgrading Your App

### Free Tier Limitations
- 1GB RAM
- Public apps only
- Community support

### Paid Tier Benefits ($25-250/month)
- More RAM (up to 32GB)
- Private apps
- Custom domains
- Priority support
- Team features

Start with free tier, upgrade if needed!

---

## 🎓 Next Steps

1. ✅ Upload to GitHub
2. ✅ Deploy to Streamlit Cloud
3. ✅ Share with friends/colleagues
4. ✅ Gather feedback
5. ✅ Add new features
6. ✅ Write a blog post about it!

---

## 📚 Useful Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [GitHub Guides](https://guides.github.com)
- [Streamlit Forum](https://discuss.streamlit.io)
- [Git Tutorial](https://git-scm.com/docs/gittutorial)

---

## 🏆 Success!

Once deployed, your app will be:
- ✅ **Live 24/7** on the internet
- ✅ **Free** to use forever
- ✅ **Accessible** to anyone worldwide
- ✅ **Impressive** for your portfolio!

**Good luck with your deployment! 🚀**

---

*Need help? Open an issue on GitHub or ask on [Streamlit Forum](https://discuss.streamlit.io)*
