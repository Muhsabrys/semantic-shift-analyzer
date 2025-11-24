#!/bin/bash

# Semantic Shift Analyzer - Quick Start Script
# This script sets up and runs the application locally

echo "🚀 Semantic Shift Analyzer - Quick Start"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""

# Check if requirements are installed
if [ ! -f "venv/lib/python*/site-packages/streamlit" ]; then
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🌐 Starting Streamlit app..."
echo "   The app will open in your browser automatically."
echo "   If not, navigate to: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Run the app
streamlit run semantic_shift_app.py
