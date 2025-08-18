#!/bin/bash

# CDR RAG Pipeline Startup Script
# This script sets up and starts the CDR RAG pipeline

echo " Starting CDR RAG Pipeline Setup..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo " Ollama not found."
    echo ""
    echo " Please install Ollama manually for macOS:"
    echo "   1. Visit: https://ollama.ai/download"
    echo "   2. Download 'Ollama for macOS'"
    echo "   3. Install the .app file"
    echo "   4. Run Ollama from Applications"
    echo "   5. Then run this script again"
    echo ""
    echo " Alternative - Install via Homebrew:"
    echo "   brew install ollama"
    echo ""
    exit 1
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo " Starting Ollama server..."
    ollama serve &
    sleep 5
fi

# Check if Mistral model is available
echo " Checking Mistral model..."
if ! ollama list | grep -q "mistral"; then
    echo " Downloading Mistral model (this may take a few minutes)..."
    ollama pull mistral
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo " Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if CDR data file exists
if [ ! -f "data/es_data.json" ]; then
    echo "CDR data file not found at data/es_data.json"
    echo "   Please place your Elasticsearch export file there"
    exit 1
fi

echo "Setup complete!"
echo ""
echo "Starting web interface..."
echo "   URL: http://localhost:8000"
echo ""

# Start the FastAPI server (virtual environment is already activated)
python app.py
