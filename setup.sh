#!/bin/bash

# Competitive Research Multi-Agent System Setup Script

echo "🚀 Setting up Competitive Research Multi-Agent System..."
echo

# Check Python version
python_version=$(python3 --version 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Found $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create one from .env.example"
    echo "   Copy .env.example to .env and add your API keys:"
    echo "   cp .env.example .env"
    echo
else
    echo "✅ .env file found"
fi

# Create reports directory
mkdir -p reports
echo "✅ Reports directory created"

echo
echo "🎉 Setup complete!"
echo
echo "Next steps:"
echo "1. If you haven't already, configure your API keys in .env"
echo "2. Test the system: python main.py validate"
echo "3. Run your first research: python main.py research 'your query here'"
echo
echo "For help: python main.py --help"