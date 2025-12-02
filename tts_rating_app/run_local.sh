#!/bin/bash

# Local development script

set -e

echo "TTS Audio Rating App - Local Development"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.example to .env and configure Google OAuth credentials."
    echo ""
    echo "Setup:"
    echo "1. cp .env.example .env"
    echo "2. Edit .env with OAuth credentials"
    echo "3. Run this script again"
    exit 1
fi

# Check for Cloud SQL proxy (optional for local development)
if command -v cloud_sql_proxy &> /dev/null; then
    echo "Cloud SQL proxy found - you can connect to Cloud SQL locally"
else
    echo "Cloud SQL proxy not found. Install with:"
    echo "curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64"
    echo "chmod +x cloud_sql_proxy"
fi

echo ""
echo "Starting Flask server..."
echo "Access: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Load environment variables and run the app
export FLASK_ENV=development
export FLASK_DEBUG=1
python tts_rating_app.py