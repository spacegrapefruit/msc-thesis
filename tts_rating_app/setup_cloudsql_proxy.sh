#!/bin/bash

# Cloud SQL Proxy setup for local development
# This allows you to connect to your Cloud SQL instance locally

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-mif-studies-ajs}"
INSTANCE_NAME="${DB_INSTANCE_NAME:-mif-tts-db-instance-1}"
REGION="${DB_REGION:-europe-west1}"

echo "Setting up Cloud SQL Proxy for local development"
echo "================================================"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not installed."
    exit 1
fi

# Check if already authenticated
if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" | grep -q "@"; then
    echo "Please authenticate with gcloud first:"
    echo "gcloud auth login"
    echo "gcloud auth application-default login"
    exit 1
fi

# Download Cloud SQL proxy if not exists
if [ ! -f "cloud_sql_proxy" ]; then
    echo "Downloading Cloud SQL proxy..."
    curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64
    chmod +x cloud_sql_proxy
    echo "Cloud SQL proxy downloaded."
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please create .env file with your configuration first."
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

CONNECTION_NAME="$PROJECT_ID:$REGION:$INSTANCE_NAME"

echo "Starting Cloud SQL proxy..."
echo "Connection: $CONNECTION_NAME"
echo "Local port: 5432"
echo ""
echo "In another terminal, you can now run:"
echo "  ./run_local.sh"
echo ""
echo "Your app will connect to Cloud SQL through the proxy."
echo "Press Ctrl+C to stop the proxy."
echo ""

# Start the proxy
./cloud_sql_proxy -instances="$CONNECTION_NAME"=tcp:5432