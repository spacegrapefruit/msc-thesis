#!/bin/bash

# Cloud Run deployment script for TTS Rating App
# Make sure you have gcloud CLI installed and authenticated

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-mif-studies-ajs}"
SERVICE_NAME="tts-rating-app"
REGION="${CLOUD_RUN_REGION:-europe-west1}"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Starting Cloud Run deployment..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not installed."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not installed."
    exit 1
fi

# Check required setup
echo "Ensure you have:"
echo "1. Google OAuth 2.0 credentials in .env file"
echo "2. Authorized redirect URI: https://your-service-url/callback"
echo ""
echo "Continue? (y/N)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Load environment variables from .env
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.example to .env and configure credentials."
    exit 1
fi

# Source the .env file
set -a
source .env
set +a

# Check required variables
if [ -z "$GOOGLE_CLIENT_ID" ] || [ -z "$GOOGLE_CLIENT_SECRET" ]; then
    echo "Error: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env"
    exit 1
fi

if [ -z "$DB_HOST" ] || [ -z "$DB_USER" ] || [ -z "$DB_PASSWORD" ]; then
    echo "Error: Database variables (DB_HOST, DB_USER, DB_PASSWORD) must be set in .env"
    exit 1
fi

if [ -z "$GCS_BUCKET" ]; then
    echo "Error: GCS_BUCKET must be set in .env for audio file storage"
    exit 1
fi

# Use SECRET_KEY from .env or generate one
if [ -z "$SECRET_KEY" ]; then
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "Generated SECRET_KEY: $SECRET_KEY"
    echo "Add this to your .env file: SECRET_KEY=$SECRET_KEY"
fi

echo "1. Setting up gcloud..."
gcloud config set project "$PROJECT_ID"

echo "2. Enabling APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage.googleapis.com

echo "3. Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "4. Configuring Docker authentication..."
gcloud auth configure-docker --quiet

echo "5. Pushing image..."
docker push "$IMAGE_NAME"

echo "6. Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --set-env-vars "GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID,GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET,SECRET_KEY=$SECRET_KEY,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,DB_HOST=$DB_HOST,DB_PORT=$DB_PORT,DB_NAME=$DB_NAME,DB_USER=$DB_USER,DB_PASSWORD=$DB_PASSWORD,DB_TABLE=$DB_TABLE,GCS_BUCKET=$GCS_BUCKET,GCS_AUDIO_PREFIX=$GCS_AUDIO_PREFIX" \
    --max-instances 10

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format="value(status.url)")

echo "Deployment completed!"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Update OAuth redirect URI: $SERVICE_URL/callback"
echo "App live at: $SERVICE_URL"