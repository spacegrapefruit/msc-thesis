# TTS Rating App

Flask web application for collecting MOS ratings on TTS audio clips.

## Prerequisites

1. **PostgreSQL Instance**: A Cloud SQL PostgreSQL instance (db-f1-micro recommended for cost efficiency)
2. **Google Cloud Storage**: A bucket containing your audio files
3. **Google OAuth**: OAuth 2.0 credentials for user authentication

## Local Development Setup

### 1. Environment Configuration

Update `.env` with your actual values:

- `DB_HOST=127.0.0.1` (when using Cloud SQL proxy)
- `DB_USER`, `DB_PASSWORD` (your PostgreSQL credentials)
- `GCS_BUCKET` (your audio files bucket)
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (OAuth credentials)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Cloud SQL Proxy (for local development)

To connect to your Cloud SQL instance locally:

```bash
# Make the script executable (if not already)
chmod +x setup_cloudsql_proxy.sh

# Start the Cloud SQL proxy in one terminal
./setup_cloudsql_proxy.sh
```

This downloads and starts the Cloud SQL proxy, allowing you to connect to your Cloud SQL instance on `localhost:5432`.

### 4. Run the Application Locally

In another terminal:

```bash
./run_local.sh
```

## Cloud Deployment

### Deploy to Cloud Run

```bash
./deploy_cloudrun.sh
```

This script will:
- Build and push the Docker image
- Deploy to Cloud Run with PostgreSQL environment variables
- Connect to your Cloud SQL instance automatically

## Database Schema

The app automatically creates this table:

```sql
CREATE TABLE ratings (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    audio_file VARCHAR(255) NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_info JSONB,
    UNIQUE(user_email, audio_file)
);
```
