import os
import random
import json
from datetime import UTC, datetime
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    session,
    flash,
    jsonify,
    send_file,
)
from authlib.integrations.flask_client import OAuth
import dotenv
import secrets
import psycopg2
import psycopg2.extras
from psycopg2 import pool
from google.cloud import storage
import io

dotenv.load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))

# Force HTTPS for OAuth redirects in production
if os.environ.get("FLASK_ENV") != "development":
    app.config["PREFERRED_URL_SCHEME"] = "https"

# OAuth configuration
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# PostgreSQL configuration
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "tts_ratings")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_TABLE = os.environ.get("DB_TABLE", "ratings")

# Google Cloud Storage configuration
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_AUDIO_PREFIX = os.environ.get("GCS_AUDIO_PREFIX", "inference_output/")
storage_client = storage.Client() if os.environ.get("GOOGLE_CLOUD_PROJECT") else None

# Connection pool
connection_pool = None


def init_connection_pool():
    """Initialize the database connection pool."""
    global connection_pool

    if not all([DB_HOST, DB_USER, DB_PASSWORD]):
        print(
            "Database configuration incomplete - check DB_HOST, DB_USER, DB_PASSWORD environment variables"
        )
        return False

    try:
        # Check if we're using Cloud SQL socket (Cloud Run environment)
        if DB_HOST.startswith("/cloudsql/"):
            # Cloud SQL socket connection
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                host=DB_HOST,  # This will be /cloudsql/project:region:instance
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=10,
            )
            print("Database connection pool initialized with Cloud SQL socket")
        else:
            # Regular TCP connection (local development with proxy or direct)
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                host=DB_HOST,
                port=int(DB_PORT),
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=10,
            )
            print("Database connection pool initialized with TCP connection")

        return True
    except Exception as e:
        print(f"Database connection pool initialization error: {e}")
        return False


def get_db_connection():
    """Get a connection from the pool."""
    global connection_pool

    if not connection_pool:
        if not init_connection_pool():
            return None

    try:
        return connection_pool.getconn()
    except Exception as e:
        print(f"Error getting connection from pool: {e}")
        return None


def return_db_connection(conn):
    """Return a connection to the pool."""
    global connection_pool

    if connection_pool and conn:
        try:
            connection_pool.putconn(conn)
        except Exception as e:
            print(f"Error returning connection to pool: {e}")


def init_database():
    """Initialize PostgreSQL database and table if they don't exist."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            # Create table if it doesn't exist
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE} (
                    id SERIAL PRIMARY KEY,
                    user_email VARCHAR(255) NOT NULL,
                    audio_file VARCHAR(255) NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    session_info JSONB,
                    UNIQUE(user_email, audio_file)
                )
            """)

            # Create index for faster lookups
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_user_email 
                ON {DB_TABLE} (user_email)
            """)

            conn.commit()
            print(f"Initialized PostgreSQL table: {DB_TABLE}")

    except Exception as e:
        print(f"Database initialization error: {e}")
        conn.rollback()
    finally:
        return_db_connection(conn)


def save_rating_to_database(user_email, audio_file, rating, session_info=None):
    """Save a rating to PostgreSQL database."""
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {DB_TABLE} (user_email, audio_file, rating, timestamp, session_info)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_email, audio_file) DO NOTHING
            """,
                (
                    user_email,
                    audio_file,
                    rating,
                    datetime.now(UTC),
                    json.dumps(session_info) if session_info else None,
                ),
            )

            if cur.rowcount == 0:
                # Record already exists
                return False

            conn.commit()
            return True

    except Exception as e:
        print(f"Database save error: {e}")
        conn.rollback()
        return False
    finally:
        return_db_connection(conn)


def check_existing_rating(user_email, audio_file):
    """Check if user has already rated this audio file in PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT rating FROM {DB_TABLE}
                WHERE user_email = %s AND audio_file = %s
                LIMIT 1
            """,
                (user_email, audio_file),
            )

            result = cur.fetchone()
            return result[0] if result else None

    except Exception as e:
        print(f"Database check error: {e}")
        return None
    finally:
        return_db_connection(conn)


def get_user_rated_files(user_email):
    """Get list of audio files already rated by user from PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT audio_file FROM {DB_TABLE}
                WHERE user_email = %s
            """,
                (user_email,),
            )

            results = cur.fetchall()
            return [row[0] for row in results]

    except Exception as e:
        print(f"Database query error: {e}")
        return []
    finally:
        return_db_connection(conn)


def get_user_rating_count(user_email):
    """Get count of ratings by user from PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return 0

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) FROM {DB_TABLE}
                WHERE user_email = %s
            """,
                (user_email,),
            )

            result = cur.fetchone()
            return result[0] if result else 0

    except Exception as e:
        print(f"Database query error: {e}")
        return 0
    finally:
        return_db_connection(conn)


def get_audio_files():
    """Get list of available audio files from GCS bucket."""
    if not storage_client or not GCS_BUCKET:
        print("GCS client or bucket not configured")
        return []

    try:
        bucket = storage_client.bucket(GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=GCS_AUDIO_PREFIX)

        audio_files = []
        for blob in blobs:
            if blob.name.endswith(".wav") and "/" in blob.name:
                # Extract just the filename from the full path
                filename = blob.name.split("/")[-1]
                audio_files.append(filename)

        return audio_files

    except Exception as e:
        print(f"Error listing GCS bucket: {e}")
        return []


def get_random_audio():
    """Get a random audio file that hasn't been rated by the current user."""
    audio_files = get_audio_files()
    if not audio_files:
        return None

    user_email = session.get("user_email")
    if not user_email:
        return None

    # Get files already rated by this user
    rated_files = get_user_rated_files(user_email)

    # Filter out already rated files
    unrated_files = [f for f in audio_files if f not in rated_files]

    if not unrated_files:
        # If all files are rated, return None to indicate completion
        return None

    return random.choice(unrated_files)


@app.route("/")
def index():
    """Main page - redirect to login if not authenticated."""
    if "user_email" not in session:
        return render_template("index.html")
    return redirect(url_for("rating"))


@app.route("/login")
def login():
    """Initiate Google OAuth login."""
    # Use HTTPS in production, HTTP in development
    scheme = "https" if os.environ.get("FLASK_ENV") != "development" else "http"
    redirect_uri = url_for("auth_callback", _external=True, _scheme=scheme)
    return google.authorize_redirect(redirect_uri)


@app.route("/callback")
def auth_callback():
    """Handle OAuth callback."""
    try:
        token = google.authorize_access_token()
        user_info = token.get("userinfo")

        if user_info:
            session["user_email"] = user_info["email"]
            session["user_name"] = user_info.get("name", "")

            flash("Successfully logged in!", "success")
            return redirect(url_for("rating"))
        else:
            flash("Authentication failed. Please try again.", "error")
            return redirect(url_for("index"))
    except Exception as e:
        flash("Authentication error occurred.", "error")
        return redirect(url_for("index"))


@app.route("/logout")
def logout():
    """Log out the user."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/rating")
def rating():
    """Show rating page with random audio clip."""
    if "user_email" not in session:
        return redirect(url_for("index"))

    # Check if user has rated all available files
    audio_files = get_audio_files()
    if not audio_files:
        flash("No audio files available for rating.", "error")
        return render_template("no_audio.html")

    user_email = session["user_email"]
    rated_count = get_user_rating_count(user_email)

    total_files = len(audio_files)

    if rated_count >= total_files:
        # User has rated all files
        return render_template(
            "completion.html", rated_count=rated_count, total_files=total_files
        )

    audio_file = get_random_audio()
    if not audio_file:
        # This shouldn't happen given the check above, but just in case
        return render_template(
            "completion.html", rated_count=rated_count, total_files=total_files
        )

    return render_template(
        "rating.html",
        audio_file=audio_file,
        progress={
            "rated": rated_count,
            "total": total_files,
            "remaining": total_files - rated_count,
        },
    )


@app.route("/submit_rating", methods=["POST"])
def submit_rating():
    """Submit a rating for an audio clip."""
    if "user_email" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        audio_file = request.form.get("audio_file")
        rating = int(request.form.get("rating"))

        if not audio_file or rating not in range(1, 6):
            return jsonify({"error": "Invalid input"}), 400

        user_email = session["user_email"]

        # Check if user has already voted for this audio file
        existing_rating = check_existing_rating(user_email, audio_file)
        if existing_rating:
            return jsonify(
                {
                    "error": "You have already voted for this audio sample",
                    "existing_rating": existing_rating,
                }
            ), 409

        # Save to PostgreSQL database
        session_info = {
            "user_agent": request.headers.get("User-Agent"),
            "ip_address": request.remote_addr,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        success = save_rating_to_database(user_email, audio_file, rating, session_info)
        if not success:
            return jsonify(
                {"error": "Failed to save rating or rating already exists"}
            ), 500

        return jsonify({"success": True, "message": "Rating submitted successfully"})

    except ValueError:
        return jsonify({"error": "Invalid rating value"}), 400
    except Exception as e:
        print(f"Error submitting rating: {e}")
        return jsonify({"error": "Failed to submit rating"}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve audio files from GCS bucket."""
    if not storage_client or not GCS_BUCKET:
        return "Storage not configured", 500

    if not filename.endswith(".wav"):
        return "Invalid file type", 400

    try:
        bucket = storage_client.bucket(GCS_BUCKET)
        blob_name = f"{GCS_AUDIO_PREFIX}{filename}"
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return "Audio file not found", 404

        # Download file content to memory
        audio_data = blob.download_as_bytes()

        # Create a file-like object
        audio_stream = io.BytesIO(audio_data)

        return send_file(
            audio_stream,
            mimetype="audio/wav",
            as_attachment=False,
            download_name=filename,
        )

    except Exception as e:
        print(f"Error serving audio file {filename}: {e}")
        return "Error serving audio file", 500


@app.route("/stats")
def stats():
    """Show rating statistics (optional feature)."""
    if "user_email" not in session:
        return redirect(url_for("index"))

    user_email = session["user_email"]

    # Get user's rating count
    user_rating_count = get_user_rating_count(user_email)

    # Get overall statistics
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Total ratings
                cur.execute(f"SELECT COUNT(*) FROM {DB_TABLE}")
                result = cur.fetchone()
                total_ratings = result[0] if result else 0

                # Total users
                cur.execute(f"SELECT COUNT(DISTINCT user_email) FROM {DB_TABLE}")
                result = cur.fetchone()
                total_users = result[0] if result else 0

                # Average rating
                cur.execute(f"SELECT AVG(rating) FROM {DB_TABLE}")
                result = cur.fetchone()
                avg_rating = float(result[0]) if result and result[0] else 0

        except Exception as e:
            print(f"Error getting database stats: {e}")
            total_ratings = total_users = avg_rating = 0
        finally:
            return_db_connection(conn)
    else:
        total_ratings = total_users = avg_rating = 0

    return render_template(
        "stats.html",
        user_rating_count=user_rating_count,
        total_ratings=total_ratings,
        total_users=total_users,
        avg_rating=round(avg_rating, 2),
    )


if __name__ == "__main__":
    init_connection_pool()
    init_database()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
