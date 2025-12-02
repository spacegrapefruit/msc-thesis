import os
import random
import sqlite3
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
from datetime import datetime

dotenv.load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))

# OAuth configuration
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Database configuration
DATABASE = "ratings.db"


def init_db():
    """Initialize the database with required tables."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Check if ratings table already exists and if it has the unique constraint
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='ratings'"
        )
        existing_table = cursor.fetchone()

        if existing_table and "UNIQUE(user_email, audio_file)" not in existing_table[0]:
            # Migrate existing table by recreating it with unique constraint
            print("Migrating existing ratings table to add unique constraint...")
            cursor.execute("""
                CREATE TABLE ratings_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    audio_file TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_email, audio_file)
                )
            """)

            # Copy unique ratings from old table (keep only the latest rating per user/audio combo)
            cursor.execute("""
                INSERT INTO ratings_new (user_email, audio_file, rating, timestamp)
                SELECT user_email, audio_file, rating, timestamp
                FROM (
                    SELECT user_email, audio_file, rating, timestamp,
                           ROW_NUMBER() OVER (PARTITION BY user_email, audio_file ORDER BY timestamp DESC) as rn
                    FROM ratings
                ) WHERE rn = 1
            """)

            # Replace old table
            cursor.execute("DROP TABLE ratings")
            cursor.execute("ALTER TABLE ratings_new RENAME TO ratings")
            print(
                "Migration completed. Duplicate ratings have been removed (kept latest per user/audio)."
            )
        else:
            # Create table with unique constraint if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    audio_file TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_email, audio_file)
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                ratings_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()


def get_audio_files():
    """Get list of available audio files."""
    audio_dir = os.path.join(os.path.dirname(__file__), "inference_output")
    if os.path.exists(audio_dir):
        return [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
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
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT audio_file FROM ratings WHERE user_email = ?", (user_email,)
        )
        rated_files = [row[0] for row in cursor.fetchall()]

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
    redirect_uri = url_for("auth_callback", _external=True)
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

            # Record user session
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO user_sessions (user_email) VALUES (?)",
                    (user_info["email"],),
                )
                conn.commit()

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
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM ratings WHERE user_email = ?", (user_email,)
        )
        rated_count = cursor.fetchone()[0]

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
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT rating FROM ratings WHERE user_email = ? AND audio_file = ?",
                (user_email, audio_file),
            )
            existing_rating = cursor.fetchone()

            if existing_rating:
                return jsonify(
                    {
                        "error": "You have already voted for this audio sample",
                        "existing_rating": existing_rating[0],
                    }
                ), 409

            # Store the rating
            cursor.execute(
                "INSERT INTO ratings (user_email, audio_file, rating) VALUES (?, ?, ?)",
                (user_email, audio_file, rating),
            )

            # Update session ratings count
            cursor.execute(
                "UPDATE user_sessions SET ratings_count = ratings_count + 1 WHERE user_email = ? AND id = (SELECT MAX(id) FROM user_sessions WHERE user_email = ?)",
                (user_email, user_email),
            )

            conn.commit()

        return jsonify({"success": True, "message": "Rating submitted successfully"})

    except sqlite3.IntegrityError:
        return jsonify({"error": "You have already voted for this audio sample"}), 409
    except Exception as e:
        return jsonify({"error": "Failed to submit rating"}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve audio files."""
    audio_dir = os.path.join(os.path.dirname(__file__), "inference_output")
    audio_path = os.path.join(audio_dir, filename)

    if os.path.exists(audio_path) and filename.endswith(".wav"):
        return send_file(audio_path, mimetype="audio/wav")
    else:
        return "Audio file not found", 404


@app.route("/stats")
def stats():
    """Show rating statistics (optional feature)."""
    if "user_email" not in session:
        return redirect(url_for("index"))

    user_email = session["user_email"]

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Get user's rating count
        cursor.execute(
            "SELECT COUNT(*) FROM ratings WHERE user_email = ?", (user_email,)
        )
        user_rating_count = cursor.fetchone()[0]

        # Get overall statistics
        cursor.execute("SELECT COUNT(*) FROM ratings")
        total_ratings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT user_email) FROM ratings")
        total_users = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(rating) FROM ratings")
        avg_rating = cursor.fetchone()[0] or 0

    return render_template(
        "stats.html",
        user_rating_count=user_rating_count,
        total_ratings=total_ratings,
        total_users=total_users,
        avg_rating=round(avg_rating, 2),
    )


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
