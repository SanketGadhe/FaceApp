# app.py
from flask import Flask
from flask_cors import CORS
from attendance_routes import attendance_bp # Assuming this exists
from memorysnap_routes import memorysnap_bp
from dotenv import load_dotenv
import os # Import os for environment variables

load_dotenv(override=True) 

# For production, `Unknown_Faces` won't be writable or publicly accessible
# for new uploads. It's fine for Flask to know about it, but your core
# logic should use S3 for storage.
app = Flask(__name__, static_url_path="/static", static_folder="Unknown_Faces")
CORS(app)

app.register_blueprint(attendance_bp)
app.register_blueprint(memorysnap_bp)

@app.route("/")
def index():
    return "Face Recognition Flask API running"

if __name__ == "__main__":
    # Use environment variable for port, default to 5001
    port = int(os.environ.get("PORT", 5001))
    # Disable debug mode for production
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug_mode, port=port, host="0.0.0.0")