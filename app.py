from flask import Flask
from flask_cors import CORS
from attendance_routes import attendance_bp
from memorysnap_routes import memorysnap_bp  

app = Flask(__name__, static_url_path="/static", static_folder="Unknown_Faces")
CORS(app)

app.register_blueprint(attendance_bp)
app.register_blueprint(memorysnap_bp)

@app.route("/")
def index():
    return "Face Recognition Flask API running"

if __name__ == "__main__":
    app.run(debug=True, port=3000, host="0.0.0.0")
