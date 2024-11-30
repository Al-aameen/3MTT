from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import time
from threading import Thread

# Flask App Initialization
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    """
    Render the main HTML page.
    """
    return render_template("index.html")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    """
    Endpoint to process uploaded audio.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_path = os.path.join("uploads", f"audio_{int(time.time())}.wav")
    audio_file.save(audio_path)

    # Transcribe Audio
    transcription = transcribe_audio(audio_path)

    # Process the transcription for entities
    entities = extract_entities(transcription)
    if entities["task"] or entities["date"] or entities["time"]:
        assistant_response = process_entities(entities)
    else:
        assistant_response = get_nlp_response(transcription)

    # Text-to-Speech
    text_to_speech(assistant_response)

    return jsonify({"transcription": transcription, "response": assistant_response})

# Run the Flask App
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  # Create uploads directory if not exists
    app.run(debug=True)