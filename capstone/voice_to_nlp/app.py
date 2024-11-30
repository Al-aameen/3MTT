from flask_cors import CORS
import time
from threading import Thread
from flask import Flask, request, jsonify
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename="assistant.log", level=logging.INFO, format="%(asctime)s - %(message)s")

@app.route("/")
def index():
    return "Voice Assistant is running!"

@app.route("/process_audio", methods=["POST"])
def process_audio():
    """
    Process the uploaded audio file and return the transcription and response.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_path = os.path.join("uploads", "audio.wav")
    audio_file.save(audio_path)

    # Transcribe the audio file
    transcription = transcribe_audio(audio_path)

    # Process the transcription
    entities = extract_entities(transcription)
    if entities["task"] or entities["date"] or entities["time"]:
        assistant_response = process_entities(entities)
    else:
        assistant_response = get_nlp_response(transcription)

    # Log and return the response
    log_interaction(transcription, assistant_response)
    return jsonify({"transcription": transcription, "response": assistant_response})


# Run the Flask app
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  # Create uploads directory
    app.run(debug=True)