import os
import warnings
import wave
import pyaudio
import pyttsx3
import whisper
import google.generativeai as genai
import time
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

ai_model = genai.GenerativeModel(
    "gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)
convo = ai_model.start_chat()
system_message = '''INSTRUCTIONS:
You are a voice assistant. Use short, direct sentences to respond.'''
convo.send_message(system_message)


# Configure Gemini API
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

GEMINI_API_KEY = config["gemini_api_key"]



# Record Audio
def record_audio(filename="audio.wav", duration=15, silence_threshold=500, silence_duration=2):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print("Recording...")

    frames = []
    silence_start_time = None

    while True:
        data = stream.read(chunk)
        frames.append(data)

        # Convert to numpy array for silence detection
        audio_data = np.frombuffer(data, dtype=np.int16)
        max_amplitude = np.max(np.abs(audio_data))

        # Detect if audio is below the threshold (considered silence)
        if max_amplitude < silence_threshold:
            if silence_start_time is None:
                silence_start_time = time.time()
            elif time.time() - silence_start_time > silence_duration:
                break  
        else:
            silence_start_time = None 

        time.sleep(0.1) 

    print("Done Recording!")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

# Transcribing Audio using Whisper
import whisper

def transcribe_audio(filename="audio.wav"):
    model = whisper.load_model("base")

    options = {
        "language": "en",     
        "beam_size": 5, 
        "temperature": 0.5,   
        "best_of": 5,         
    }

    result = model.transcribe(filename, **options)
    return result["text"]

def get_nlp_response(prompt):
    model_name = ai_model
    response = model_name.generate_content(prompt)
    
    # Extract the actual content from the response using attribute access
    if response and response.candidates:
        return response.candidates[0].content.parts[0].text
    else:
        return "I'm sorry, I couldn't generate a response."

# Text-to-Speech Conversion
def text_to_speech(text):
    engine = pyttsx3.init()  
    engine.say(text)
    engine.runAndWait()
    engine.stop()  
    
# Main Voice Assistant Function
def voice_assistant():
    print("Voice Assistant Active. Say 'exit' to end the conversation.")
    while True:
        # Step 1: Record audio input
        print("Listening for your input...")
        record_audio()  
        
        # Step 2: Transcribe the audio input
        user_input = transcribe_audio()
        print("User:", user_input)

        # Exit condition
        if user_input.strip().lower() in ["exit", "quit", "stop"]:
            print("Goodbye!")
            text_to_speech("Goodbye!")
            break  
        
        # Step 3: Get the model's response to the user input
        assistant_response = get_nlp_response(user_input)
        print("Assistant:", assistant_response)

        # Step 4: Convert the model's response to speech
        text_to_speech(assistant_response)

# Run the voice assistant
voice_assistant()