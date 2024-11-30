import os
import json
import wave
import whisper
import logging
import pyaudio
import pyttsx3
import google.cloud.speech as speech
import google.generativeai as genai
import time
import requests
import numpy as np
import spacy
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# os.system('python -m spacy download en_core_web_sm')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("SpaCy model not found. Run `python -m spacy download en_core_web_sm` to install.")
    raise


def get_weather(city):
    """
    Fetch weather information for the given city using WeatherAPI.
    """
    base_url = "http://api.weatherapi.com/v1/current.json"
    url = f"{base_url}?key={WEATHER_API_KEY}&q={city}&lang=en"

    try:
        response = requests.get(url)
        data = response.json()

        if "error" in data:
            return f"Sorry, I couldn't find weather information for {city}."
        
        # Extract weather information
        weather = data["current"]["condition"]["text"]
        temp = data["current"]["temp_c"]
        return f"The current weather in {city} is {weather} with a temperature of {temp}°C."
    except Exception as e:
        return f"Unable to fetch weather data. Error: {str(e)}"


def extract_entities(user_input):
    """
    Extract entities like dates, times, and tasks from user input using spaCy.
    """
    doc = nlp(user_input)
    entities = {"date": None, "time": None, "task": None}

    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME"]:
            if "time" in ent.text.lower():
                entities["time"] = ent.text
            else:
                entities["date"] = ent.text

    # Extract task-related keywords manually
    if "remind me to" in user_input.lower():
        entities["task"] = user_input.lower().split("remind me to")[-1].strip()

    return entities

def save_reminder(task, date=None, time=None):
    """
    Save a reminder to a local file.
    """
    with open("reminders.txt", "a") as file:
        reminder = f"Task: {task}"
        if date:
            reminder += f", Date: {date}"
        if time:
            reminder += f", Time: {time}"
        file.write(reminder + "\n")
    return "Reminder saved successfully."

# Configure logging
logging.basicConfig(filename="assistant.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Exit commands
EXIT_COMMANDS = ["exit", "quit", "stop", "goodbye"]

# Initialize TTS engine globally
tts_engine = pyttsx3.init()

# Load all API Keys
try:
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    
    GEMINI_API_KEY = config.get("gemini_api_key")
    WEATHER_API_KEY = config.get("weather_api_key")  # Load WeatherAPI key from config

    if not GEMINI_API_KEY or not WEATHER_API_KEY:
        raise ValueError("API keys missing in config.json.")
    
    
except FileNotFoundError:
    raise FileNotFoundError("config.json not found. Ensure it is in the working directory.")
except Exception as e:
    raise Exception(f"Error loading configuration: {e}")


# Initialize Google Generative AI
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
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel(
    "gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)
convo = ai_model.start_chat()
system_message = '''INSTRUCTIONS:
You are a voice assistant. Use short, direct sentences to respond.'''
convo.send_message(system_message)

from googlesearch import search

def web_search(query, num_results=3):
    """
    Perform a Google search and return the top results.
    """
    try:
        results = []
        for url in search(query, num_results=num_results):
            results.append(url)
        return results
    except Exception as e:
        return f"Unable to perform web search. Error: {str(e)}"


def is_exit_command(command):
    return command.strip().lower() in EXIT_COMMANDS

# Logging interactions
def log_interaction(user_input, assistant_response):
    logging.info(f"User: {user_input}")
    logging.info(f"Assistant: {assistant_response}")

# Record Audio
def record_audio(filename="audio.wav", duration=10, silence_threshold=500, silence_duration=2):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print("Recording...")
    frames = []
    silence_start_time = None
    try:
        while True:
            data = stream.read(chunk)
            frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            max_amplitude = np.max(np.abs(audio_data))

            if max_amplitude < silence_threshold:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > silence_duration:
                    break
            else:
                silence_start_time = None

            time.sleep(0.1)
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        print("Done Recording!")
        stream.stop_stream()
        stream.close()
        p.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

def transcribe_audio(filename="audio.wav"):
    """
    Transcribes audio using Whisper and returns the transcription text.
    
    Args:
        filename (str): Path to the audio file to be transcribed.
        
    Returns:
        str: Transcribed text from the audio file.
    """
    model = whisper.load_model("base")

    options = {
        "language": "en",     
        "beam_size": 5,       
        "temperature": 0.5,    
        "best_of": 5,          
    }

    try:
        result = model.transcribe(filename, **options)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""


# Generate NLP Response
def get_nlp_response(prompt):
    try:
        response = convo.generate_content(prompt)
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "I'm sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm having trouble understanding you. Please try again."

# Text-to-Speech Conversion
def text_to_speech(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error during TTS: {e}")


def process_entities(entities):
    """
    Handle extracted entities and perform corresponding actions.
    """
    if entities["task"]:
        response = f"Setting a reminder: {entities['task']}."
        if entities["date"] or entities["time"]:
            response += f" Scheduled for {entities['date'] or ''} {entities['time'] or ''}."
        save_reminder(entities["task"], date=entities["date"], time=entities["time"])
        return response

    elif "weather" in entities["task"].lower():
        city = entities["task"].replace("weather in", "").strip()
        return get_weather(city)

    elif "search" in entities["task"].lower():
        query = entities["task"].replace("search for", "").strip()
        results = web_search(query)
        return f"Here are the top search results:\n" + "\n".join(results)

    return "I couldn't find a specific task to perform."


def voice_assistant():
    print("Voice Assistant Active. Say 'exit' to end the conversation.")
    while True:
        print("Listening for your input...")
        record_audio()  
        user_input = transcribe_audio()  
        print("User:", user_input)

        if is_exit_command(user_input):  
            print("Goodbye!")
            text_to_speech("Goodbye!")
            break

        entities = extract_entities(user_input)
        if entities["task"] or entities["date"] or entities["time"]:
            assistant_response = process_entities(entities)
        else:
            assistant_response = get_nlp_response(user_input)

        print("Assistant:", assistant_response)
        log_interaction(user_input, assistant_response)  
        text_to_speech(assistant_response)  

# Run the voice assistant
if __name__ == "__main__":
    voice_assistant()

app = Flask(__name__)