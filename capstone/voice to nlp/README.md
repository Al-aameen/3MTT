# Voice Assistant with Google Cloud and Gemini Integration

This is a voice assistant application that uses Google Cloud's Speech-to-Text, Gemini (Google's generative AI), and various other APIs to provide a smart assistant experience. The assistant listens for user input via audio, processes it, and responds with text-to-speech. It can handle tasks such as setting reminders, checking the weather, and performing web searches.

## Features

- **Voice Interaction**: The assistant listens to your voice, transcribes it, processes it, and responds verbally.
- **Reminder System**: Set and save reminders with a specified date and time.
- **Weather Queries**: Get current weather information for any city.
- **Web Search**: Perform Google searches and retrieve the top results.
- **Text-to-Speech (TTS)**: Responds audibly to user input with synthesized speech.
- **Generative AI**: Utilizes Google's Gemini API to generate intelligent responses.

## Requirements

- Python 3.x
- Install the necessary dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

- **API Keys**: You need API keys for Google Cloud (Speech-to-Text, Gemini) and WeatherAPI.

## Setup

1. **Google Cloud Setup**:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
   - Enable the Speech-to-Text and Gemini APIs.
   - Download your Google Cloud credentials file (usually a `.json` file).
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your credentials file.

2. **WeatherAPI**:
   - Sign up at [WeatherAPI](https://www.weatherapi.com/) and get an API key.

3. **Create `config.json`**:
   - Store your API keys in a `config.json` file in the project directory:
   
     ```json
     {
         "gemini_api_key": "your_gemini_api_key",
         "google_cloud_credentials": "path/to/your/google_credentials.json",
         "weather_api_key": "your_weather_api_key"
     }
     ```

4. **Install Dependencies**:
   The project uses the following libraries:
   - `pyaudio` (for recording audio)
   - `google-cloud-speech` (for transcription)
   - `pyttsx3` (for text-to-speech)
   - `spacy` (for natural language processing)
   - `requests` (for API calls)
   - `google-generativeai` (for AI-based responses)
   - `numpy`, `wave`, `logging`, `time` for audio handling and logging

   To install the dependencies, create a `requirements.txt` file with the following content:

   ```
   pyaudio
   google-cloud-speech
   pyttsx3
   spacy
   google-generativeai
   requests
   numpy
   ```

   Then install the dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

   Also, you will need to download the spaCy language model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Run the Assistant**:
   Once everything is set up, you can run the assistant using:

   ```bash
   python assistant.py
   ```

## Functionality

### 1. **Audio Recording**:
   The assistant listens for your voice commands. The audio is recorded and stored as `audio.wav`.

### 2. **Speech-to-Text**:
   The recorded audio is transcribed using Google's Speech-to-Text API.

### 3. **Task Processing**:
   The assistant processes the transcribed text to identify tasks, dates, and times using spaCy's NLP capabilities.

### 4. **Action Execution**:
   Based on the recognized entities, the assistant can:
   - Set reminders
   - Get weather information
   - Perform web searches
   - Respond with text-to-speech using pyttsx3

### 5. **Generative AI Responses**:
   If the input doesn't match any predefined tasks, the assistant uses Gemini to generate a response.

## Logging

The assistant logs all interactions (both user input and assistant responses) in a file called `assistant.log`. This helps track the history of conversations for debugging and improvement.

## Exit Commands

To exit the assistant, simply say one of the following commands:
- "exit"
- "quit"
- "stop"
- "goodbye"

## Notes

- Ensure the `config.json` file is **not committed to any public repositories** as it contains sensitive information such as API keys. You can add `config.json` to `.gitignore` to avoid accidentally pushing it to a repository.
- Make sure that all the required API keys are properly set in `config.json`.