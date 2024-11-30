import tkinter as tk
from tkinter import messagebox
from threading import Thread

class VoiceAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Assistant")
        self.root.geometry("500x400")
        
        # Title Label
        self.title_label = tk.Label(root, text="Voice Assistant", font=("Arial", 20))
        self.title_label.pack(pady=10)
        
        # Transcription Box
        self.transcription_label = tk.Label(root, text="Transcription:", font=("Arial", 14))
        self.transcription_label.pack(anchor="w", padx=10)
        
        self.transcription_text = tk.Text(root, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.transcription_text.pack(padx=10, pady=5, fill=tk.BOTH)
        
        # Assistant Response Box
        self.response_label = tk.Label(root, text="Assistant Response:", font=("Arial", 14))
        self.response_label.pack(anchor="w", padx=10)
        
        self.response_text = tk.Text(root, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.response_text.pack(padx=10, pady=5, fill=tk.BOTH)
        
        # Record Button
        self.record_button = tk.Button(root, text="Record", font=("Arial", 14), command=self.start_voice_assistant)
        self.record_button.pack(pady=20)
        
        # Exit Button
        self.exit_button = tk.Button(root, text="Exit", font=("Arial", 14), command=self.exit_app)
        self.exit_button.pack(pady=10)

    def update_text(self, widget, text):
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    def start_voice_assistant(self):
        def run_assistant():
            # Record audio and transcribe
            record_audio()
            user_input = transcribe_audio()
            self.update_text(self.transcription_text, user_input)
            
            if is_exit_command(user_input):
                self.update_text(self.response_text, "Goodbye!")
                self.root.quit()
                return
            
            # Process user input
            entities = extract_entities(user_input)
            if entities["task"] or entities["date"] or entities["time"]:
                assistant_response = process_entities(entities)
            else:
                assistant_response = get_nlp_response(user_input)
            
            self.update_text(self.response_text, assistant_response)
            text_to_speech(assistant_response)

        # Run the assistant in a separate thread to avoid freezing the UI
        assistant_thread = Thread(target=run_assistant)
        assistant_thread.start()

    def exit_app(self):
        self.root.quit()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistantApp(root)
    root.mainloop()