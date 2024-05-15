import tkinter as tk
from tkinter import ttk
from transformers import pipeline
import threading

text_to_speech_generator = pipeline("text-to-speech", model="facebook/musicgen-small")

def generate_audio():
    prompt = entry.get()
    thread = threading.Thread(target=generate_audio_thread, args=(prompt,))
    thread.start()

def generate_audio_thread(prompt):
    progress_bar['value'] = 0
    root.update_idletasks()

    # Simuler la progression 
    total_steps = 10  
    for i in range(total_steps):
        results = text_to_speech_generator(prompt) 
        progress_bar['value'] += 100 / total_steps  
        root.update_idletasks()  

    text_output.after(0, text_output.delete, "1.0", tk.END)
    text_output.after(0, text_output.insert, tk.END, "Audio generated successfully (check your audio output).")
    progress_bar['value'] = 0  

# Création de la fenêtre principale
root = tk.Tk()
root.title("Music Generator Interface")
entry = tk.Entry(root, width=50)
entry.pack(padx=20, pady=20)
generate_button = tk.Button(root, text="Generate Audio", command=generate_audio)
generate_button.pack(pady=10)

progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress_bar.pack(pady=20)

text_output = tk.Text(root, height=10, width=50)
text_output.pack(padx=20, pady=20)

root.mainloop()
