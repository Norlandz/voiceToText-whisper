import whisper
import sounddevice as sd
import numpy as np
import torch
import sys
import os
from pathlib import Path
from dataclasses import asdict
import speech_recognition as sr
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
# import keyboard
import pyperclip

# whisper_real_time/transcribe_demo.py at master · davabase/whisper_real_time
# https://github.com/davabase/whisper_real_time/blob/master/transcribe_demo.py

cwd = Path.cwd()
argv0 = sys.argv[0]
path_entryScript = Path(argv0).resolve()
path_projectDir = path_entryScript.parent
download_root = path_projectDir / "model"

has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

modelSizeAndName = "base.en"  # tiny base small medium large
language = "en"  # zh

initial_prompt = """
"""

sample_rate = 16000
energy_threshold = 1000  # Adjust for your mic's sensitivity
record_timeout = 3 # 2  # How long to record each chunk (seconds) // seems like a hard limit... 
phrase_timeout = 4.6 # 3  # Pause before considering it a new line (seconds)

stop_key = "q"

# 1. Load the Whisper model
model = whisper.load_model(name=modelSizeAndName, device=device, download_root=download_root.resolve().__fspath__())
oDecodingOptions = whisper.DecodingOptions(
    language=language,
)


# 2. Audio Capture Settings
# Setup audio source (Microphone)
recorder = sr.Recognizer()
recorder.energy_threshold = energy_threshold
recorder.dynamic_energy_threshold = False  # Important: Keep this off!

source = sr.Microphone(sample_rate=sample_rate)  # keep sample_rate fixed

with source:
    recorder.adjust_for_ambient_noise(source)  # Calibrate noise level

data_queue = Queue()


def record_callback(_, audio: sr.AudioData) -> None:
    """Threaded callback function to receive audio data."""
    data = audio.get_raw_data()
    data_queue.put(data)


recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

print("Model loaded.\n")
transcription = [""]
phrase_time = None


while True:
    try:
        now = datetime.utcnow()
        if not data_queue.empty():
            phrase_complete = False

            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = True

            phrase_time = now

            # Combine audio data from queue
            audio_data = b"".join(data_queue.queue)
            data_queue.queue.clear()

            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio
            result = model.transcribe(audio_np, fp16=has_cuda)
            text: str = result["text"]  # type: ignore
            text = text.strip()

            # if phrase_complete:
            #     transcription.append(text)
            # else:
            #     transcription[-1] = text
            # bit buggy
            transcription.append(text)
            last_line = transcription[-1]
            pyperclip.copy(last_line)
            print(last_line)

            # # Clear and reprint transcription
            # os.system("cls" if os.name == "nt" else "clear")  # Clear screen (Windows/Linux)
            # for line in transcription:
            #     print(line)
            # print("", end="", flush=True)

            # not_working
            # if keyboard.is_pressed(stop_key):  # Check for key press
            #     print(f"'{stop_key}' pressed. Stopping transcription.")
            #     break  # Exit the loop
        else:
            sleep(0.25)  # Reduce CPU usage

    except KeyboardInterrupt:
        break
        # print("@messy KeyboardInterrupt is captured for convenience.")
        # num_last_lines = 3
        # if len(transcription) >= num_last_lines:
        #     last_lines = " ".join(transcription[-num_last_lines:])
        # else:
        #     last_lines = " ".join(transcription)
        # pyperclip.copy(last_lines)

print("\n\nTranscription:")
for line in transcription:
    print(line)
