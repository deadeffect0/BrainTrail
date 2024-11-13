import pyaudio
import keyboard
import time
from vosk import Model, KaldiRecognizer
import intent_classifier as ic
import json
import text_speech

model = Model("vosk-model-small-en-in-0.4")
recognizer = KaldiRecognizer(model, 16000)



def recognize_speech():
    mic = pyaudio.PyAudio()
    print("Press spacebar to start speaking")
    while True:
        if keyboard.is_pressed('space'):
            print("Speak Now...")
            stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
            stream.start_stream()
            buffer = b''
            last_speech_time = time.time()
            while True:
                data = stream.read(4096)
                buffer += data
                current_time = time.time()
                if current_time - last_speech_time > 6:
                    break

            stream.stop_stream()
            stream.close()
            if len(buffer) > 0:
                if recognizer.AcceptWaveform(buffer):
                    result = recognizer.Result()
                    result_dict = json.loads(result)
                    print(result_dict["text"])
                    intent = ic.intent_classify(result_dict["text"])
                    if intent == "GoodBye":
                        text_speech.speak("GoodBye")
                        mic.terminate()
                        break
                else:
                    print("Sorry, I did not get that")
            else:
                print("Sorry, I did not get that : 0 length buffer")

