import pyttsx3
import time

def speak(text):
    text_speech = pyttsx3.init()
    text_speech.setProperty('rate', 110)
    text_speech.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
    time.sleep(0.5)
    text_speech.say(text)
    text_speech.runAndWait()