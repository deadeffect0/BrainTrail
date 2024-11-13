import speech_recognition as sr
import keyboard
import intent_classifier as ic
recognizer = sr.Recognizer()

def recognize_speech():
    print("Press spacebar to start speaking")
    while True:
        try:
            if keyboard.is_pressed(' '):  # check if spacebar is pressed
                with sr.Microphone() as source:
                    print("Speak now ......... ")
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = recognizer.listen(source)
                    text = recognizer.recognize_google(audio)
                    text = text.lower()
                    print("You said: {}".format(text))
                    intent = ic.intent_classify(text)
                    if intent == "GoodBye":
                        break
        except:
            print("Sorry, I did not get that")
