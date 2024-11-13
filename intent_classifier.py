from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import face_detector
import text_speech
import nltk
from nltk.stem import SnowballStemmer


# Load the trained model
detected_intent = ""
tags = ['Greetings', 'NameInquiry', 'AddressInquiry',
        'LastmeetingInquiry', 'RelationshipInquiry', 'GoodBye']
model = DistilBertForSequenceClassification.from_pretrained(
    "intent_cf_model")

# Define the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
ignore_words = ['?', '!', '.', ',','is','are','am','was','were','do','does','did','can','could','may','might','must','shall','should','will','would','have','has','had','a','an','the','of','in','on','at','to','for','from','by','with','and','or','but','if','then','else','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
stemmer = SnowballStemmer('english')
model.eval()

def intent_classify(text):
    with torch.no_grad():
            if text == "exit":
                face_detector.releaseCam()
            
            words = text.lower().split()
            filtered_words = [word for word in words if word not in ignore_words]
            stemmed_words = [stemmer.stem(word) for word in filtered_words]
            text = ' '.join(stemmed_words)


            inputs = tokenizer(text, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")

            # Pass the tokenized text to the model to get the predictions
            outputs = model(**inputs)
            predictions = F.softmax(outputs['logits'], dim=1)
            print(predictions)
            max_score, max_index = torch.max(predictions[0], dim=0)

            if max_score > 0.90:
                detected_intent = tags[max_index]
                print(tags[max_index])

                if detected_intent == "NameInquiry":
                    face_detector.take_photo('name')
                elif detected_intent == "AddressInquiry":
                    face_detector.take_photo('address')
                elif detected_intent == "LastmeetingInquiry":
                    face_detector.take_photo('last_meet')
                elif detected_intent == "RelationshipInquiry":
                    face_detector.take_photo('relationship')
                elif detected_intent == "GoodBye":
                    return detected_intent
                elif detected_intent == "Greetings":
                    text_speech.speak("Hello, I am here to help you.")

            else:
                print("I am not sure what you are asking.")
                text_speech.speak("I am not sure what you are asking.")

def text_mode():
    while True:
        text = input("Enter your query: ")
        intent = intent_classify(text)
        if intent == "GoodBye":
            text_speech.speak("Goodbye")
            break
