import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import numpy as np
from pymongo import MongoClient
import datetime
import pytz
import text_speech


workers = 0 if os.name == 'nt' else 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



cam = cv2.VideoCapture(0)
ret, frame = cam.read()
if not ret:
        print("Failed to take frame")

name = ""

url = "mongodb://localhost:27017/"
client = MongoClient(url)
db = client.test
relatives = db.relatives


try:
    loaded_dict = torch.load("face_embeddings.pt")
except:
    loaded_dict = {}

def updateEmbeddings():
    global loaded_dict
    try:
        loaded_dict = torch.load("face_embeddings.pt")
    except:
        loaded_dict = {}
        
def take_photo(intent):

    
    ret, frame = cam.read()
    if not ret:
        print("Failed to take frame")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    x_aligned, prob = mtcnn(frame, return_prob=True)
    if x_aligned is not None:
        # print('Face detected with probability: {:8f}'.format(prob))
        aligned = []
        aligned.append(x_aligned)

        # Generate embedding
        aligned = torch.stack(aligned).to(device)
        embeddings = resnet(aligned).detach().cpu()
        distances = []
        # Compare with existing embeddings (Calculate distance)
        for x in loaded_dict:
            for y in loaded_dict[x]:
                tempDistance = []
                tempDistance.append((embeddings - y).norm().item())
            tempDistance = np.array(tempDistance)
            distances.append(tempDistance.mean())

        distances = np.array(distances)

        # Find the closest match

        minIndex = np.argmin(distances)
        if distances[minIndex] > 0.9:
            print("Unknown face")
            text_speech.speak("Sorry, I don't know him")

        else:
            name = list(loaded_dict.keys())[minIndex]
            print("Face Recognized :", name)
            from bson.objectid import ObjectId
            result = relatives.find_one({"_id" : ObjectId(name)})
            

            if result:

                if result['gender'].lower() == 'male':
                    pronoun = "He"
                elif result['gender'].lower() == 'female':   
                    pronoun = "She"
                else:
                    pronoun = "It"
                
                if intent == 'name':
                    text_speech.speak(pronoun + "is" + result[intent])
                    relatives.update_one({"name" : result['name']}, {"$set" : {"last_meet" : datetime.datetime.utcnow()}})
                
                elif intent == 'address':
                    text_speech.speak(pronoun + "lives in " + result[intent])
                    relatives.update_one({"name" : result['name']}, {"$set" : {"last_meet" : datetime.datetime.utcnow()}})

                elif intent == 'last_meet':
                    if 'last_meet' in result:
                        ist = pytz.timezone('Asia/Kolkata')
                        ist_now = result[intent].replace(tzinfo=pytz.utc).astimezone(ist)
                        # print("Last time you met was on", ist_now.strftime("%d %B %Y at %H:%M"))
                        text_speech.speak("Last time you met was on " + ist_now.strftime("%d %B %Y at %H:%M"))
                    else:
                        text_speech.speak("I don't remember when you met him the last time")
                
                elif intent == 'relationship':
                    text_speech.speak(pronoun + "is your " + result[intent])
                    relatives.update_one({"name" : result['name']}, {"$set" : {"last_meet" : datetime.datetime.utcnow()}})
                  

    else:
        print("No face detected")
        text_speech.speak("Sorry, I didn't see any face")

def releaseCam():
    cam.release()
