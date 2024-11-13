import os
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from pymongo import MongoClient
import embedding_gen
import intent_classifier
import offline_speech_rec
import face_detector


relative_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "relationship", "address", "gender"],
            "properties": {
                "name": {
                    "bsonType": "string",
                    "description": "name is required"
                },
                "relationship": {
                    "bsonType": "string", "description": "relationship is required"
                },
                "last_meet": {
                    "bsonType": "date",
                    "description": "last_meet is required"
                },
                "address": {
                    "bsonType": "string",
                    "description": "address is required"

                },
                "gender" : {
                    "bsonType" : "string",
                    "description" : "gender is required"
                }
            }
        }
    }

def insert_relative(name, address, relationship, gender):
    objectid = ""
    url = "mongodb://localhost:27017/"
    client = MongoClient(url)
    db = client.test

    try:
        db.create_collection("relatives")
    except Exception as e:
        print(e)

    db.command("collMod", "relatives", validator=relative_validator)

    relatives = db.relatives

    relative = {"name": name, "relationship": relationship, "address": address, "gender" : gender}

    try:
        result = relatives.insert_one(relative)
        objectid = result.inserted_id
    except Exception as e:
        print(e)
    
    return objectid


def copy_images(destination):
    
    # Open a file dialog box to select the image files
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    files = askopenfilenames(title='Select image files', filetypes=[('Image files', '*.jpg;*.jpeg;*.png;*.gif')])

    # Copy the selected images to the destination folder
    if not os.path.exists(destination):
        os.makedirs(destination)
    num_copied = 0
    for file in files:
        filename = os.path.basename(file)
        shutil.copy(file, os.path.join(destination, filename))
        num_copied += 1

    print(f'{num_copied} images have been copied to {destination}.')

if __name__ == '__main__':

    while True:
        print("Assistance Bot Welcomes You!!")
        print("Please select the option you want to perform")
        print("1. Add a new relative")
        print("2. Start recognition")
        print("3. Exit")
        options = input("Input your Choice:  ")

        if options == "1":
            print("Add a new relative")
            print("Please enter the following details")
            name = input("Name: ")
            address = input("Address: ")
            relationship = input("Relationship: ")
            gender = input("Gender: ")
            objectid = insert_relative(name,address,relationship,gender)
            print("Please upload the images of the relative")
            copy_images(f"images/{objectid}")
            print("Generating embeddings")
            embedding_gen.get_embeddings()
            face_detector.updateEmbeddings()

        elif options == "2":
            print("Choose mode: ")
            print("1. Voice")
            print("2. Text")
            mode = input("Input your Choice:  ")

            if mode == "1":
                print("Voice mode activated")
                offline_speech_rec.recognize_speech()
                
            
            elif mode == "2":
                print("Text mode activated")
                intent_classifier.text_mode()
        
        elif options == "3":
            print("Exiting")
            break

        else :
            print("Invalid Choice")

    


