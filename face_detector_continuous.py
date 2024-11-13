import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import numpy as np


workers = 0 if os.name == 'nt' else 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

try:
    loaded_dict = torch.load("face_embeddings.pt")
except:
    loaded_dict = {}

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera")

img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to take frame")
        break

    

    k = cv2.waitKey(1)
    # ESC pressed
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    bounding_box,prob = mtcnn.detect(frame)
    if(bounding_box is not None):
        bounding_box = bounding_box[0]
        bounding_box = bounding_box.astype(int)
        start_x,start_y,width,height = bounding_box
        prob = prob[0]

        cv2.rectangle(frame, (start_x, start_y), (width, height), (0, 255, 0), 2)
    
    x_aligned = mtcnn(frame)
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
        # print(distances)
        if distances[minIndex] > 1:
            print("Unknown user")
            cv2.putText(frame, "Unknown", (width, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            name = list(loaded_dict.keys())[minIndex]
            print("Face Recognized :", name)
            cv2.putText(frame, name, (width, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)


    else:
        print("No face detected")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Camera", frame)

cam.release()
cv2.destroyAllWindows()
