from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os


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
    loaded_dict ={}


def collate_fn(x):
    return x[0]

def get_embeddings():
    dataset = datasets.ImageFolder(
        'images')

    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []
    names = []

    existingUser = []

    for keys in loaded_dict:
        existingUser.append(keys)

    for x, y in loader:
        if dataset.idx_to_class[y] in existingUser:
            # print("User already exists")
            pass
        else:
            x_aligned, prob = mtcnn(x, return_prob=True)
            if x_aligned is not None:
                print('Face detected with probability: {:8f}'.format(prob))
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

    if(len(aligned)!= 0): 
        aligned = torch.stack(aligned).to(device)
        embeddings = resnet(aligned).detach().cpu()
        embeddings_dict = {}
        for name in set(names):
            embeddings_dict[name] = []
            for i in range(len(names)):
                if names[i] == name:
                    embeddings_dict[name].append(embeddings[i])

        loaded_dict.update(embeddings_dict)

        torch.save(loaded_dict, "face_embeddings.pt")
        print("New embeddings added")
