import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from nltk.stem import SnowballStemmer

# Load the intent.json file
with open('intents.json') as file:
    data = json.load(file)


patterns = []
labels = []


tag_dict = {}
counter = 0


for sample in data['samples']:
    tag = sample['tags']
    if tag not in tag_dict:
        tag_dict[tag] = counter
        counter += 1
    tag_label = tag_dict[tag]

    for pattern in sample['patterns']:
        patterns.append(pattern)
        labels.append(tag_label)

print(patterns)
print(labels)


patterns = np.array(patterns)
labels = np.array(labels)


train_message, test_message, train_labels, test_labels = train_test_split(
    patterns, labels, shuffle=True, test_size=0.2, random_state=321)


class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).clone().detach().long()
        return item

    def __len__(self):
        return len(self.labels)


model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

ignore_words = ['?', '!', '.', ',','is','are','am','was','were','do','does','did','can','could','may','might','must','shall','should','will','would','have','has','had','a','an','the','of','in','on','at','to','for','from','by','with','and','or','but','if','then','else','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
stemmer = SnowballStemmer('english')

for i in range(len(train_message)):
    words = train_message[i].lower().split()
    filtered_words = [word for word in words if word not in ignore_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    train_message[i] = ' '.join(stemmed_words)

# Loop over the test_message list and remove stopwords
for i in range(len(test_message)):
    words = test_message[i].lower().split()
    filtered_words = [word for word in words if word not in ignore_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    test_message[i] = ' '.join(stemmed_words)

train_message = train_message.tolist()
test_message = test_message.tolist()



train_encodings = tokenizer(
    train_message, truncation=True, padding=True, max_length=512)
test_encondings = tokenizer(
    test_message, truncation=True, padding=True, max_length=512)


train_dataset = IntentDataset(train_encodings, train_labels)
test_dataset = IntentDataset(test_encondings, test_labels)

traning_args = TrainingArguments(
    output_dir='./resultsss',          # output directory
    num_train_epochs=20,             # total # of training epochs
    per_device_train_batch_size=9,   # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",

)

model = DistilBertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(tag_dict))

trainer = Trainer(
    # the instantiated 🤗 Transformers model to be trained
    model=model,
    args=traning_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,             # evaluation dataset
)

trainer.train()
trainer.save_model("intent_cf_modellll")
