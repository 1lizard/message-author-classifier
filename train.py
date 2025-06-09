import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re
import random
df = pd.read_csv("messages.csv")

texts = df["text"].tolist()
labels = df["label"].tolist()

def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()
tokenized_texts = [tokenize(text) for text in texts]

all_words = [word for tokens in tokenized_texts for word in tokens]
most_common = Counter(all_words).most_common(2000)
vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

def compute_idf(tokenized_texts, vocab): #computing inverse document frequency
    doc_count = len(tokenized_texts)
    idf = {}
    for word in vocab:
        docs_with_word = sum(1 for tokens in tokenized_texts if word in tokens)
        idf[word] = torch.log(torch.tensor(doc_count / (1 + docs_with_word), dtype=torch.float32))
    return idf

idf = compute_idf(tokenized_texts, vocab)

def vectorize_tfidf(tokens): #vectorising tokens
    vec = torch.zeros(len(vocab))
    counts = Counter(tokens)
    for word, count in counts.items():
        if word in vocab:
            vec[vocab[word]] = (count / len(tokens)) * idf[word]
    return vec

vectors = [vectorize_tfidf(tokens) for tokens in tokenized_texts]
label_tensors = [torch.tensor([label], dtype=torch.float32) for label in labels]

data = list(zip(vectors, label_tensors))
random.shuffle(data)

split = int(len(data) * 0.8)
train_data = data[:split]
test_data = data[split:]

class SimpleNN(nn.Module): #model setup
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))

        return x

model = SimpleNN(input_size=len(vocab))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs): #trainign
    total_loss = 0
    model.train()
    for x, y in train_data:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"\nEpoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for x, y in train_data:
            output = model(x)
            pred = (output > 0.5).float()
            correct_train += (pred == y).sum().item()
            total_train += 1
    train_acc = correct_train/total_train

    correct_test=0
    total_test=0
    with torch.no_grad():
        for x, y in test_data: #testing
            output = model(x)
            pred = (output > 0.5).float()
            correct_test += (pred == y).sum().item()
            total_test += 1
    test_acc = correct_test / total_test

    print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

while True: #predict with user inputted messages
    user_input = input("\nType a message (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    vec = vectorize_tfidf(tokenize(user_input))
    with torch.no_grad():
        output = model(vec)
        pred = (output > 0.5).float().item()

    author = "Me" if pred == 1.0 else "Other person"
    print(f"→ Predicted author: {author}")
