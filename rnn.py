import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # inputs shape: (seq_len, batch, input_dim)
        _, hidden = self.rnn(inputs)  # hidden shape: (num_layers, batch, hidden_dim)
        
        
        last_hidden = hidden[-1]  # shape: (batch, hidden_dim)
        
        
        output = self.W(last_hidden)  # shape: (batch, 5)
        
        
        predicted_vector = self.softmax(output)
        
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    # Load word embeddings
    word_embedding = pickle.load(open(r'/content/drive/MyDrive/NLP2/RNN/word_embedding.pkl', 'rb'))

    # Initialize model and move to device
    model = RNN(50, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None

            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]
                
                vectors = torch.tensor(vectors, device=device).view(len(vectors), 1, -1)
                gold_label = torch.tensor([gold_label], device=device)

                output = model(vectors)

                example_loss = model.compute_Loss(output.view(1, -1), gold_label)

                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label.item())
                total += 1

                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        print(loss_total / loss_count)
        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")
        trainning_accuracy = correct / total

        # Validation
        model.eval()
        correct = 0
        total = 0
        print(f"Validation started for epoch {epoch + 1}")

        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]

                vectors = torch.tensor(vectors, device=device).view(len(vectors), 1, -1)
                gold_label = torch.tensor([gold_label], device=device)

                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label.item())
                total += 1

        validation_accuracy = correct / total
        print(f"Validation completed for epoch {epoch + 1}")
        print(f"Validation accuracy for epoch {epoch + 1}: {validation_accuracy}")

        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1



    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
