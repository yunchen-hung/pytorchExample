# models.py

import torch
from torch import nn
import torch.nn.functional as F
#from sklearn.feature_extraction.text import CountVectorizer
#from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, DAN_embed
from torch.nn.utils.rnn import pad_sequence


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_train_loss = []
    all_test_accuracy = []
    all_test_loss = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
            print(f'Epoch #{epoch + 1}: train loss {train_loss:.3f}, dev loss {test_loss:.3f}')

    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_dataBOW = SentimentDatasetBOW("data/train.txt")
    dev_dataBOW = SentimentDatasetBOW("data/dev.txt")
    train_loaderBOW = DataLoader(train_dataBOW, batch_size=16, shuffle=True)
    test_loaderBOW = DataLoader(dev_dataBOW, batch_size=16, shuffle=False)

    def collate(data):
            indices = [item[0] for item in data]
            labels = [item[1] for item in data]
            indices = pad_sequence(indices, padding_value=0)
            labels = torch.stack(labels) 
            return (indices, labels)
        
    train_dataDAN = SentimentDatasetDAN("data/train.txt")
    dev_dataDAN = SentimentDatasetDAN("data/dev.txt")
    
    train_loaderDAN = DataLoader(train_dataDAN, batch_size=16, shuffle=True,collate_fn=collate)
    test_loaderDAN = DataLoader(dev_dataDAN, batch_size=16, shuffle=False,collate_fn=collate)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
         # Load dataset
        start_time = time.time()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy, \
            nn2_train_loss, nn2_test_loss = experiment(NN2BOW(input_size=512, hidden_size=100), train_loaderBOW, test_loaderBOW)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy, \
             nn3_train_loss, nn3_test_loss = experiment(NN3BOW(input_size=512, hidden_size=100), train_loaderBOW, test_loaderBOW)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        start_time = time.time()
        print('\n2 layers:') 
        dan_train_accuracy, dan_test_accuracy, \
            dan_train_loss, dan_test_loss = experiment(DAN(input_size=300, hidden_size=100), 
                                            train_loaderDAN, test_loaderDAN)
        print('\n2 layers, own embedding:') 
        dan_train_accuracy_embed, dan_test_accuracy_embed, \
            dan_train_loss_embed, dan_test_loss_embed = experiment(DAN_embed(input_size=300, hidden_size=100), 
                                                        train_loaderDAN, test_loaderDAN)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='2 layers')
        plt.plot(dan_train_accuracy_embed, label='2 layers embedding')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracyDAN.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='2 layers')
        plt.plot(dan_test_accuracy_embed, label='2 layers, own embedding')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracyDAN.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # Plot the training loss
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_loss, label='2 layers')
        plt.plot(dan_train_loss_embed, label='2 layers, own embedding')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Training Loss for 2, own Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_loss_file = 'train_lossDAN.png'
        plt.savefig(training_loss_file)
        print(f"\n\nTraining loss plot saved as {training_loss_file}")

        # Plot the testing loss
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_loss, label='2 layers')
        plt.plot(dan_test_loss_embed, label='2 layers, own embedding')
        plt.xlabel('Epochs')
        plt.ylabel('Testing Loss')
        plt.title('Testing Loss for 2, own Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        testing_loss_file = 'test_lossDAN.png'
        plt.savefig(testing_loss_file)
        print(f"\n\nTesting loss plot saved as {testing_loss_file}")

if __name__ == "__main__":
    main()

