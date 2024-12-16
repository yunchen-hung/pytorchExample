# imports 
import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset
    
# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.labels = [ex.label for ex in self.examples]

        # Initialize the WordEmbedding Class
        self.word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        
        # Get indices for each sentence
        self.labels = torch.tensor(self.labels)

        self.word_indices = [torch.tensor([self.word_embeddings.word_indexer.index_of(word) 
                                       if self.word_embeddings.word_indexer.index_of(word) != -1 
                                       else self.word_embeddings.word_indexer.index_of("UNK")
                                       for word in ex.words]) for ex in self.examples]
        


    def __len__(self):
        return self.word_embeddings.get_embedding_length()

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.word_indices[idx], self.labels[idx]



# DAN
class DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # get the embedding layer for neural network
        # Initialize the WordEmbedding Class
        self.word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        self.embeddings = self.word_embeddings.get_initialized_embedding_layer()

        # Hidden Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_indices):
        word_indices = word_indices.long()
        word_indices = self.embeddings(word_indices)
        word_indices = torch.mean(word_indices, dim=0)
        word_indices = F.relu(self.fc1(word_indices))
        word_indices = self.fc2(word_indices)
        word_indices = self.log_softmax(word_indices)
        return word_indices
    

class DAN_embed(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # get the embedding layer for neural network
        # Initialize the Embedding 
        self.embeddings = nn.Embedding(14923, 300)

        # Hidden Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_indices):
        word_indices = word_indices.long()
        word_indices = self.embeddings(word_indices)
        word_indices = torch.mean(word_indices, dim=0)
        word_indices = F.relu(self.fc1(word_indices))
        word_indices = self.fc2(word_indices)
        word_indices = self.log_softmax(word_indices)
        return word_indices