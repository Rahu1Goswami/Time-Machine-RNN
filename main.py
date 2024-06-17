from getdata import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def main():
    file = Data("book.txt")
    corpus, vocab = file.build()
    data = Book(corpus, vocab)
    print(torch.argmax(data[:5], dim=1))
    print(corpus[:5])


class ScrRNN(nn.Module):
    def __init__(self, num_inp, num_hidden):
        super().__init__()
        self.rnn = nn.RNN(num_inp, num_hidden, batch_first=True) # Input should be N x L x H_in
        self.fc = nn.Linear(num_hidden, num_inp)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(x)

class Book(Dataset):
    def __init__(self, corpus, vocab):
        self.corpus, self.vocab = torch.tensor(corpus), vocab
        self.tokens = F.one_hot(self.corpus, num_classes=len(self.vocab))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]






if __name__ == '__main__':
    main()
