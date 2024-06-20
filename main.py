from getdata import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

device = "cuda"

def main():
    file = Data("book.txt")
    corpus, vocab = file.build()
    data = Book(corpus, vocab)
    # print(torch.argmax(data[0][0], dim=1))
    dataloader = DataLoader(data, batch_size=1024*4, shuffle=True)
    # X = N x L x H
    #   = 64 x 30 x 28
    # y = N x H
    #   = 64 x 28
    model = ScrRNN(len(vocab), 128).cuda()

    try:
        model.load_state_dict(torch.load("./model.pt"))
    except:
        pass

    num_epochs = 700
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(f"runs/trying-tensorboard")
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        sum = 0
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            sum += 1

        print(f"Epoch: {epoch+1}, loss: {epoch_loss}")
        writer.add_scalar("Training Loss", epoch_loss, global_step=epoch)

        if (epoch == 0) or (losses[-1] > epoch_loss.item()):
            torch.save(model.state_dict(), "./model.pt")


        losses.append(epoch_loss.item())

    txt = input("Prefix: ")
    print(gen_text(model, file, vocab, txt, 30))

    #sns.relplot(losses, kind="line")
    #plt.show()


def gen_text(model, file, vocab, prefix, length):
    prefix = file._preprocess(prefix)
    gen_text = prefix
    for i in range(length):
        prefix = torch.tensor(vocab[file._tokenize(gen_text)])
        prefix = F.one_hot(prefix, num_classes=len(vocab)).to(dtype=torch.float32, device=device)
        txt = model(prefix)
        txt = txt[-1, :]
        txt = vocab.to_tokens(torch.argmax(txt))
        gen_text += txt
    return gen_text

class ScrRNN(nn.Module):
    def __init__(self, num_inp, num_hidden):
        super().__init__()
        self.rnn = nn.RNN(num_inp, num_hidden, batch_first=True) # Input should be N x L x H_in
        self.fc = nn.Linear(num_hidden, num_inp)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

class Book(Dataset):
    def __init__(self, corpus, vocab, seq_len=30):
        self.corpus, self.vocab = torch.tensor(corpus), vocab
        self.seq_len = seq_len
        self.tokens = F.one_hot(self.corpus, num_classes=len(self.vocab))
        self.tokens = self.tokens.to(dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        return self.tokens[idx:idx+self.seq_len], self.tokens[idx+1:idx+self.seq_len+1]



if __name__ == '__main__':
    main()
