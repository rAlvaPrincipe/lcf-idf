
from torch.nn import Linear, Module, MSELoss, ReLU
from src.classifier.early_stopping import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from torch import nn
import torch
from tqdm import tqdm
from src.classifier.dataloader import AutoencoderDataset
from src.classifier.BERT_clf import Layer


class Autoencoder (Module):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            Layer(768, 600, ReLU),
            Layer(600, 500, ReLU),
            Linear(500, 100)
        )
        self.decoder = nn.Sequential(
            Linear(100, 500),
            Layer(500, 600, ReLU),
            Linear(600, 768)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def train_loop(self, epochs, train_loader, val_loader):
        early_stopping = EarlyStopping(patience=30, verbose=True, path=self.base_dir + "/autoencoder.pt")
        criterion = MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        for e in tqdm(range(epochs)):
            train_epoch_loss = 0
            val_epoch_loss = 0
            self.train()
            for batch in train_loader:
                embeddings = batch.to("cuda")
                optimizer.zero_grad()
                reconstructed = self(embeddings)#.to("cuda")
                loss = criterion(embeddings, reconstructed)#.to("cuda")
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss
            train_loss = train_epoch_loss/len(train_loader)
            #print("Epoch: {}/{}.. ".format(e+1, epochs),"Training Loss: {:.3f}.. ".format(train_loss))
            #print("Epoch: {}/{}.. ".format(e+1, epochs),"Training Loss: {:.3f}.. ".format(train_loss),"Validation Loss: {:.3f}.. ".format(val_loss))

            self.eval()
            with torch.no_grad():
                for batch in val_loader:
                    embeddings = batch.to("cuda")
                    reconstructed = self(embeddings)
                    val_epoch_loss += loss

            val_loss = val_epoch_loss/len(val_loader)
            print("Epoch: {}/{}.. ".format(e+1, epochs),"Training Loss: {:.3f}.. ".format(train_loss),"Validation Loss: {:.3f}.. ".format(val_loss))

            early_stopping(val_loss, self)             
            if early_stopping.early_stop:
                print("Early stopping")
                break
            

    def fit(self, train_embeddings, val_embeddings):
        print("--> autoencoder")
        data = AutoencoderDataset(train_embeddings)
        train_loader = DataLoader(data, batch_size=512, shuffle=True)
        data = AutoencoderDataset(val_embeddings)
        val_loader = DataLoader(data, batch_size=512, shuffle=True)
        self.train_loop(1000, train_loader, val_loader)
        print("------------- getting reduced embeddings from the training set")
        return self.transform(train_embeddings)


    # we make inference in cpu since datasets may too large to fir in the GPU.
    # another option might be to make predictions of chunks in GPU and for each chunk move it to cpu and iterate
    def transform(self, embeddings):
        self.eval()
        with torch.no_grad():
            return self.encoder.to("cpu")(embeddings)
    
    def save(self, path):
        torch.save(self.state_dict(), path )

    def load(self, path):
        self.load_state_dict(torch.load(path ))